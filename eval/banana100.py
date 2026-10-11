"""Fetch a few chain steps from Banana100 without downloading the 218 GB parquet.

What we found (also written to eval/BANANA100_NOTES.md):
  * The dataset is one parquet file, 28,000 rows in 14 row groups of 2,000 rows.
  * Each row group holds 4 chains x 5 runs x 100 steps, in order.
  * The ``output_image`` column is dictionary-encoded: the first ~2 GB of images of every row group
    sit in ONE Snappy-compressed dictionary page. pyarrow has to download and hold that whole page
    to return a single row, which does not fit in memory on a laptop.

So this module reads the page by hand: it range-requests the start of the dictionary page, parses
the Thrift page header, and decodes the raw Snappy stream only as far as the first N images. That
gives the leading steps of the first chain of a row group for a few hundred megabytes.

    python -m eval.banana100 inspect
    python -m eval.banana100 fetch --group 10 --steps 20
"""

from __future__ import annotations

import argparse
import io
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, BinaryIO

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download
from PIL import Image

REPO = "kenantang/Banana100"
PARQUET = f"datasets/{REPO}/banana100.parquet"
CACHE = Path(__file__).resolve().parent / "cache" / "banana100"
META_COLUMNS = ["model", "image_name", "hyperparameter", "run_idx", "step"]
Image.MAX_IMAGE_PIXELS = 120_000_000


def open_parquet() -> pq.ParquetFile:
    fs = HfFileSystem()
    return pq.ParquetFile(fs.open(PARQUET, "rb", block_size=8 * 1024 * 1024))


def group_summaries(pf: pq.ParquetFile) -> list[dict[str, Any]]:
    names = pf.schema_arrow.names
    out = []
    for i in range(pf.metadata.num_row_groups):
        rg = pf.metadata.row_group(i)
        row: dict[str, Any] = {
            "group": i,
            "rows": rg.num_rows,
            "gb": round(rg.total_byte_size / 1e9, 2),
        }
        for c in range(rg.num_columns):
            col = rg.column(c)
            if (
                names[c] in META_COLUMNS
                and col.statistics is not None
                and col.statistics.has_min_max
            ):
                row[names[c]] = [col.statistics.min, col.statistics.max]
        out.append(row)
    return out


def leading_chain(pf: pq.ParquetFile, group: int) -> dict[str, Any]:
    """Model, image, chain and run of the first rows of a row group (small columns only)."""
    meta = pf.read_row_group(group, columns=META_COLUMNS).slice(0, 100).to_pylist()
    first = meta[0]
    steps = [m["step"] for m in meta if all(m[k] == first[k] for k in META_COLUMNS[:4])]
    return {**{k: first[k] for k in META_COLUMNS[:4]}, "steps": steps}


def inspect() -> None:
    pf = open_parquet()
    print("schema:", pf.schema_arrow)
    print("rows:", pf.metadata.num_rows, "row groups:", pf.metadata.num_row_groups)
    for row in group_summaries(pf):
        lead = leading_chain(pf, row["group"])
        row["leading_chain"] = (
            f"{lead['model']}/{lead['image_name']}/{lead['hyperparameter']} run {lead['run_idx']}"
        )
        print(json.dumps(row, default=str), flush=True)


# ----------------------------------------------------------------------------- page decoding


def _varint(stream: BinaryIO) -> int:
    result = shift = 0
    while True:
        byte = stream.read(1)[0]
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result
        shift += 7


def _zigzag(stream: BinaryIO) -> int:
    value = _varint(stream)
    return (value >> 1) ^ -(value & 1)


def _thrift_struct(stream: BinaryIO) -> dict[int, Any]:
    """Minimal Thrift compact-protocol struct reader: enough for a parquet PageHeader."""
    fields: dict[int, Any] = {}
    field_id = 0
    while True:
        head = stream.read(1)[0]
        if head == 0:
            return fields
        delta, kind = head >> 4, head & 0x0F
        field_id = field_id + delta if delta else _zigzag(stream)
        if kind in (1, 2):
            fields[field_id] = kind == 1
        elif kind in (4, 5, 6):
            fields[field_id] = _zigzag(stream)
        elif kind == 3:
            fields[field_id] = stream.read(1)[0]
        elif kind == 8:
            fields[field_id] = stream.read(_varint(stream))
        elif kind == 12:
            fields[field_id] = _thrift_struct(stream)
        else:
            raise ValueError(f"unsupported thrift type {kind} in page header")


def _snappy_prefix(stream: BinaryIO) -> Iterator[bytes]:
    """Decode a raw Snappy stream incrementally, yielding output as it becomes available.

    PNG bytes are incompressible, so the stream is almost all 64 kB literals and this pure-Python
    loop is fast. Copies can reference anything earlier, so the full output so far is kept.
    """
    _varint(stream)  # uncompressed length; we stop long before the end
    out = bytearray()
    emitted = 0
    while True:
        tag_byte = stream.read(1)
        if not tag_byte:
            return
        tag = tag_byte[0]
        kind = tag & 3
        if kind == 0:
            length = (tag >> 2) + 1
            if length > 60:
                extra = length - 60
                length = int.from_bytes(stream.read(extra), "little") + 1
            out += stream.read(length)
        else:
            if kind == 1:
                length = ((tag >> 2) & 7) + 4
                offset = ((tag >> 5) << 8) | stream.read(1)[0]
            elif kind == 2:
                length = (tag >> 2) + 1
                offset = int.from_bytes(stream.read(2), "little")
            else:
                length = (tag >> 2) + 1
                offset = int.from_bytes(stream.read(4), "little")
            start = len(out) - offset
            for i in range(length):  # may overlap its own output, so byte by byte
                out.append(out[start + i])
        if len(out) - emitted >= 1 << 20:
            yield bytes(out[emitted:])
            emitted = len(out)


def leading_images(group: int, count: int) -> Iterator[bytes]:
    """The first ``count`` images of a row group, read straight from its dictionary page."""
    pf = open_parquet()
    rg = pf.metadata.row_group(group)
    names = pf.schema_arrow.names
    col = rg.column(names.index("output_image"))
    if col.compression != "SNAPPY" or col.dictionary_page_offset is None:
        raise SystemExit(
            f"unexpected layout: {col.compression}, dict offset {col.dictionary_page_offset}"
        )
    stream = HfFileSystem().open(PARQUET, "rb", block_size=16 * 1024 * 1024, cache_type="readahead")
    stream.seek(col.dictionary_page_offset)
    header = _thrift_struct(stream)
    if header.get(1) != 2:  # PageType.DICTIONARY_PAGE
        raise SystemExit(f"expected a dictionary page, got page type {header.get(1)}")
    available = header[7][1]  # DictionaryPageHeader.num_values
    count = min(count, available)

    pending = bytearray()
    produced = 0
    for chunk in _snappy_prefix(stream):
        pending += chunk
        while len(pending) >= 4:  # PLAIN byte arrays: 4-byte little-endian length, then the bytes
            size = int.from_bytes(pending[:4], "little")
            if len(pending) < 4 + size:
                break
            yield bytes(pending[4 : 4 + size])
            del pending[: 4 + size]
            produced += 1
            if produced >= count:
                stream.close()
                return


def fetch(group: int, steps: int, long_side: int) -> Path:
    pf = open_parquet()
    lead = leading_chain(pf, group)
    model, image, chain, run = (lead[k] for k in META_COLUMNS[:4])
    steps = min(steps, len(lead["steps"]))
    out_dir = CACHE / f"{image}_{chain}_run{run}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"row group {group}: {model}/{image}/{chain} run {run}, steps 1..{steps}", flush=True)

    config = hf_hub_download(REPO, f"hyperparameters/{chain}.jsonl", repo_type="dataset")
    prompts = [
        json.loads(line)["prompt"] for line in Path(config).read_text().splitlines() if line.strip()
    ]
    source = Image.open(
        hf_hub_download(REPO, f"inputs/{model}/{image}.png", repo_type="dataset")
    ).convert("RGB")
    native_source = source.size
    source.thumbnail((long_side, long_side), Image.Resampling.LANCZOS)
    source.save(out_dir / "step_000.png")

    manifest = []
    for index, data in enumerate(leading_images(group, steps)):
        step = lead["steps"][index]
        img = Image.open(io.BytesIO(data)).convert("RGB")
        native = img.size
        img.thumbnail((long_side, long_side), Image.Resampling.LANCZOS)
        img.save(out_dir / f"step_{step:03d}.png")
        manifest.append({"step": step, "instruction": prompts[step - 1], "native_size": native})
        print(f"  step {step}: {native} -> {img.size} ({len(data) / 1e6:.1f} MB)", flush=True)
    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": REPO,
                "license": "CC BY-NC-SA 4.0",
                "model": model,
                "image": image,
                "chain": chain,
                "run": run,
                "source_native_size": native_source,
                "long_side": long_side,
                "steps": manifest,
            },
            indent=2,
        )
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inspect")
    f = sub.add_parser("fetch")
    f.add_argument("--group", type=int, default=10)
    f.add_argument("--steps", type=int, default=20)
    f.add_argument("--long-side", type=int, default=1280)
    args = parser.parse_args()
    if args.cmd == "inspect":
        inspect()
    else:
        print("wrote", fetch(args.group, args.steps, args.long_side))


if __name__ == "__main__":
    main()
