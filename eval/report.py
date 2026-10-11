"""Render the measured numbers into Markdown.

Writes docs/numbers.md in full and replaces the blocks between `<!-- BEGIN NUMBERS -->` and
`<!-- END NUMBERS -->` in README.md, docs/writeup.md and docs/video-script.md. Nothing is typed in
by hand: if a result file is missing, the text says so.

    python -m eval.report
"""

from __future__ import annotations

import json
import re
from typing import Any

from eval.common import RESULTS, ROOT

BEGIN, END = "<!-- BEGIN NUMBERS -->", "<!-- END NUMBERS -->"


def load(name: str) -> dict[str, Any] | None:
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def variant(data: dict[str, Any], vid: str) -> dict[str, Any]:
    return next(v for v in data["variants"] if v["id"] == vid)


def synthetic_table(data: dict[str, Any], ids: list[str]) -> str:
    info = data["metric_info"]
    head = (
        "| Method | "
        + " | ".join(f"{m['label']} {'↑' if m['better'] == 'higher' else '↓'}" for m in info)
        + " |"
    )
    rule = "| --- | " + " | ".join("---:" for _ in info) + " |"
    lines = [head, rule]
    for vid in ids:
        v = variant(data, vid)
        label = f"**{v['label']}**" if vid == "peel" else v["label"]
        cells = " | ".join(f"{v['metrics'][m['key']]:.{m['digits']}f}" for m in info)
        lines.append(f"| {label} | {cells} |")
    return "\n".join(lines)


def focus_rows(cases: list[dict[str, Any]]) -> str:
    """Each ablation on the cases where its component has something to do."""
    import numpy as np

    def subset(pred: Any) -> list[dict[str, Any]]:
        return [c for c in cases if pred(c) and c["metrics"].get("peel")]

    specs = [
        (
            "no_rereg",
            "Re-registration",
            "cases with a frame shift",
            lambda c: "shift" in c["drifts"],
        ),
        ("no_cast", "Cast separation", "cases with a colour cast", lambda c: "cast" in c["drifts"]),
        (
            "no_intent",
            "Intent grounding",
            "cases with a local drift patch",
            lambda c: any(d.startswith("patch") for d in c["drifts"]),
        ),
    ]
    lines = [
        "| Component removed | Evaluated on | n | Intended IoU (full → ablated) | Collateral reverted % (full → ablated) | Kept-edit PSNR dB (full → ablated) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for vid, label, where, pred in specs:
        rows = [c for c in subset(pred) if c["metrics"].get(vid)]
        if not rows:
            continue

        def m(v: str, key: str, rows: list[dict[str, Any]] = rows) -> float:
            return float(np.mean([c["metrics"][v][key] for c in rows]))

        lines.append(
            f"| {label} | {where} | {len(rows)} | {m('peel', 'intended_iou'):.3f} → {m(vid, 'intended_iou'):.3f} | "
            f"{m('peel', 'collateral_reverted_pct'):.1f} → {m(vid, 'collateral_reverted_pct'):.1f} | "
            f"{m('peel', 'edit_psnr'):.1f} → {m(vid, 'edit_psnr'):.1f} |"
        )
    return "\n".join(lines)


def chain_tables(data: dict[str, Any]) -> str:
    parts = []
    for group in data["groups"]:
        lines = [
            f"**{group['label']}** (n = {len(group['chains'])}). {group['note']}",
            "",
            "| Step | Raw LPIPS ↓ | Peeled LPIPS ↓ | Raw PSNR dB ↑ | Peeled PSNR dB ↑ |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
        for c in group["checkpoints"]:
            peeled = "identical" if c["peeled_psnr"] >= 99 else f"{c['peeled_psnr']:.1f}"
            lines.append(
                f"| {c['step']} | {c['raw_lpips']:.3f} | {c['peeled_lpips']:.3f} | {c['raw_psnr']:.1f} | {peeled} |"
            )
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def latency_table(data: dict[str, Any]) -> str:
    lines = ["| Stage | p50 ms | p95 ms |", "| --- | ---: | ---: |"]
    for s in data["stages"]:
        lines.append(f"| {s['stage']} | {s['p50']:.0f} | {s['p95']:.0f} |")
    lines.append(
        f"| **End to end** | **{data['total']['p50']:.0f}** | **{data['total']['p95']:.0f}** |"
    )
    return "\n".join(lines)


def summary_sentences() -> list[str]:
    """Short, source-backed statements for the README, write-up and video script."""
    out: list[str] = []
    synthetic = load("synthetic")
    if synthetic:
        raw, base, peel = (variant(synthetic, v)["metrics"] for v in ("raw", "diff_merge", "peel"))
        n = synthetic["meta"]["n_cases"]
        out.append(
            f"On {n} synthetic cases with known ground truth, Peel restores {peel['collateral_reverted_pct']:.1f}% of "
            f"unrequested pixel changes byte for byte; the diff-merge baseline restores {base['collateral_reverted_pct']:.1f}%."
        )
        out.append(
            f"Peel's kept region matches the requested edit with IoU {peel['intended_iou']:.2f} "
            f"(diff-merge: {base['intended_iou']:.2f}), and unedited-region LPIPS drops from {raw['unedited_lpips']:.3f} "
            f"in the raw edit to {peel['unedited_lpips']:.3f}."
        )
    if synthetic:
        peel, no_intent, no_rereg, no_cast = (
            variant(synthetic, v)["metrics"] for v in ("peel", "no_intent", "no_rereg", "no_cast")
        )
        out.append(
            f"Ablations: without instruction grounding the kept-region IoU falls from {peel['intended_iou']:.2f} to "
            f"{no_intent['intended_iou']:.2f}; without cast separation kept-edit fidelity falls from "
            f"{peel['edit_psnr']:.1f} to {no_cast['edit_psnr']:.1f} dB (while IoU moves from {peel['intended_iou']:.2f} to "
            f"{no_cast['intended_iou']:.2f}); removing the background-locked re-registration changes IoU from "
            f"{peel['intended_iou']:.3f} to {no_rereg['intended_iou']:.3f} on this suite, where shifts are global and the "
            f"first-pass fit already solves them."
        )
    banana = load("banana100")
    if banana:
        edit = next((g for g in banana["groups"] if g["id"] == "edit"), None)
        if edit:
            last = edit["mean_curve"][-1]
            out.append(
                f"On a real Nano Banana chain from Banana100, after {last['step']} chained edits the untouched areas of the "
                f"raw chain are at LPIPS {last['raw_lpips']:.3f} and {last['raw_psnr']:.1f} dB PSNR against step 0; the "
                f"peeled chain is at LPIPS {last['peeled_lpips']:.3f} and {last['peeled_psnr']:.1f} dB."
            )
    face = load("face_lock")
    if face:
        full, off = face["variants"]
        out.append(
            f"When a re-rendered face touches the edit ({face['meta']['n_cases']} cases), the face lock restores "
            f"{full['face_restored_pct']:.0f}% of face pixels to the original (without it: {off['face_restored_pct']:.0f}%)."
        )
    latency = load("latency") or load("latency_local")
    if latency:
        out.append(
            f"End-to-end latency is {latency['total']['p50'] / 1000:.1f} s at the median and "
            f"{latency['total']['p95'] / 1000:.1f} s at p95 on {latency['meta']['image']} ({latency['meta']['hardware']}), CPU only."
        )
    real = load("real_pairs")
    if real and real.get("summary"):
        s = real["summary"]
        out.append(
            f"On {real['meta']['n']} hand-labelled Nano Banana edits: intended recall {s['intended_recall']:.2f}, "
            f"collateral-revert precision {s['collateral_revert_precision']:.2f}."
        )
    return out


def full_report() -> str:
    parts = [
        "# Numbers",
        "",
        "Generated by `python -m eval.report` from `eval/results/*.json`. Do not edit by hand.",
        "",
    ]
    synthetic = load("synthetic")
    if synthetic:
        meta = synthetic["meta"]
        parts += [
            "## Synthetic drift suite",
            "",
            f"{meta['n_cases']} cases at {meta['long_side']} px, {len(meta['bases'])} base photos, edit kinds "
            f"{', '.join(meta['edit_kinds'])}, drift kinds {', '.join(meta['drift_kinds'])}. "
            f"Commit `{meta['git_sha']}`, {meta['generated_at']}. PSNR is capped at {meta['psnr_cap_db']:.0f} dB.",
            "",
            synthetic_table(synthetic, ["raw", "diff_merge", "peel"]),
            "",
            "## Ablations (all cases)",
            "",
            synthetic_table(
                synthetic, ["peel", "no_rereg", "no_intent", "no_cast", "no_face", "diff_merge"]
            ),
            "",
        ]
        cases_path = RESULTS / "synthetic_cases.json"
        if cases_path.exists():
            parts += [
                "## Ablations where each component has work to do",
                "",
                focus_rows(json.loads(cases_path.read_text())),
                "",
            ]
        parts += [
            "## Full Peel by edit kind",
            "",
            "| Edit kind | n | Intended IoU | Collateral reverted % |",
            "| --- | ---: | ---: | ---: |",
        ]
        parts += [
            f"| {k} | {v['n']} | {v['intended_iou']:.3f} | {v['collateral_reverted_pct']:.1f} |"
            for k, v in synthetic["peel_by_edit_kind"].items()
        ]
        parts += [
            "",
            "## Full Peel by drift kind",
            "",
            "| Drift kind | n | Intended IoU | Collateral reverted % |",
            "| --- | ---: | ---: | ---: |",
        ]
        parts += [
            f"| {k} | {v['n']} | {v['intended_iou']:.3f} | {v['collateral_reverted_pct']:.1f} |"
            for k, v in synthetic["peel_by_drift_kind"].items()
        ]
        parts.append("")
    face = load("face_lock")
    if face:
        parts += [
            "## Face lock, edit touching a re-rendered face",
            "",
            f"{face['meta']['n_cases']} cases. {face['meta']['note']} Mean SFace identity distance of the raw edit's face to the original: {face['raw_identity_distance']:.3f}.",
            "",
            "| Method | Face pixels restored % ↑ | Identity distance after repair ↓ | Faces flagged % |",
            "| --- | ---: | ---: | ---: |",
        ]
        parts += [
            f"| {v['label']} | {v['face_restored_pct']:.1f} | {v['identity_distance']:.3f} | {v['faces_flagged_pct']:.0f} |"
            for v in face["variants"]
        ]
        parts.append("")
    tuning = load("tuning")
    if tuning:
        parts += [
            "## Threshold sweep",
            "",
            f"{tuning['meta']['n_cases']} cases. The shipped default is marked.",
            "",
            "| k (MAD) | Ceiling | Intended IoU | Collateral reverted % | Kept-edit PSNR dB | Lost edits | |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
        parts += [
            f"| {r['k_mad']:.0f} | {r['threshold_ceiling']:.1f} | {r['intended_iou']:.3f} | {r['collateral_reverted_pct']:.1f} | {r['edit_psnr']:.1f} | {r['lost_edits']} | {'default' if r['is_default'] else ''} |"
            for r in tuning["grid"]
        ]
        parts.append("")
    banana = load("banana100")
    if banana and banana["groups"]:
        parts += ["## Banana100 chains", "", banana["meta"]["note"], "", chain_tables(banana), ""]
    real = load("real_pairs")
    parts += ["## Real pairs", ""]
    if real and real.get("summary"):
        s = real["summary"]
        parts += [
            f"{real['meta']['n']} pairs. Intended recall {s['intended_recall']:.3f}, collateral-revert precision {s['collateral_revert_precision']:.3f}, mean drift score {s['mean_drift_score']:.1f}.",
            "",
        ]
    else:
        parts += [f"Not measured: {(real or {}).get('skipped', 'no result file')}", ""]
    for name, title in (
        ("latency", "Latency on the deployed hardware"),
        ("latency_local", "Latency on a laptop limited to 2 threads"),
    ):
        data = load(name)
        if data:
            parts += [
                f"## {title}",
                "",
                f"{data['meta']['n']} runs, {data['meta']['image']}, {data['meta']['hardware']}.",
                "",
                latency_table(data),
                "",
            ]
    return "\n".join(parts)


def readme_block() -> str:
    lines = [f"- {s}" for s in summary_sentences()] or ["- Not measured yet. Run `make eval`."]
    synthetic = load("synthetic")
    block = ["", *lines, ""]
    if synthetic:
        block += [synthetic_table(synthetic, ["raw", "diff_merge", "peel"]), ""]
    block.append(
        "Full tables, ablations and method: [docs/numbers.md](docs/numbers.md) and [docs/eval.md](docs/eval.md)."
    )
    block.append("")
    return "\n".join(block)


def replace_block(path: str, content: str) -> None:
    file = ROOT / path
    if not file.exists():
        return
    text = file.read_text()
    if BEGIN not in text:
        return
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)
    file.write_text(pattern.sub(lambda _: f"{BEGIN}\n{content}\n{END}", text))


def main() -> None:
    (ROOT / "docs" / "numbers.md").write_text(full_report() + "\n")
    replace_block("README.md", readme_block())
    sentences = summary_sentences()
    brief = [s for s in sentences if not s.startswith("Ablations:")][:3]
    replace_block("docs/writeup.md", "\n".join(f"- {s}" for s in brief) or "- Not measured yet.")
    replace_block(
        "docs/video-script.md", "\n".join(f"- {s}" for s in sentences) or "- Not measured yet."
    )
    print("\n".join(summary_sentences()))


if __name__ == "__main__":
    main()
