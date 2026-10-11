from __future__ import annotations

import io
import json

import numpy as np
import pytest
from app import limits
from app.limits import RateLimiter
from fastapi.testclient import TestClient
from peel.synth import COLORS, SynthCase, drift_blur, drift_cast, paste_sprite, render_sprite
from PIL import Image
from skimage import data

from .conftest import decode, png

Files = dict[str, tuple[str, bytes, str]]


def error_code(response) -> str:  # type: ignore[no-untyped-def]
    body = response.json()
    assert set(body) == {"error"} and body["error"]["message"]
    return body["error"]["code"]


def test_healthz(client: TestClient) -> None:
    body = client.get("/healthz").json()
    assert body["status"] in ("ok", "warming")
    assert set(body["models"]) == {"clipseg", "lpips", "faces"}
    assert body["edit"] == {"available": False, "model": None}
    assert body["version"] and body["git_sha"]


def test_openapi_docs_are_served(client: TestClient) -> None:
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    for path in (
        "/v1/peel",
        "/v1/peel/jobs",
        "/v1/jobs/{job_id}/events",
        "/v1/recomposite",
        "/v1/chain",
        "/v1/edit",
        "/v1/examples",
        "/healthz",
    ):
        assert path in paths


def test_exported_contract_is_current(client: TestClient) -> None:
    from pathlib import Path

    exported = json.loads((Path(__file__).resolve().parents[1] / "openapi.json").read_text())
    live = client.get("/openapi.json").json()
    assert exported["paths"].keys() == live["paths"].keys()
    assert exported["components"]["schemas"] == json.loads(
        json.dumps(live["components"]["schemas"])
    )


@pytest.fixture(scope="module")
def result(client: TestClient, files: Files, case: SynthCase) -> dict:
    response = client.post("/v1/peel", files=files, data={"instruction": case.instruction})
    assert response.status_code == 200, response.text
    assert response.headers["x-request-id"]
    return response.json()


def test_peel_returns_ledger_timings_and_images(
    client: TestClient, result: dict, case: SynthCase
) -> None:
    ledger = result["ledger"]
    assert ledger["summary"]["n_intended"] == 1 and ledger["summary"]["n_collateral"] >= 1
    assert 0 < ledger["drift_score"] <= 100 and ledger["headline"]
    assert result["timings"][0]["stage"] == "decode"
    assert result["total_ms"] > 0 and result["cached"] is False
    h, w = case.original.shape[:2]
    assert (result["width"], result["height"]) == (w, h)
    for name, url in result["images"].items():
        response = client.get(url)
        assert response.status_code == 200 and response.headers["content-type"] == "image/png", name
        image = decode(response.content)
        assert image.ndim == 3
        if name in ("original", "edited", "peeled"):
            assert image.shape[:2] == (h, w)
    assert np.array_equal(decode(client.get(result["images"]["original"]).content), case.original)


def test_identical_request_is_served_from_cache(
    client: TestClient, files: Files, case: SynthCase, result: dict
) -> None:
    again = client.post("/v1/peel", files=files, data={"instruction": case.instruction}).json()
    assert again["cached"] is True and again["id"] != result["id"]
    assert again["ledger"] == result["ledger"]
    assert (
        client.get(again["images"]["peeled"]).content
        == client.get(result["images"]["peeled"]).content
    )


def test_image_can_be_downscaled_and_downloaded(client: TestClient, result: dict) -> None:
    response = client.get(result["images"]["peeled"], params={"max": 128, "download": True})
    assert max(decode(response.content).shape[:2]) == 128
    assert "attachment" in response.headers["content-disposition"]
    assert client.get(f"/v1/jobs/{result['id']}/images/nope.png").status_code == 404


def test_recomposite_flips_a_region(client: TestClient, files: Files, case: SynthCase) -> None:
    result = client.post(
        "/v1/peel",
        files=files,
        data={"instruction": case.instruction, "options": json.dumps({"k_mad": 6.5})},
    ).json()
    intended = next(r for r in result["ledger"]["regions"] if r["classification"] == "intended")
    flipped = client.post(
        "/v1/recomposite", json={"job_id": result["id"], "overrides": {intended["id"]: "revert"}}
    )
    assert flipped.status_code == 200, flipped.text
    body = flipped.json()
    region = next(r for r in body["ledger"]["regions"] if r["id"] == intended["id"])
    assert region["decision"] == "revert" and region["overridden"] is True
    assert body["ledger"]["summary"]["bit_identical_pct"] == 100.0
    assert "?v=1" in body["images"]["peeled"]
    assert np.array_equal(decode(client.get(body["images"]["peeled"]).content), case.original)
    assert (
        error_code(
            client.post(
                "/v1/recomposite", json={"job_id": result["id"], "overrides": {"r99": "keep"}}
            )
        )
        == "INVALID_OPTIONS"
    )
    missing = client.post("/v1/recomposite", json={"job_id": "nope", "overrides": {}})
    assert missing.status_code == 404 and error_code(missing) == "JOB_NOT_FOUND"


def test_job_with_server_sent_events(client: TestClient, files: Files, case: SynthCase) -> None:
    created = client.post(
        "/v1/peel/jobs",
        files=files,
        data={"instruction": case.instruction, "options": json.dumps({"k_mad": 7.0})},
    )
    assert created.status_code == 202, created.text
    job = created.json()
    events: list[tuple[str, dict]] = []
    with client.stream("GET", job["events_url"]) as stream:
        assert stream.headers["content-type"].startswith("text/event-stream")
        name = ""
        for line in stream.iter_lines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                events.append((name, json.loads(line[6:])))
    kinds = [k for k, _ in events]
    assert kinds[-1] == "done" and "progress" in kinds
    stages = [e["stage"] for k, e in events if k == "progress"]
    for stage in ("register", "change", "intent", "classify", "composite"):
        assert stage in stages
    pcts = [e["pct"] for k, e in events if k == "progress"]
    assert pcts == sorted(pcts)
    done = events[-1][1]
    assert done["state"] == "done" and done["result"]["ledger"]["summary"]["n_intended"] == 1
    status = client.get(job["status_url"]).json()
    assert status["state"] == "done" and status["result"]["id"] == job["id"]


def test_job_error_arrives_as_an_event(client: TestClient, case: SynthCase) -> None:
    other = np.ascontiguousarray(data.coffee())
    files = {
        "original": ("a.png", png(case.original), "image/png"),
        "edited": ("b.png", png(other), "image/png"),
    }
    job = client.post("/v1/peel/jobs", files=files, data={"instruction": "add a hat"}).json()
    with client.stream("GET", job["events_url"]) as stream:
        text = "".join(stream.iter_text())
    assert "event: error" in text and "NOT_SAME_SCENE" in text
    assert client.get(job["status_url"]).json()["error"]["code"] == "NOT_SAME_SCENE"


def test_not_same_scene(client: TestClient, case: SynthCase) -> None:
    other = np.ascontiguousarray(data.coffee())
    files = {
        "original": ("a.png", png(case.original), "image/png"),
        "edited": ("b.png", png(other), "image/png"),
    }
    response = client.post("/v1/peel", files=files, data={"instruction": "add a hat"})
    assert response.status_code == 422 and error_code(response) == "NOT_SAME_SCENE"
    assert "different photos" in response.json()["error"]["message"]
    assert response.json()["error"]["request_id"] == response.headers["x-request-id"]


def test_input_errors_are_typed(
    client: TestClient, files: Files, monkeypatch: pytest.MonkeyPatch
) -> None:
    bmp = io.BytesIO()
    Image.new("RGB", (64, 64)).save(bmp, format="BMP")
    bad = {**files, "edited": ("e.bmp", bmp.getvalue(), "image/bmp")}
    response = client.post("/v1/peel", files=bad, data={"instruction": "x"})
    assert response.status_code == 415 and error_code(response) == "UNSUPPORTED_FORMAT"

    garbage = {**files, "edited": ("e.png", b"not an image at all", "image/png")}
    assert (
        error_code(client.post("/v1/peel", files=garbage, data={"instruction": "x"}))
        == "INVALID_IMAGE"
    )

    assert (
        error_code(client.post("/v1/peel", files=files, data={"instruction": "   "}))
        == "EMPTY_INSTRUCTION"
    )
    assert (
        error_code(client.post("/v1/peel/jobs", files=files, data={"instruction": " "}))
        == "EMPTY_INSTRUCTION"
    )

    response = client.post(
        "/v1/peel", files=files, data={"instruction": "x", "options": '{"k_mad": 999}'}
    )
    assert response.status_code == 422 and error_code(response) == "INVALID_OPTIONS"
    assert (
        error_code(
            client.post("/v1/peel", files=files, data={"instruction": "x", "options": "{nope"})
        )
        == "INVALID_OPTIONS"
    )

    response = client.post(
        "/v1/peel", files={"original": files["original"]}, data={"instruction": "x"}
    )
    assert response.status_code == 422 and error_code(response) == "INVALID_OPTIONS"
    assert "edited" in response.json()["error"]["message"]

    monkeypatch.setattr(limits, "MAX_BYTES", 1000)
    response = client.post("/v1/peel", files=files, data={"instruction": "x"})
    assert response.status_code == 413 and error_code(response) == "IMAGE_TOO_LARGE"


def test_chain(client: TestClient) -> None:
    original = np.ascontiguousarray(data.astronaut())
    raw = original
    uploads = [("original", ("o.png", png(original), "image/png"))]
    for name, color, x, y in (("ball", "red", 380, 40), ("star", "yellow", 30, 380)):
        sprite, mask = render_sprite(name, COLORS[color], 90)  # type: ignore[arg-type]
        raw, _ = paste_sprite(raw, sprite, mask, x, y)
        raw = drift_cast(drift_blur(raw, 0.7), db=2.5)
        uploads.append(("edited", (f"{name}.png", png(raw), "image/png")))
    response = client.post(
        "/v1/chain",
        files=uploads,
        data={"instructions": json.dumps(["Add a red ball", "Add a yellow star"])},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert [s["step"] for s in body["steps"]] == [1, 2] and len(body["metrics"]) == 2
    for metric in body["metrics"]:
        assert (
            metric["peeled_psnr"] > metric["raw_psnr"]
            and metric["peeled_lpips"] < metric["raw_lpips"]
        )
    for step in body["steps"]:
        assert client.get(step["raw"]).status_code == 200
        assert client.get(step["peeled"]).status_code == 200
        assert client.get(step["result"]["images"]["change_map"]).status_code == 200
    assert client.get(body["original"]).status_code == 200

    wrong = client.post(
        "/v1/chain", files=uploads, data={"instructions": json.dumps(["a", "b", "c"])}
    )
    assert error_code(wrong) == "INVALID_OPTIONS"
    too_many = uploads[:1] + [uploads[1]] * 9
    assert (
        error_code(
            client.post("/v1/chain", files=too_many, data={"instructions": "add a red ball"})
        )
        == "CHAIN_TOO_LONG"
    )


def test_edit_is_unavailable_without_a_key(client: TestClient, files: Files) -> None:
    response = client.post(
        "/v1/edit", files={"original": files["original"]}, data={"instruction": "add a hat"}
    )
    assert response.status_code == 503 and error_code(response) == "EDIT_UNAVAILABLE"


def test_examples_endpoint(client: TestClient) -> None:
    response = client.get("/v1/examples")
    assert response.status_code == 200
    for example in response.json():
        assert example["result"]["id"] == f"ex-{example['id']}"
        assert client.get(example["result"]["images"]["peeled"]).status_code == 200


def test_cors_allows_the_web_app_and_n8n(client: TestClient) -> None:
    for origin in (
        "http://localhost:5173",
        "https://peel-demo.vercel.app",
        "https://acme.app.n8n.cloud",
        "https://peel.xyz",
    ):
        response = client.options(
            "/v1/peel", headers={"Origin": origin, "Access-Control-Request-Method": "POST"}
        )
        assert response.headers.get("access-control-allow-origin") == origin, origin
    response = client.options(
        "/v1/peel",
        headers={"Origin": "https://evil.example.com", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in response.headers


def test_token_bucket() -> None:
    bucket = RateLimiter(capacity=2, refill_seconds=60)
    assert bucket.take("a") == 0 and bucket.take("a") == 0
    assert bucket.take("a") > 0, "third request in a burst is refused"
    assert bucket.take("b") == 0, "other clients are unaffected"


def test_rate_limit_response(
    client: TestClient, files: Files, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(limits, "limiter", RateLimiter(capacity=1, refill_seconds=60))
    headers = {"x-forwarded-for": "203.0.113.9"}
    heavy = client.post(
        "/v1/edit",
        files={"original": files["original"]},
        data={"instruction": "x"},
        headers=headers,
    )
    assert heavy.status_code == 429, "a route that costs more than the bucket holds is refused"
    first = client.post(
        "/v1/peel", files=files, data={"instruction": "add a red ball"}, headers=headers
    )
    assert first.status_code == 200
    second = client.post(
        "/v1/peel", files=files, data={"instruction": "add a red ball"}, headers=headers
    )
    assert second.status_code == 429 and error_code(second) == "RATE_LIMITED"
    assert "seconds" in second.json()["error"]["message"]
    other = client.post(
        "/v1/peel",
        files=files,
        data={"instruction": "add a red ball"},
        headers={"x-forwarded-for": "203.0.113.10"},
    )
    assert other.status_code == 200, "another address has its own bucket"


def test_health_reports_the_host_limits(client: TestClient) -> None:
    limits = client.get("/healthz").json()["limits"]
    assert limits["max_upload_bytes"] == 15 * 1024 * 1024
    assert limits["max_long_side"] == 4096 and limits["max_request_bytes"] == 0
    assert limits["max_chain_steps"] == 8
