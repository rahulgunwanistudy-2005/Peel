"""Generate peel-gate.workflow.json (and the copy the web app offers for download).

    python integrations/n8n/build_workflow.py

Kept as a script so the workflow can be reviewed as code and regenerated when the API changes.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
WEB_COPY = HERE.parents[1] / "web" / "public" / "peel-gate.workflow.json"
PEEL_API = "https://peel-api-rahulgunwanistudy-5466s-projects.vercel.app"
NAMESPACE = uuid.UUID("5b0f2f0e-7c1d-4a8e-9a52-0d2c0e6b7a11")

nodes: list[dict[str, Any]] = []
connections: dict[str, dict[str, list[list[dict[str, Any]]]]] = {}


def nid(name: str) -> str:
    """Stable ids, so regenerating the file does not churn the diff."""
    return str(uuid.uuid5(NAMESPACE, name))


def node(name: str, kind: str, version: float, params: dict[str, Any], pos: list[int], **extra: Any) -> None:
    nodes.append(
        {"parameters": params, "id": nid(name), "name": name, "type": kind, "typeVersion": version, "position": pos, **extra}
    )


def link(a: str, b: str, out: int = 0, inp: int = 0) -> None:
    main = connections.setdefault(a, {"main": []})["main"]
    while len(main) <= out:
        main.append([])
    main[out].append({"node": b, "type": "main", "index": inp})


def condition(left: str, right: str, kind: str, operation: str, single: bool = False) -> dict[str, Any]:
    operator: dict[str, Any] = {"type": kind, "operation": operation}
    if single:
        operator["singleValue"] = True
    return {
        "conditions": {
            "options": {"caseSensitive": True, "leftValue": "", "typeValidation": "loose", "version": 2},
            "conditions": [{"id": nid(left + operation), "leftValue": left, "rightValue": right, "operator": operator}],
            "combinator": "and",
        },
        "options": {},
    }


node(
    "Read me",
    "n8n-nodes-base.stickyNote",
    1,
    {
        "width": 520,
        "height": 350,
        "content": (
            "## Peel gate\n"
            "An AI edit only goes through if it left the rest of the photo alone.\n\n"
            "1. Pick your **Google Gemini(PaLM) Api** credential on *Edit with Nano Banana*.\n"
            "2. Open **Config**: `maxDriftScore` is the gate (5 is strict). `peelApi` points at the public "
            "demo server; change it if you run your own. `geminiModel` must be an image-editing model your key can use.\n"
            "3. Optional: put a Slack or Discord incoming-webhook URL in `chatWebhook`.\n"
            "4. Activate, open the form URL, upload a product photo and an instruction.\n\n"
            "Pass: the AI edit is returned as is. Fail: Peel's repaired image is returned instead, and the "
            "ledger summary is posted to chat.\n\n"
            "Docs: https://github.com/rahulgunwanistudy-2005/Peel/tree/main/integrations/n8n"
        ),
    },
    [-760, -440],
)
node(
    "On form submission",
    "n8n-nodes-base.formTrigger",
    2.2,
    {
        "formTitle": "Edit a product photo (with a drift gate)",
        "formDescription": "Upload a product photo and say what to change. The edit is checked by Peel before it is accepted.",
        "formFields": {
            "values": [
                {
                    "fieldLabel": "Product photo",
                    "fieldType": "file",
                    "multipleFiles": False,
                    "acceptFileTypes": ".png, .jpg, .jpeg, .webp",
                    "requiredField": True,
                },
                {"fieldLabel": "Instruction", "placeholder": "Put the bottle on a marble counter", "requiredField": True},
            ]
        },
        "options": {},
    },
    [-760, 0],
    webhookId=nid("form-webhook"),
)
node(
    "Config",
    "n8n-nodes-base.set",
    3.4,
    {
        "assignments": {
            "assignments": [
                {"id": nid("peelApi"), "name": "peelApi", "value": PEEL_API, "type": "string"},
                {"id": nid("geminiModel"), "name": "geminiModel", "value": "gemini-2.5-flash-image", "type": "string"},
                {"id": nid("maxDriftScore"), "name": "maxDriftScore", "value": 5, "type": "number"},
                {"id": nid("chatWebhook"), "name": "chatWebhook", "value": "", "type": "string"},
            ]
        },
        "includeOtherFields": True,
        # Keep the uploaded file on the item: the option is named differently across Set versions.
        "options": {"stripBinary": False, "includeBinary": True},
    },
    [-540, 0],
)
node(
    "Photo to base64",
    "n8n-nodes-base.extractFromFile",
    1,
    {"operation": "binaryToPropery", "binaryPropertyName": "Product_photo", "destinationKey": "photoBase64", "options": {}},
    [-320, 120],
)
node(
    "Edit with Nano Banana",
    "n8n-nodes-base.httpRequest",
    4.2,
    {
        "method": "POST",
        "url": "=https://generativelanguage.googleapis.com/v1beta/models/{{ $('Config').first().json.geminiModel }}:generateContent",
        "authentication": "predefinedCredentialType",
        "nodeCredentialType": "googlePalmApi",
        "sendBody": True,
        "specifyBody": "json",
        "jsonBody": (
            "={{ JSON.stringify({\n"
            "  contents: [{\n"
            "    parts: [\n"
            "      { inline_data: { mime_type: $('On form submission').first().binary.Product_photo.mimeType, data: $json.photoBase64 } },\n"
            "      { text: $('On form submission').first().json.Instruction }\n"
            "    ]\n"
            "  }],\n"
            "  generationConfig: { responseModalities: ['TEXT', 'IMAGE'] }\n"
            "}) }}"
        ),
        "options": {"timeout": 120000},
    },
    [-100, 120],
)
node(
    "Pick the edited image",
    "n8n-nodes-base.code",
    2,
    {
        "jsCode": (
            "// Gemini answers with text and image parts; keep the first image.\n"
            "const parts = $input.first().json.candidates?.[0]?.content?.parts ?? [];\n"
            "const image = parts.map((p) => p.inlineData ?? p.inline_data).find((d) => d && d.data);\n"
            "if (!image) {\n"
            "  throw new Error('The image model answered without an image. Try another instruction.');\n"
            "}\n"
            "return [{ json: { editedBase64: image.data, editedMime: image.mimeType ?? image.mime_type ?? 'image/png' } }];"
        )
    },
    [120, 120],
)
node(
    "Edited image to file",
    "n8n-nodes-base.convertToFile",
    1.1,
    {
        "operation": "toBinary",
        "sourceProperty": "editedBase64",
        "binaryPropertyName": "edited",
        "options": {"fileName": "edited.png", "mimeType": "={{ $json.editedMime }}"},
    },
    [340, 120],
)
node("Original + edit", "n8n-nodes-base.merge", 3, {"mode": "combine", "combineBy": "combineByPosition", "options": {}}, [560, 0])
node(
    "Peel: measure the drift",
    "n8n-nodes-base.httpRequest",
    4.2,
    {
        "method": "POST",
        # inline=base64: the images come back inside this response, so the flow does not depend on
        # the server still holding the result when the next node runs.
        "url": "={{ $('Config').first().json.peelApi }}/v1/peel?inline=base64",
        "sendBody": True,
        "contentType": "multipart-form-data",
        "bodyParameters": {
            "parameters": [
                {"parameterType": "formBinaryData", "name": "original", "inputDataFieldName": "Product_photo"},
                {"parameterType": "formBinaryData", "name": "edited", "inputDataFieldName": "edited"},
                {"name": "instruction", "value": "={{ $('On form submission').first().json.Instruction }}"},
            ]
        },
        "options": {"timeout": 180000},
    },
    [780, 0],
)
node(
    "Drift within limit?",
    "n8n-nodes-base.if",
    2.2,
    condition("={{ $json.ledger.drift_score }}", "={{ $('Config').first().json.maxDriftScore }}", "number", "lte"),
    [1000, 0],
)
node(
    "Pass: keep the AI edit",
    "n8n-nodes-base.convertToFile",
    1.1,
    {
        "operation": "toBinary",
        "sourceProperty": "images.edited",
        "binaryPropertyName": "image",
        "options": {"fileName": "edit.webp", "mimeType": "image/webp"},
    },
    [1240, -120],
)
node(
    "Fail: use Peel's repair",
    "n8n-nodes-base.convertToFile",
    1.1,
    {
        "operation": "toBinary",
        "sourceProperty": "images.peeled",
        "binaryPropertyName": "image",
        "options": {"fileName": "peeled.webp", "mimeType": "image/webp"},
    },
    [1240, 120],
)
node(
    "Chat webhook set?",
    "n8n-nodes-base.if",
    2.2,
    condition("={{ $('Config').first().json.chatWebhook }}", "", "string", "notEmpty", single=True),
    [1460, 120],
)
node(
    "Post the ledger to chat",
    "n8n-nodes-base.httpRequest",
    4.2,
    {
        "method": "POST",
        "url": "={{ $('Config').first().json.chatWebhook }}",
        "sendBody": True,
        "specifyBody": "json",
        "jsonBody": (
            "={{ JSON.stringify({\n"
            "  text: 'Peel gate blocked an AI edit (drift score ' + $('Peel: measure the drift').first().json.ledger.drift_score"
            " + ', limit ' + $('Config').first().json.maxDriftScore + ').\\n'\n"
            "    + 'Instruction: ' + $('Peel: measure the drift').first().json.ledger.instruction + '\\n'\n"
            "    + $('Peel: measure the drift').first().json.ledger.headline + '\\n'\n"
            "    + $('Peel: measure the drift').first().json.ledger.regions.filter((r) => r.decision === 'revert')"
            ".slice(0, 5).map((r) => '- ' + r.sentence).join('\\n') + '\\n'\n"
            "    + 'The repaired image was used instead; it is attached to this workflow run.'\n"
            "}) }}"
        ),
        "options": {},
    },
    [1700, 40],
    onError="continueRegularOutput",
)

link("On form submission", "Config")
link("On form submission", "Original + edit", 0, 0)
link("Config", "Photo to base64")
link("Photo to base64", "Edit with Nano Banana")
link("Edit with Nano Banana", "Pick the edited image")
link("Pick the edited image", "Edited image to file")
link("Edited image to file", "Original + edit", 0, 1)
link("Original + edit", "Peel: measure the drift")
link("Peel: measure the drift", "Drift within limit?")
link("Drift within limit?", "Pass: keep the AI edit", 0)
link("Drift within limit?", "Fail: use Peel's repair", 1)
link("Fail: use Peel's repair", "Chat webhook set?")
link("Chat webhook set?", "Post the ledger to chat", 0)

workflow = {
    "name": "Peel gate: check AI product-photo edits for drift",
    "nodes": nodes,
    "connections": connections,
    "pinData": {},
    "settings": {"executionOrder": "v1"},
    "meta": {"templateCredsSetupCompleted": False},
    "tags": [],
}

if __name__ == "__main__":
    text = json.dumps(workflow, indent=2) + "\n"
    (HERE / "peel-gate.workflow.json").write_text(text)
    WEB_COPY.write_text(text)
    print(f"wrote {len(nodes)} nodes")
