#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight Flask backend using Google Gen AI SDK (Gemini 2.0 Flash).
Endpoints:
  - GET  /health
  - POST /chat                -> free-form chat completion
  - POST /classify/retail     -> classify if a message is retail-related, JSON output
Set env:
  - GOOGLE_API_KEY=<your key>
Optional:
  - MODEL_ID (default: gemini-2.0-flash-001)
"""
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from google import genai
from google.genai import types as gtypes

MODEL_ID = os.getenv("MODEL_ID", "gemini-2.0-flash-001")
API_KEY  = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Lazy client creation (so the app can boot without key for docs/health checks)
_client = None
def get_client():
    global _client
    if _client is None:
        if not API_KEY:
            raise RuntimeError("Missing GOOGLE_API_KEY (or GEMINI_API_KEY) environment variable")
        _client = genai.Client(api_key=API_KEY)
    return _client

@app.get("/health")
def health():
    return jsonify({"ok": True, "model": MODEL_ID})

@app.post("/chat")
def chat():
    """
    Request JSON:
      {
        "q": "your user message",
        "system": "optional system prompt",
        "model": "optional model override, default MODEL_ID",
        "json": false  # if true, ask model to return JSON (response_mime_type)
      }
    """
    data = request.get_json(force=True, silent=True) or {}
    user_msg = (data.get("q") or "").strip()
    system_msg = (data.get("system") or "").strip()
    model_id = data.get("model") or MODEL_ID
    want_json = bool(data.get("json", False))

    if not user_msg:
        return jsonify({"error": "Missing 'q'"}), 400

    client = get_client()

    contents = []
    if system_msg:
        contents.append(gtypes.SystemInstruction.from_text(system_msg))
    contents.append(user_msg)

    gen_cfg = None
    if want_json:
        gen_cfg = gtypes.GenerateContentConfig(
            response_mime_type="application/json"
        )

    try:
        resp = client.models.generate_content(
            model=model_id,
            contents=contents,
            config=gen_cfg
        )
        # Prefer text if present; if JSON was requested, sdk maps to .candidates[0].content/parts[...]
        text = getattr(resp, "text", None)
        if text is not None and text.strip():
            return jsonify({"model": model_id, "text": text})
        # Fall back to raw response dict (safe subset)
        return jsonify({"model": model_id, "raw": resp.to_dict()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/classify/retail")
def classify_retail():
    """
    Simple retail-related classifier using Gemini JSON responses.
    Request JSON:
      {"text": "nội dung cần phân loại"}
    Response JSON:
      {"is_retail": true/false, "labels": [...], "confidence": 0-1, "explanation": "..."}
    """
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Missing 'text'"}), 400

    client = get_client()

    system = (
        "Bạn là bộ phân loại ngắn gọn. Hãy quyết định xem thông điệp sau có liên quan tới lĩnh vực bán lẻ "
        "(ví dụ: sản phẩm, tồn kho, giá cả, khuyến mãi, đơn hàng, cửa hàng, đổi trả, bảo hành, khách hàng) hay không. "
        "Nếu có, trả về các nhãn con phù hợp (ví dụ: 'price', 'stock', 'promotion', 'order', 'store', 'return', 'warranty', 'customer_support', 'product_info'). "
        "Trả lời theo JSON schema."
    )

    schema = {
        "type": "object",
        "properties": {
            "is_retail": {"type": "boolean"},
            "labels": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"},
            "explanation": {"type": "string"},
        },
        "required": ["is_retail", "labels", "confidence"]
    }

    cfg = gtypes.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema
    )

    try:
        resp = client.models.generate_content(
            model=MODEL_ID,
            contents=[
                gtypes.SystemInstruction.from_text(system),
                f"Văn bản: {text}"
            ],
            config=cfg
        )
        # SDK will map JSON responses into resp.parsed if schema is set
        parsed = getattr(resp, "parsed", None)
        if parsed:
            return jsonify(parsed)
        # else, try text -> json
        txt = getattr(resp, "text", "") or ""
        import json as _json
        try:
            return jsonify(_json.loads(txt))
        except Exception:
            return jsonify({"raw_text": txt})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Dev server. In production use: gunicorn -w 2 -k gthread -b 0.0.0.0:$PORT app:app
    port = int(os.getenv("PORT", "7860"))
    app.run(host="0.0.0.0", port=port, debug=True)
