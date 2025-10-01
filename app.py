#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flask backend + Gemini 2.0 Flash + simple CSV RAG (load/preview/ask).
New endpoints:
  - POST /data/load        (upload CSV or load by path)
  - GET  /data/preview     (head of loaded CSV)
  - POST /rag/ask          (semantic search over loaded CSV, rule-based answer)
Existing:
  - GET  /health
  - POST /chat
  - POST /classify/retail
Env:
  - GOOGLE_API_KEY=<key>              (for Gemini)
  - MODEL_ID=gemini-2.0-flash-001     (optional)
  - PRODUCT_CSV=/app/products.csv     (optional auto-load on boot)
"""
import os, re, json
from datetime import datetime
from typing import List, Dict, Any

from flask import Flask, jsonify, request
from flask_cors import CORS
from google import genai
from google.genai import types as gtypes

# ---------- Gemini config ----------
MODEL_ID = os.getenv("MODEL_ID", "gemini-2.0-flash-001")
API_KEY  = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

_client = None
def get_client():
    global _client
    if _client is None:
        if not API_KEY:
            raise RuntimeError("Missing GOOGLE_API_KEY (or GEMINI_API_KEY) environment variable")
        _client = genai.Client(api_key=API_KEY)
    return _client

# ---------- Minimal CSV + RAG layer ----------
import pandas as pd
import numpy as np
from unidecode import unidecode
from sentence_transformers import SentenceTransformer
import faiss

_df: pd.DataFrame | None = None          # loaded CSV
_rag_model: SentenceTransformer | None = None
_rag_index: faiss.Index | None = None
_passages: List[str] = []
_meta: List[Dict[str, Any]] = []

def _strip(x): return "" if pd.isna(x) else str(x)
def _nd(x):    return unidecode(x or "")

def _row_to_passage(row: pd.Series) -> str:
    name = _strip(row.get("product_name"))
    desc = _strip(row.get("description"))
    status = _strip(row.get("product_status"))
    qty   = _strip(row.get("available_quantity"))
    onhand = _strip(row.get("onhand_quantity"))

    sale = _strip(row.get("saleprice"))
    start_date = _strip(row.get("start_date"))
    reg = _strip(row.get("regprice"))
    reg_s = _strip(row.get("reg_sdate")); reg_e = _strip(row.get("reg_edate"))
    prm = _strip(row.get("prmprice")); prm_wo = _strip(row.get("prmprice_wo_vat"))
    prm_s = _strip(row.get("prm_sdate")); prm_e = _strip(row.get("prm_edate"))

    lines = [
        f"Tên sản phẩm: {name}",
        f"Trạng thái: {status}",
        f"Tồn kho: {qty} (available), {onhand} (onhand)",
        f"Giá bán hiện tại (sale price): {sale}",
        f"Giá niêm yết (regular): {reg}, hiệu lực: {reg_s} → {reg_e}",
        f"Giá khuyến mãi (promo): {prm} (không VAT: {prm_wo}), hiệu lực: {prm_s} → {prm_e}",
        f"Ngày bắt đầu bán: {start_date}",
        "Mô tả: " + (desc[:800] + ("..." if len(desc) > 800 else "")),
        "UNACCENT_NAME: " + _nd(name),
        "UNACCENT_DESC: " + _nd(desc[:400]),
    ]
    return "\n".join(lines)

def _row_to_meta(row: pd.Series) -> Dict[str, Any]:
    fields = ["product_name","available_quantity","onhand_quantity","product_status",
              "saleprice","start_date","regprice","reg_sdate","reg_edate",
              "prmprice","prmprice_wo_vat","prm_sdate","prm_edate","rundate","description","document"]
    m = {k: _strip(row.get(k)) for k in fields}
    # convenience
    try:
        m["_qty"] = int(float(m.get("available_quantity") or 0))
    except Exception:
        m["_qty"] = 0
    return m

def _ensure_model():
    global _rag_model
    if _rag_model is None:
        _rag_model = SentenceTransformer("intfloat/multilingual-e5-small")

def _build_index(df: pd.DataFrame):
    global _df, _rag_index, _passages, _meta
    _df = df
    _ensure_model()
    _passages = [_row_to_passage(df.iloc[i]) for i in range(len(df))]
    _meta = [_row_to_meta(df.iloc[i]) for i in range(len(df))]
    enc = [f"passage: {p}" for p in _passages]
    X = _rag_model.encode(enc, normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    _rag_index = index

def _search(query: str, k: int = 3):
    if _rag_index is None:
        return []
    _ensure_model()
    qv = _rag_model.encode([f"query: {query}"], normalize_embeddings=True).astype("float32")
    sims, idxs = _rag_index.search(qv, k)
    out = []
    for rank, (i, s) in enumerate(zip(idxs[0], sims[0]), 1):
        if i < 0: continue
        out.append({"rank": rank, "score": float(s), "passage": _passages[i], "meta": _meta[i]})
    return out

# quick rule-based answer on top hit
PRICE_PAT = re.compile(r"(gia|giá|price|bao nhi[uê]u|cost)", re.I)
STOCK_PAT = re.compile(r"(c[oò]n h[aà]ng|t[oà]n|stock|available|quantity|s[ốo] l[uư][ơo]ng)", re.I)
PROMO_PAT = re.compile(r"(kh[uuy]ến m[aã]i|promo|sale|gi[aá] [gG]i[aả]m)", re.I)
STATUS_PAT = re.compile(r"(tr[aạ]ng th[aá]i|status|active|inactive)", re.I)
DESC_PAT = re.compile(r"(m[oô] t[aả]|description|gi[ơo]i thi[eệ]u)", re.I)

def _money(v):
    try: return f"{int(float(v)):,}".replace(",", ".")
    except: return str(v)

def _answer_direct(q: str, results: List[Dict[str,Any]]) -> str:
    if not results:
        return "Chưa có dữ liệu nào được nạp hoặc không tìm thấy sản phẩm phù hợp."
    hit = results[0]["meta"]; name = hit.get("product_name","(không tên)")
    q_nodiac = unidecode(q.lower())
    now = datetime.now()

    def _inwin(s, e):
        try:
            sd = pd.to_datetime(s) if s else None
            ed = pd.to_datetime(e) if e else None
            if sd and ed: return sd <= now <= ed
        except: pass
        return False

    if PRICE_PAT.search(q) or "gia" in q_nodiac or "price" in q_nodiac:
        if _inwin(hit.get("prm_sdate"), hit.get("prm_edate")) and hit.get("prmprice") not in ("", "0", "0.0", None):
            return (f"Giá khuyến mãi của **{name}**: **{_money(hit['prmprice'])}₫** "
                    f"({hit.get('prm_sdate','?')} → {hit.get('prm_edate','?')}).")
        if hit.get("saleprice"):
            return f"Giá bán của **{name}**: **{_money(hit['saleprice'])}₫**."
        if hit.get("regprice"):
            return f"Giá niêm yết của **{name}**: **{_money(hit['regprice'])}₫**."
        return f"Chưa có giá cho **{name}**."

    if STOCK_PAT.search(q):
        return f"**{name}** còn {hit.get('_qty',0)} (available), onhand: {hit.get('onhand_quantity','?')}."

    if PROMO_PAT.search(q):
        if hit.get("prmprice") not in ("", "0", "0.0", None):
            return (f"**{name}** có khuyến mãi: **{_money(hit['prmprice'])}₫** "
                    f"({hit.get('prm_sdate','?')} → {hit.get('prm_edate','?')}).")
        return f"**{name}** hiện chưa có giá khuyến mãi trong dữ liệu."

    if STATUS_PAT.search(q):
        return f"Trạng thái **{name}**: {hit.get('product_status','?')}."

    if DESC_PAT.search(q) or "mo ta" in q_nodiac or "mieu ta" in q_nodiac:
        d = hit.get("description","")
        return f"**{name}** – mô tả:\n{d if d else 'Chưa có mô tả chi tiết.'}"

    return (f"Tìm thấy **{name}**.\n"
            f"- Trạng thái: {hit.get('product_status','?')}\n"
            f"- Còn: {hit.get('available_quantity','?')} | Onhand: {hit.get('onhand_quantity','?')}\n"
            f"- Giá bán: {_money(hit.get('saleprice',''))}₫ | Niêm yết: {_money(hit.get('regprice',''))}₫\n"
            f"- Khuyến mãi: {_money(hit.get('prmprice',''))}₫ ({hit.get('prm_sdate','?')} → {hit.get('prm_edate','?')})")

# ---------- Health ----------
@app.get("/health")
def health():
    return jsonify({"ok": True, "model": MODEL_ID, "data_loaded": _df is not None, "rows": 0 if _df is None else len(_df)})

# ---------- Data endpoints ----------
@app.post("/data/load")
def data_load():
    """
    Load CSV into memory and build RAG index.
    Send either:
      - multipart/form-data with file field name 'file', or
      - JSON: {"path": "/path/to/file.csv"}
    """
    try:
        if "file" in request.files:
            f = request.files["file"]
            df = pd.read_csv(f)
        else:
            data = request.get_json(force=True, silent=True) or {}
            path = (data.get("path") or "").strip()
            if not path:
                return jsonify({"error": "Provide a CSV via multipart 'file' or JSON {'path': '...'}"}), 400
            df = pd.read_csv(path)

        if df.empty:
            return jsonify({"error": "CSV is empty"}), 400

        _build_index(df)
        return jsonify({"ok": True, "rows": len(df)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/data/preview")
def data_preview():
    if _df is None:
        return jsonify({"error": "No CSV loaded. POST /data/load or set env PRODUCT_CSV"}), 400
    n = int(request.args.get("n", 5))
    return jsonify({"rows": _df.head(n).to_dict(orient="records")})

# ---------- RAG ask ----------
@app.post("/rag/ask")
def rag_ask():
    """
    { "q": "giá của silkygirl xịt khóa matte 70ml?", "k": 3 }
    """
    data = request.get_json(force=True, silent=True) or {}
    q = (data.get("q") or "").strip()
    k = int(data.get("k", 3))
    if not q:
        return jsonify({"error":"Missing 'q'"}), 400
    if _rag_index is None:
        return jsonify({"error":"No CSV loaded. POST /data/load or set env PRODUCT_CSV"}), 400

    results = _search(q, k=k)
    answer = _answer_direct(q, results)
    return jsonify({
        "answer": answer,
        "matches": [
            {"rank": r["rank"], "score": r["score"], "name": r["meta"]["product_name"]}
            for r in results
        ]
    })

# ---------- Gemini (unchanged) ----------
@app.post("/chat")
def chat():
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
        gen_cfg = gtypes.GenerateContentConfig(response_mime_type="application/json")

    try:
        resp = client.models.generate_content(model=model_id, contents=contents, config=gen_cfg)
        text = getattr(resp, "text", None)
        if text is not None and text.strip():
            return jsonify({"model": model_id, "text": text})
        return jsonify({"model": model_id, "raw": resp.to_dict()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/classify/retail")
def classify_retail():
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
            contents=[gtypes.SystemInstruction.from_text(system), f"Văn bản: {text}"],
            config=cfg
        )
        parsed = getattr(resp, "parsed", None)
        if parsed:
            return jsonify(parsed)
        txt = getattr(resp, "text", "") or ""
        try:
            return jsonify(json.loads(txt))
        except Exception:
            return jsonify({"raw_text": txt})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Auto-load CSV if PRODUCT_CSV is set
    csv_path = os.getenv("PRODUCT_CSV")
    if csv_path and os.path.exists(csv_path):
        try:
            df0 = pd.read_csv(csv_path)
            if not df0.empty:
                _build_index(df0)
                app.logger.info(f"Loaded PRODUCT_CSV: {csv_path} with {len(df0)} rows")
        except Exception as e:
            app.logger.error(f"Failed to load PRODUCT_CSV: {e}")

    port = int(os.getenv("PORT", "7860"))
    app.run(host="0.0.0.0", port=port, debug=True)
