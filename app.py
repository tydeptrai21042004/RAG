#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra-light Flask backend:
- Gemini 2.0 Flash
- CSV RAG via TF-IDF (no Torch, no FAISS)
- Endpoints: /health, /data/load, /data/preview, /rag/ask, /chat, /classify/retail
"""

import os, re, json
from datetime import datetime
from typing import List, Dict, Any

from flask import Flask, jsonify, request
from flask_cors import CORS
from google import genai
from google.genai import types as gtypes

# ---------------- Gemini ----------------
MODEL_ID = os.getenv("MODEL_ID", "gemini-2.0-flash-001")
API_KEY  = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

_client = None
def get_client():
    global _client
    if _client is None:
        if not API_KEY:
            raise RuntimeError("Missing GOOGLE_API_KEY (or GEMINI_API_KEY)")
        _client = genai.Client(api_key=API_KEY)
    return _client

# ---------------- CSV + TF-IDF RAG ----------------
import pandas as pd
from unidecode import unidecode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel  # fast dot for CSR

_df: pd.DataFrame | None = None
_vec: TfidfVectorizer | None = None
_mat = None  # scipy.sparse CSR
_passages: List[str] = []
_meta: List[Dict[str, Any]] = []

def _s(x): return "" if pd.isna(x) else str(x)
def _nd(x): return unidecode(x or "")

def _row_to_passage(row: pd.Series) -> str:
    name = _s(row.get("product_name"))
    desc = _s(row.get("description"))
    status = _s(row.get("product_status"))
    qty = _s(row.get("available_quantity"))
    onhand = _s(row.get("onhand_quantity"))
    sale = _s(row.get("saleprice"))
    start_date = _s(row.get("start_date"))
    reg = _s(row.get("regprice"))
    reg_s = _s(row.get("reg_sdate")); reg_e = _s(row.get("reg_edate"))
    prm = _s(row.get("prmprice")); prm_wo = _s(row.get("prmprice_wo_vat"))
    prm_s = _s(row.get("prm_sdate")); prm_e = _s(row.get("prm_edate"))

    core = "\n".join([
        f"Tên sản phẩm: {name}",
        f"Trạng thái: {status}",
        f"Tồn kho: {qty} (available), {onhand} (onhand)",
        f"Giá bán (sale): {sale}",
        f"Niêm yết: {reg} ({reg_s}→{reg_e})",
        f"Khuyến mãi: {prm} (kh VAT: {prm_wo}) ({prm_s}→{prm_e})",
        f"Ngày bắt đầu: {start_date}",
        "Mô tả: " + (desc[:400] + ("..." if len(desc) > 400 else "")),
    ])
    # Accent-fold duplicates to help VI without diacritics
    return core + "\n" + "UNACCENT: " + _nd(" ".join([name, desc]))

def _row_to_meta(row: pd.Series) -> Dict[str, Any]:
    fields = ["product_name","available_quantity","onhand_quantity","product_status",
              "saleprice","start_date","regprice","reg_sdate","reg_edate",
              "prmprice","prmprice_wo_vat","prm_sdate","prm_edate","rundate","description","document"]
    m = {k: _s(row.get(k)) for k in fields}
    try:
        m["_qty"] = int(float(m.get("available_quantity") or 0))
    except Exception:
        m["_qty"] = 0
    return m

def _build_index(df: pd.DataFrame):
    global _df, _vec, _mat, _passages, _meta
    _df = df
    _passages = [_row_to_passage(df.iloc[i]) for i in range(len(df))]
    _meta = [_row_to_meta(df.iloc[i]) for i in range(len(df))]

    # Lightweight TF-IDF for Vietnamese:
    # - unicode accent strip by pre-adding UNACCENT text
    # - word+bigram to catch short product names
    _vec = TfidfVectorizer(
        ngram_range=(1,2),
        min_df=1,
        max_df=0.98,
        lowercase=True,
        strip_accents="unicode",  # helps VN
    )
    _mat = _vec.fit_transform(_passages)  # CSR sparse matrix

def _search(query: str, k: int = 3):
    if _mat is None or _vec is None:
        return []
    q = f"{query}\nUNACCENT: {_nd(query)}"
    qv = _vec.transform([q])  # 1 x N
    sims = linear_kernel(qv, _mat).ravel()  # dot since tf-idf is L2-normalized
    topk = sims.argsort()[::-1][:k]
    out = []
    for rank, i in enumerate(topk, 1):
        out.append({"rank": rank, "score": float(sims[i]), "passage": _passages[i], "meta": _meta[i]})
    return out

# --------- Rule-based short answers ----------
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
        return "Chưa có dữ liệu hoặc không tìm thấy sản phẩm."
    hit = results[0]["meta"]; name = hit.get("product_name","(không tên)")
    q_nodiac = unidecode(q.lower())
    now = datetime.now()

    def _inwin(s, e):
        try:
            import pandas as _pd
            sd = _pd.to_datetime(s) if s else None
            ed = _pd.to_datetime(e) if e else None
            if sd and ed: return sd <= now <= ed
        except: pass
        return False

    if PRICE_PAT.search(q) or "gia" in q_nodiac or "price" in q_nodiac:
        if _inwin(hit.get("prm_sdate"), hit.get("prm_edate")) and hit.get("prmprice") not in ("", "0", "0.0", None):
            return f"Giá khuyến mãi của **{name}**: **{_money(hit['prmprice'])}₫**."
        if hit.get("saleprice"):
            return f"Giá bán của **{name}**: **{_money(hit['saleprice'])}₫**."
        if hit.get("regprice"):
            return f"Giá niêm yết của **{name}**: **{_money(hit['regprice'])}₫**."
        return f"Chưa có giá cho **{name}**."

    if STOCK_PAT.search(q):
        return f"**{name}** còn {hit.get('_qty',0)} (available)."

    if PROMO_PAT.search(q):
        if hit.get("prmprice") not in ("", "0", "0.0", None):
            return f"**{name}** có khuyến mãi: **{_money(hit['prmprice'])}₫**."
        return f"**{name}** hiện chưa có khuyến mãi."

    if STATUS_PAT.search(q):
        return f"Trạng thái **{name}**: {hit.get('product_status','?')}."

    if DESC_PAT.search(q):
        d = hit.get("description","")
        return f"**{name}** – mô tả:\n{d if d else 'Chưa có mô tả.'}"

    return f"Tìm thấy **{name}**. Giá bán: {_money(hit.get('saleprice',''))}₫."

# ---------------- Endpoints ----------------
@app.get("/health")
def health():
    return jsonify({"ok": True, "model": MODEL_ID, "data_loaded": _df is not None, "rows": 0 if _df is None else len(_df)})

@app.post("/data/load")
def data_load():
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
        return jsonify({"error": "No CSV loaded"}), 400
    n = int(request.args.get("n", 5))
    return jsonify({"rows": _df.head(n).to_dict(orient="records")})

@app.post("/rag/ask")
def rag_ask():
    data = request.get_json(force=True, silent=True) or {}
    q = (data.get("q") or "").strip()
    k = int(data.get("k", 3))
    if not q:
        return jsonify({"error":"Missing 'q'"}), 400
    if _mat is None:
        return jsonify({"error":"No CSV loaded"}), 400
    results = _search(q, k=k)
    return jsonify({
        "answer": _answer_direct(q, results),
        "matches": [
            {"rank": r["rank"], "score": r["score"], "name": r["meta"]["product_name"]}
            for r in results
        ]
    })

@app.post("/chat")
def chat():
    data = request.get_json(force=True, silent=True) or {}
    user_msg = (data.get("q") or "").strip()
    if not user_msg:
        return jsonify({"error": "Missing 'q'"}), 400
    client = get_client()
    try:
        resp = client.models.generate_content(model=MODEL_ID, contents=[user_msg])
        return jsonify({"model": MODEL_ID, "text": resp.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/classify/retail")
def classify_retail():
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Missing 'text'"}), 400
    client = get_client()
    system = "Phân loại xem văn bản này có liên quan đến bán lẻ (giá, tồn kho, khuyến mãi, đơn hàng, sản phẩm...)."
    schema = {
        "type": "object",
        "properties": {
            "is_retail": {"type": "boolean"},
            "labels": {"type": "array", "items": {"type": "string"}},
            "confidence": {"type": "number"}
        },
        "required": ["is_retail","labels","confidence"]
    }
    cfg = gtypes.GenerateContentConfig(response_mime_type="application/json", response_schema=schema)
    try:
        resp = client.models.generate_content(model=MODEL_ID, contents=[gtypes.SystemInstruction.from_text(system), text], config=cfg)
        return jsonify(resp.parsed or {"raw": resp.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
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
