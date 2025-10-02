#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultra-light Flask backend:
- Gemini 2.0 Flash
- CSV RAG via pure-Python TF-IDF (no pandas, no numpy, no sklearn)
- Endpoints: /health, /data/load, /data/preview, /rag/ask, /chat, /classify/retail
"""

import os, re, json, math, csv
from datetime import datetime
from typing import List, Dict, Any, Tuple
from collections import Counter, defaultdict

from flask import Flask, jsonify, request
from flask_cors import CORS
from google import genai
from google.genai import types as gtypes
from unidecode import unidecode

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

# ---------------- Tiny CSV + TF-IDF RAG ----------------
_rows: List[Dict[str, str]] = []     # raw CSV rows (list of dicts)
_passages: List[str] = []            # text used for retrieval
_meta: List[Dict[str, Any]] = []     # small metadata per row

# TF-IDF structures (pure Python)
_vocab_df: Dict[str, int] = {}       # document frequency per term
_doc_vecs: List[Dict[str, float]] = []  # tf-idf sparse vectors per doc
_doc_norms: List[float] = []         # L2 norm per doc
_N_docs: int = 0

# ------------ CSV helpers ------------
def _s(x): 
    if x is None: return ""
    return str(x)

def _row_to_passage(row: Dict[str, str]) -> str:
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
    return core + "\nUNACCENT: " + unidecode(" ".join([name, desc]))

def _row_to_meta(row: Dict[str, str]) -> Dict[str, Any]:
    fields = ["product_name","available_quantity","onhand_quantity","product_status",
              "saleprice","start_date","regprice","reg_sdate","reg_edate",
              "prmprice","prmprice_wo_vat","prm_sdate","prm_edate","rundate","description","document"]
    m = {k: _s(row.get(k)) for k in fields}
    try:
        m["_qty"] = int(float(m.get("available_quantity") or 0))
    except Exception:
        m["_qty"] = 0
    return m

# ------------ Tiny tokenizer ------------
TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)

def tokenize(text: str) -> List[str]:
    # lowercase + remove accents
    t = unidecode((text or "").lower())
    toks = TOKEN_RE.findall(t)
    # add bigrams for better matching of short names
    bigrams = [f"{toks[i]}_{toks[i+1]}" for i in range(len(toks)-1)]
    return toks + bigrams

# ------------ Build TF-IDF ------------
def build_tfidf(passages: List[str]) -> Tuple[List[Dict[str,float]], List[float], Dict[str,int], int]:
    N = len(passages)
    df = defaultdict(int)
    doc_tokens: List[List[str]] = []
    # collect DF
    for p in passages:
        toks = tokenize(p)
        doc_tokens.append(toks)
        for term in set(toks):
            df[term] += 1
    # compute tf-idf vectors
    vecs = []
    norms = []
    for toks in doc_tokens:
        counts = Counter(toks)
        length = sum(counts.values()) or 1
        vec = {}
        for term, cnt in counts.items():
            tf = cnt / length
            idf = math.log((N + 1) / (df[term] + 1)) + 1.0
            vec[term] = tf * idf
        # L2 norm
        norm = math.sqrt(sum(w*w for w in vec.values())) or 1.0
        vecs.append(vec)
        norms.append(norm)
    return vecs, norms, dict(df), N

def cosine_for_query(q: str, vecs: List[Dict[str,float]], norms: List[float], df: Dict[str,int], N: int) -> List[float]:
    toks = tokenize(q + "\nUNACCENT: " + unidecode(q))
    counts = Counter(toks)
    length = sum(counts.values()) or 1
    q_vec = {}
    for term, cnt in counts.items():
        tf = cnt / length
        idf = math.log((N + 1) / ((df.get(term) or 0) + 1)) + 1.0
        q_vec[term] = tf * idf
    q_norm = math.sqrt(sum(w*w for w in q_vec.values())) or 1.0

    sims: List[float] = []
    for dvec, dnorm in zip(vecs, norms):
        # sparse dot over intersection
        dot = 0.0
        # iterate over smaller dict
        (small, big) = (q_vec, dvec) if len(q_vec) < len(dvec) else (dvec, q_vec)
        for term, w in small.items():
            if term in big:
                dot += w * big[term]
        sims.append(dot / (dnorm * q_norm))
    return sims

# ------------ RAG builder ------------
def build_index(rows: List[Dict[str,str]]):
    global _rows, _passages, _meta, _doc_vecs, _doc_norms, _vocab_df, _N_docs
    _rows = rows
    _passages = [_row_to_passage(r) for r in rows]
    _meta = [_row_to_meta(r) for r in rows]
    _doc_vecs, _doc_norms, _vocab_df, _N_docs = build_tfidf(_passages)

def preview_rows(n: int = 5) -> List[Dict[str,str]]:
    return _rows[:n]

def search(query: str, k: int = 3):
    if not _rows:
        return []
    sims = cosine_for_query(query, _doc_vecs, _doc_norms, _vocab_df, _N_docs)
    topk = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]
    results = []
    for rank, i in enumerate(topk, 1):
        results.append({"rank": rank, "score": float(sims[i]), "passage": _passages[i], "meta": _meta[i]})
    return results

# ------------ Rule-based answers ------------
PRICE_PAT = re.compile(r"(gia|giá|price|bao nhi[uê]u|cost)", re.I)
STOCK_PAT = re.compile(r"(c[oò]n h[aà]ng|t[oà]n|stock|available|quantity|s[ốo] l[uư][ơo]ng)", re.I)
PROMO_PAT = re.compile(r"(kh[uuy]ến m[aã]i|promo|sale|gi[aá] [gG]i[aả]m)", re.I)
STATUS_PAT = re.compile(r"(tr[aạ]ng th[aá]i|status|active|inactive)", re.I)
DESC_PAT = re.compile(r"(m[oô] t[aả]|description|gi[ơo]i thi[eệ]u)", re.I)

def _money(v):
    try: return f"{int(float(v)):,}".replace(",", ".")
    except: return str(v)

def answer_direct(q: str, results: List[Dict[str,Any]]) -> str:
    if not results:
        return "Chưa có dữ liệu hoặc không tìm thấy sản phẩm."
    hit = results[0]["meta"]; name = hit.get("product_name","(không tên)")
    q_nodiac = unidecode(q.lower())
    now = datetime.now()

    def inwin(s, e):
        try:
            from datetime import datetime
            from dateutil.parser import isoparse
        except Exception:
            # Fallback: naive parse
            pass
        try:
            import datetime as _dt
            sd = _try_parse_date(s)
            ed = _try_parse_date(e)
            if sd and ed: return sd <= now <= ed
        except:
            pass
        return False

    if PRICE_PAT.search(q) or "gia" in q_nodiac or "price" in q_nodiac:
        if inwin(hit.get("prm_sdate"), hit.get("prm_edate")) and hit.get("prmprice") not in ("", "0", "0.0", None):
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

def _try_parse_date(s: str):
    if not s: return None
    # very loose: try fromisoformat; else return None
    try:
        from datetime import datetime
        return datetime.fromisoformat(s.replace("Z","").strip())
    except:
        return None

# ---------------- Endpoints ----------------
@app.get("/health")
def health():
    return jsonify({"ok": True, "model": MODEL_ID, "data_loaded": bool(_rows), "rows": len(_rows)})

@app.post("/data/load")
def data_load():
    """
    Load CSV to memory and build TF-IDF index.
    Accepts:
      - multipart/form-data with file field 'file'
      - JSON: {"path": "/path/to/file.csv"}
    """
    try:
        if "file" in request.files:
            f = request.files["file"]
            rows = list(csv.DictReader((line.decode("utf-8") for line in f.stream)))
        else:
            data = request.get_json(force=True, silent=True) or {}
            path = (data.get("path") or "").strip()
            if not path:
                return jsonify({"error": "Provide a CSV via multipart 'file' or JSON {'path': '...'}"}), 400
            with open(path, "r", encoding="utf-8") as fp:
                rows = list(csv.DictReader(fp))
        if not rows:
            return jsonify({"error": "CSV is empty"}), 400
        build_index(rows)
        return jsonify({"ok": True, "rows": len(rows)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/data/preview")
def data_preview():
    if not _rows:
        return jsonify({"error": "No CSV loaded"}), 400
    n = int(request.args.get("n", 5))
    return jsonify({"rows": preview_rows(n)})

@app.post("/rag/ask")
def rag_ask():
    data = request.get_json(force=True, silent=True) or {}
    q = (data.get("q") or "").strip()
    k = int(data.get("k", 3))
    if not q:
        return jsonify({"error":"Missing 'q'"}), 400
    if not _rows:
        return jsonify({"error":"No CSV loaded"}), 400
    results = search(q, k=k)
    return jsonify({
        "answer": answer_direct(q, results),
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
    # Auto-load via env PRODUCT_CSV if provided
    csv_path = os.getenv("PRODUCT_CSV")
    if csv_path and os.path.exists(csv_path):
        try:
            with open(csv_path, "r", encoding="utf-8") as fp:
                rows = list(csv.DictReader(fp))
            if rows:
                build_index(rows)
                app.logger.info(f"Loaded PRODUCT_CSV: {csv_path} with {len(rows)} rows")
        except Exception as e:
            app.logger.error(f"Failed to load PRODUCT_CSV: {e}")
    port = int(os.getenv("PORT", "7860"))
    app.run(host="0.0.0.0", port=port, debug=True)
