# Gemini 2.0 Flash – Flask Backend (Lightweight)

A tiny Flask server that calls **Gemini 2.0 Flash** via the **Google Gen AI SDK**.

## 1) Run locally

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Set your key
export GOOGLE_API_KEY="YOUR_KEY"

# Dev server
python app.py
# or production server
gunicorn -w 2 -k gthread -b 0.0.0.0:$PORT app:app
```

### Test
```bash
curl -s http://localhost:7860/health

curl -s -X POST http://localhost:7860/chat \
  -H "Content-Type: application/json" \
  -d '{"q":"Viết một câu chào ngắn gọn!", "model":"gemini-2.0-flash-001"}'

curl -s -X POST http://localhost:7860/classify/retail \
  -H "Content-Type: application/json" \
  -d '{"text":"Sản phẩm này giá bao nhiêu và còn hàng không?"}'
```

## 2) Deploy to Render (Docker)

- Connect your repo (or upload), include `Dockerfile` and `render.yaml`.
- In Render dashboard, set env var `GOOGLE_API_KEY`.
- Deploy. Render will build the Docker image and start Gunicorn.

## 3) Deploy to Railway (Procfile)

- Create a new Railway project, connect repo.
- Add environment variable `GOOGLE_API_KEY`.
- Railway auto-detects `Procfile` and boots Gunicorn (`web` process).

## 4) Notes

- Default model: `gemini-2.0-flash-001` (override with `MODEL_ID` env).
- Uses JSON response mode in `/classify/retail` for structured outputs.
- For streaming or file uploads, extend with the Gen AI SDK docs.
- If you're on Google Cloud (Vertex AI), you can switch to Vertex by:
  - Set env `GOOGLE_GENAI_USE_VERTEXAI=true`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`
  - Update client creation as needed (see Google Gen AI SDK docs).

## 5) Security

- Add an auth token or API key gate in production.
- Restrict CORS to your frontend origin(s).
```python
CORS(app, resources={r"/*": {"origins": ["https://your-frontend.app"]}})
```
