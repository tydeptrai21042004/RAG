# Python slim image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (none required; keep lightweight)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Expose port (Render/Railway set $PORT)
ENV PORT=7860
CMD ["gunicorn", "-w", "2", "-k", "gthread", "-b", "0.0.0.0:7860", "app:app"]
