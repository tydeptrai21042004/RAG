# Dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps: minimal (add build tools only if you compile libs)
# RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt /app/requirements.txt

# Install Python deps
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app code
COPY . /app

# Default port for the app (Render/Railway will set $PORT; we keep a fallback)
ENV PORT=7860

# Start with gunicorn in container
# You can override CMD at docker run if needed.
CMD ["gunicorn", "-w", "2", "-k", "gthread", "-b", "0.0.0.0:7860", "app:app"]
