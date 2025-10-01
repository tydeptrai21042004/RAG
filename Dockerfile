FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements first (layer caching)
COPY requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

# Copy your Flask app code
COPY . /app

# Render injects $PORT automatically
ENV PORT=7860

# Start app with gunicorn
CMD ["gunicorn", "-w", "2", "-k", "gthread", "-b", "0.0.0.0:7860", "app:app"]
