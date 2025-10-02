FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Render injects $PORT; default to 7860 for local
ENV PORT=7860

# Single worker to minimize RAM
CMD ["gunicorn", "-w", "1", "-k", "gthread", "-b", "0.0.0.0:7860", "app:app"]
