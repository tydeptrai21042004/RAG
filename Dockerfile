FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    PRODUCT_CSV=/app/data/product.csv

WORKDIR /app

# System deps needed by start.sh
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy code + start script
COPY . /app
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Use start.sh as ENTRYPOINT so it runs even if render.yaml sets dockerCommand
ENTRYPOINT ["/app/start.sh"]

# Default command (Render may override via dockerCommand; ENTRYPOINT still runs)
CMD ["gunicorn", "-w", "1", "-k", "gthread", "-b", "0.0.0.0:7860", "app:app"]
