FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    PRODUCT_CSV=/app/data/products.csv

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy code + the data folder into the image
COPY . /app

# 1 worker to keep RAM low
CMD ["gunicorn", "-w", "1", "-k", "gthread", "-b", "0.0.0.0:7860", "app:app"]
