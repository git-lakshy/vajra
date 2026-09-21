# VAJRA nowcast server (FastAPI + static Leaflet dashboard)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# OpenCV headless needs libglib; keep layer small
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 curl \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY vajra/ ./vajra/
COPY dashboard/ ./dashboard/

EXPOSE 8000
CMD ["python", "-m", "vajra.api.server", "--source", "synthetic",
     "--interval", "2.0", "--port", "8000"]
