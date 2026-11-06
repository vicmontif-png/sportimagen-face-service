FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Descarga anticipada del modelo InsightFace buffalo_s (15 MB, solo 150 MB RAM)
RUN python -c "from insightface.app import FaceAnalysis; app = FaceAnalysis(name='buffalo_s', allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider']); app.prepare(ctx_id=0, det_size=(640, 640))" || true

EXPOSE 5000

ENV PORT=5000
ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
