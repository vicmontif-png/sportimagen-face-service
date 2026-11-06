# -*- coding: utf-8 -*-
import os
import sys
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify

app = Flask(__name__)

# Cargar InsightFace una sola vez en memoria
print("Cargando modelo InsightFace (buffalo_l)...")
face_app = None
try:
    from insightface.app import FaceAnalysis
    # Solo cargamos detection y recognition para consumir menos de 300 MB de RAM y caber en planes gratuitos
    face_app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    print(">>> [OK] InsightFace ArcFace 512-D listo y escuchando en la nube!")
except Exception as e:
    print(f"Error cargando InsightFace: {e}")

@app.route('/', methods=['GET'])
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "online",
        "service": "SportImagen Biometric AI Cloud",
        "model": "buffalo_l",
        "dim": 512,
        "engine": "ArcFace"
    })

@app.route('/get_embedding', methods=['POST'])
def get_embedding():
    if face_app is None:
        return jsonify({"error": "Motor InsightFace no inicializado"}), 500

    img = None

    # 1. Soporte para archivo subido (multipart/form-data)
    if request.files:
        file = next(iter(request.files.values()))
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # 2. Soporte para Base64 (JSON o form)
    elif request.is_json or 'image' in request.form or 'image_base64' in request.form:
        raw_b64 = ''
        if request.is_json:
            data = request.get_json(silent=True) or {}
            raw_b64 = data.get('image', '') or data.get('image_base64', '')
        else:
            raw_b64 = request.form.get('image', '') or request.form.get('image_base64', '')

        if raw_b64:
            if ',' in raw_b64:
                raw_b64 = raw_b64.split(',', 1)[1]
            try:
                decoded = base64.b64decode(raw_b64)
                file_bytes = np.frombuffer(decoded, np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            except Exception as e:
                return jsonify({"error": f"Base64 inválido: {e}"}), 400

    # 3. Soporte para bytes directos (raw body)
    elif request.data:
        try:
            file_bytes = np.frombuffer(request.data, np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        except Exception:
            pass

    if img is None:
        return jsonify({"error": "No se recibió ninguna imagen válida"}), 400

    try:
        faces = face_app.get(img)
        if not faces:
            return jsonify({"error": "No se detectó ningún rostro en la imagen"}), 404

        # Seleccionar la cara principal en primer plano
        faces_sorted = sorted(faces, key=lambda f: ((f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]) * f.det_score), reverse=True)
        best_face = faces_sorted[0]

        embedding = best_face.normed_embedding.tolist()
        return jsonify({
            "ok": True,
            "vector": embedding,
            "dim": len(embedding),
            "det_score": float(best_face.det_score),
            "total_caras": len(faces)
        })

    except Exception as e:
        return jsonify({"error": f"Error procesando rostro: {e}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

