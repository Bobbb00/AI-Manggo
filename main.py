from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import io
import os
import subprocess
from PIL import Image

app = Flask(__name__, template_folder='templates')



# --- Konfigurasi Model ---
TFLITE_MODEL_PATH = 'model/model_mangga2_quantized_float16.tflite'
GOOGLE_DRIVE_ID = '1lVpRMHLQv95FJd82UL6jXnYngP8fk-nA'  # ganti dengan ID file kamu

# Download model dari Google Drive jika belum ada
if not os.path.exists(TFLITE_MODEL_PATH):
    print("Model belum ada, download dari Google Drive...")
    os.makedirs('model', exist_ok=True)
    subprocess.run([
        'gdown',
        f'https://drive.google.com/uc?id={GOOGLE_DRIVE_ID}',
        '-O', TFLITE_MODEL_PATH
    ], check=True)
    
IMAGE_SIZE = (224, 224)

CLASS_NAMES = ['Antraknosa', 'Alternaria', 'Jamur hitam', 'Bukan Mangga', 'Sehat', 'Busuk pangkal batang' ]
# Threshold confidence minimal untuk deteksi "bukan buah mangga"
CONFIDENCE_THRESHOLD = 0.5  # 50%

interpreter = None
input_details = None
output_details = None

# --- Load Model ---
def load_and_initialize_tflite():
    global interpreter, input_details, output_details

    if not os.path.exists(TFLITE_MODEL_PATH):
        print(f"ERROR: TFLite model tidak ditemukan: {TFLITE_MODEL_PATH}")
        return

    try:
        interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("TFLite model berhasil dimuat.")
    except Exception as e:
        print(f"Gagal memuat TFLite model: {e}")
        interpreter = None
        input_details = None
        output_details = None

load_and_initialize_tflite()


# --- Preprocess Image ---
def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(IMAGE_SIZE)

    img_array = np.array(img) / 255.0  # Jika training pakai normalisasi 0–1
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():
    if interpreter is None:
        return jsonify({'error': 'Model TFLite gagal dimuat. Cek log server.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang dikirim.'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Nama file kosong.'}), 400

    try:
        # Load gambar
        image_bytes = file.read()

        processed_image = preprocess_image(image_bytes)

        # Pastikan input sesuai dengan shape TFLite
        input_data = processed_image.astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])[0]

        best_idx = np.argmax(preds)
        best_label = CLASS_NAMES[best_idx]
        confidence = float(preds[best_idx])

        # Jika confidence di bawah threshold atau label 'Bukan Mangga', tidak perlu memberikan confidence
        if confidence < CONFIDENCE_THRESHOLD or best_label.lower().startswith("bukan"):
            return jsonify({
                "label": "Bukan buah mangga",
                "confidence": None,
                "predictions": []
            })

        all_pred = [
            {"label": CLASS_NAMES[i], "p": float(p)}
            for i, p in enumerate(preds)
        ]

        return jsonify({
            "label": best_label,
            "confidence": confidence,
            "predictions": all_pred
        })

    except Exception as e:
        return jsonify({"error": f"Kesalahan saat memproses gambar: {e}"}), 500


# if __name__ == '__main__':
#     app.run(debug=True)
