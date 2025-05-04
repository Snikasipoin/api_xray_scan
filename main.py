import os
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename

load_dotenv()

app = Flask(__name__)

CORS(app)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
MODEL_PATH = 'model.pth.tar'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.densenet121(weights=None)
num_classes = 14
model.classifier = nn.Linear(model.classifier.in_features, num_classes)

try:
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    corrected_state_dict = {k.replace("densenet121.", "").replace("classifier.0.", "classifier."): v for k, v in state_dict.items()}
    model.load_state_dict(corrected_state_dict)
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")
    exit(1)

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


@app.route("/")
def hello():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Файл не выбран."}), 400
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # pred_class, original_path, heatmap_path, plot_path, probs = process_image(file_path)
        # interpretation = interpret_result(pred_class, probs)
        # gpt_diagnosis = generate_medical_summary(interpretation)
        #
        # base_url = request.url_root.rstrip('/')
        # return jsonify({
        #     "original_url": f"{base_url}/{original_path}",
        #     "heatmap_url": f"{base_url}/{heatmap_path}",
        #     "plot_url": f"{base_url}/{plot_path}",
        #     "interpretation": interpretation,
        #     "gpt_diagnosis": gpt_diagnosis
        # })
    except Exception as e:
        print(f"❌ Ошибка обработки запроса: {e}")
        return jsonify({"error": f"Ошибка обработки: {str(e)}"}), 500

if __name__ == "__main__":
    port = 5000
    app.run(debug=True,host='0.0.0.0',port=port)