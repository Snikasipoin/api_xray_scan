import os
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torchvision.models as models

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

@app.route("/")
def hello():
    return render_template('index.html')

if __name__ == "__main__":
    port = 5000
    app.run(debug=True,host='0.0.0.0',port=port)