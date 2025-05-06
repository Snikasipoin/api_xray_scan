import os
import torch
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

load_dotenv()

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_KEY:
    raise EnvironmentError("❌ Переменная окружения OPENROUTER_API_KEY не установлена")

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
MODEL_PATH = 'model.pth.tar'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
transform = None


def load_model():
    global model, transform
    if model is None:
        print("📦 Загрузка модели и зависимостей...")
        import torch.nn as nn
        import torchvision.models as models
        import torchvision.transforms as transforms

        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 14)

        checkpoint = torch.load(MODEL_PATH, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        corrected_state_dict = {
            k.replace("densenet121.", "").replace("classifier.0.", "classifier."): v
            for k, v in state_dict.items()
        }
        model.load_state_dict(corrected_state_dict)
        model.to(device)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.model.eval()
        self.target_layer.register_forward_hook(self.save_activations)
        self.target_layer.register_full_backward_hook(self.save_gradients)

    def save_activations(self, module, input, output):
        self.activations = output

    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):
        import numpy as np
        import cv2

        input_tensor = input_tensor.to(device)
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam = cam / np.max(cam)
        return cam, class_idx


def process_image(image_path):
    import numpy as np
    import cv2
    from PIL import Image

    load_model()

    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    grad_cam = GradCAM(model, model.features.denseblock4)
    heatmap, pred_class = grad_cam.generate(img_tensor)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.sigmoid(output)[0].cpu().numpy()

    img_cv = cv2.imread(image_path)
    img_cv = cv2.resize(img_cv, (224, 224))
    heatmap_img = np.uint8(255 * heatmap)
    heatmap_img = cv2.applyColorMap(heatmap_img, cv2.COLORMAP_JET)
    superimposed_img = heatmap_img * 0.4 + img_cv
    heatmap_path = os.path.join(RESULT_FOLDER, "heatmap.jpg")
    cv2.imwrite(heatmap_path, superimposed_img)

    return pred_class, image_path, heatmap_path, probs


def interpret_result(pred_class, probs):
    class_names = [
        "Норма", "Кардиомегалия", "Эмфизема", "Отек", "Грыжа", "Инфильтрация",
        "Масса", "Узелок", "Ателектаз", "Пневмония", "Плеврит", "Пневмоторакс",
        "Фиброз", "Консолидация"
    ]
    top_probs_idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:4]
    details = [
        {"label": class_names[i], "confidence": round(probs[i] * 100, 2)}
        for i in top_probs_idx
    ]
    top_3_idx = top_probs_idx[:3]
    summary = "\n".join([f"{class_names[i]}: {probs[i] * 100:.2f}%" for i in top_3_idx])
    return details, summary, top_3_idx[0]


def generate_medical_summary(summary_text: str) -> str:
    import httpx

    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://yourdomain.com",  # замените, если хотите
            "X-Title": "XRayScanApp"
        }

        payload = {
            "model": "deepseek/deepseek-prover-v2:free",
            "messages": [
                {"role": "system", "content": "Вы опытный врач-рентгенолог."},
                {"role": "user", "content": (
                    f"Вы рентгенолог. Вот вероятности по классам:\n{summary_text}\n"
                    "Сформулируйте медицинское заключение кратко, как в протоколе. "
                    "Выделите патологические находки и степень уверенности."
                )}
            ]
        }

        response = httpx.post("https://openrouter.ai/api/v1/chat/completions",
                              headers=headers, json=payload, timeout=60)

        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Ошибка получения заключения врача: {str(e)}"


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Файл не выбран."}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        pred_class, original_path, heatmap_path, probs = process_image(file_path)
        details, interpretation, _ = interpret_result(pred_class, probs)
        gpt_diagnosis = generate_medical_summary(interpretation)

        return jsonify({
            "original_url": f"/{original_path}",
            "heatmap_url": f"/{heatmap_path}",
            "interpretation": interpretation,
            "details": details,
            "gpt_diagnosis": gpt_diagnosis
        })

    except Exception as e:
        print(f"❌ Ошибка обработки: {e}")
        return jsonify({"error": f"Ошибка обработки: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
