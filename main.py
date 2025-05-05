import os
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

# Загрузка переменных окружения
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

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Глобальные переменные
model = None
transform = None
client = None

def load_model():
    global model, transform, client
    if model is None:
        print("📦 Загрузка модели и зависимостей...")
        import torch.nn as nn
        import torchvision.models as models
        import torchvision.transforms as transforms
        from openai import OpenAI

        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 14)

        checkpoint = torch.load(MODEL_PATH, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        corrected_state_dict = {k.replace("densenet121.", "").replace("classifier.0.", "classifier."): v for k, v in state_dict.items()}
        model.load_state_dict(corrected_state_dict)

        model.to(device)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_KEY
        )

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
    import matplotlib.pyplot as plt

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

    class_names = [
        "Норма", "Кардиомегалия", "Эмфизема", "Отек", "Грыжа", "Инфильтрация",
        "Масса", "Узелок", "Ателектаз", "Пневмония", "Плеврит", "Пневмоторакс",
        "Фиброз", "Консолидация"
    ]
    top_probs_idx = np.argsort(probs)[::-1][:5]
    top_probs = probs[top_probs_idx]
    top_labels = [class_names[idx] for idx in top_probs_idx]

    plot_path = os.path.join(RESULT_FOLDER, "probs_plot.png")
    plt.figure(figsize=(10, 5))
    plt.bar(top_labels, top_probs * 100, color='skyblue')
    plt.xlabel('Классы')
    plt.ylabel('Вероятность (%)')
    plt.title('Топ-5 вероятностей')
    plt.xticks(rotation=45)
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

    return pred_class, image_path, heatmap_path, plot_path, probs

def interpret_result(pred_class, probs):
    class_names = [
        "Норма", "Кардиомегалия", "Эмфизема", "Отек", "Грыжа", "Инфильтрация",
        "Масса", "Узелок", "Ателектаз", "Пневмония", "Плеврит", "Пневмоторакс",
        "Фиброз", "Консолидация"
    ]
    top_probs_idx = np.argsort(probs)[::-1][:3]
    prob_str = "\n".join([f"{class_names[idx]}: {probs[idx] * 100:.2f}%" for idx in top_probs_idx])
    max_prob = probs[pred_class]
    if max_prob < 0.5:
        return f"Предсказание: Неопределенно (уверенность: {max_prob * 100:.2f}%)\nВероятности:\n{prob_str}"
    return f"Предсказание: {class_names[pred_class]}\nВероятности:\n{prob_str}"

def generate_medical_summary(interpretation: str) -> str:
    try:
        load_model()
        print("📡 Отправка запроса в OpenRouter...")
        response = client.chat.completions.create(
            model="deepseek/deepseek-prover-v2:free",
            messages=[
                {"role": "system", "content": "Вы опытный врач-рентгенолог."},
                {"role": "user", "content": (
                    "Вы рентгенолог. Проанализируйте результат работы ИИ-модели по рентгену грудной клетки. "
                    f"Вот вероятности по классам и вывод модели:\n{interpretation}\n\n"
                    "Сформулируйте медицинское заключение кратко, как врач бы написал его в протоколе. Укажите возможные патологии и уровень уверенности."
                )}
            ],
            temperature=0.5,
            max_tokens=300
        )
        print("✅ Ответ от OpenRouter получен.")
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("❌ Ошибка от OpenRouter API:")
        print(str(e))
        return f"Ошибка получения заключения врача: {str(e)}"

@app.route('/')
def index():
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

        pred_class, original_path, heatmap_path, plot_path, probs = process_image(file_path)
        interpretation = interpret_result(pred_class, probs)
        gpt_diagnosis = generate_medical_summary(interpretation)

        base_url = request.url_root.rstrip('/')
        return jsonify({
            "original_url": f"{base_url}/{original_path}",
            "heatmap_url": f"{base_url}/{heatmap_path}",
            "plot_url": f"{base_url}/{plot_path}",
            "interpretation": interpretation,
            "gpt_diagnosis": gpt_diagnosis
        })
    except Exception as e:
        print(f"❌ Ошибка обработки запроса: {e}")
        return jsonify({"error": f"Ошибка обработки: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
