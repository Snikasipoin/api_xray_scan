import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
from PIL import Image
from openai import OpenAI

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
openai_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_KEY) if OPENROUTER_KEY else None

def lazy_load_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 14)
    checkpoint = torch.load("model.pth.tar", map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict({k.replace("densenet121.", "").replace("classifier.0.", "classifier."): v for k, v in state_dict.items()})
    model.to(device)
    model.eval()
    return model

def process_image(path, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.sigmoid(output)[0].cpu().numpy()
        pred_class = np.argmax(probs)

    grad_cam = GradCAM(model, model.features.denseblock4)
    heatmap, _ = grad_cam.generate(tensor)
    heatmap_img = overlay_heatmap(path, heatmap)
    heatmap_path = "static/results/heatmap.jpg"
    cv2.imwrite(heatmap_path, heatmap_img)

    return pred_class, probs, heatmap_path

def overlay_heatmap(img_path, cam):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    cam = cv2.resize(np.uint8(255 * cam), (224, 224))
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    return np.uint8(0.4 * heatmap + 0.6 * img)

def interpret_result(pred_class, probs):
    labels = [
        "Норма", "Кардиомегалия", "Эмфизема", "Отек", "Грыжа", "Инфильтрация",
        "Масса", "Узелок", "Ателектаз", "Пневмония", "Плеврит", "Пневмоторакс",
        "Фиброз", "Консолидация"
    ]
    top3 = probs.argsort()[::-1][:3]
    return "\n".join([f"{labels[i]}: {probs[i]*100:.1f}%" for i in top3])

def generate_medical_summary(interpretation):
    if not openai_client:
        return {"conclusion": "API ключ не установлен."}

    try:
        response = openai_client.chat.completions.create(
            model="deepseek/deepseek-prover-v2:free",
            messages=[
                {"role": "system", "content": "Вы опытный врач-рентгенолог."},
                {"role": "user", "content": f"Вот вероятности:\n{interpretation}\nСформулируй краткое заключение, диагноз по разделам: Легочная ткань, Сердце и сосуды, Сравнительный анализ, Заключение"}
            ],
            temperature=0.5,
            max_tokens=400
        )
        content = response.choices[0].message.content
        sections = {"lungs": "", "heart": "", "comparison": "", "conclusion": ""}
        for key in sections:
            start = content.lower().find(key)
            if start != -1:
                end = content.find("\n", start)
                sections[key] = content[start:end].strip()
        return sections
    except Exception as e:
        return {"conclusion": f"Ошибка от API: {str(e)}"}

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _, __, output): self.activations = output
    def _save_gradient(self, _, grad_in, grad_out): self.gradients = grad_out[0]

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = torch.argmax(output)
        self.model.zero_grad()
        output[0, class_idx].backward()

        grads = self.gradients[0].cpu().numpy()
        acts = self.activations[0].cpu().numpy()
        weights = grads.mean(axis=(1, 2))

        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]
        return np.maximum(cam, 0) / np.max(cam), class_idx
