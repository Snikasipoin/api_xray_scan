import os
import traceback
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from chexnet_model import process_image, interpret_result, generate_medical_summary, lazy_load_model

load_dotenv()

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({"error": "Файл не выбран"}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        model = lazy_load_model()
        pred_class, probs, heatmap_path = process_image(file_path, model)
        interpretation = interpret_result(pred_class, probs)
        gpt_analysis = generate_medical_summary(interpretation)

        class_names = [
            "Норма", "Кардиомегалия", "Эмфизема", "Отек", "Грыжа", "Инфильтрация",
            "Масса", "Узелок", "Ателектаз", "Пневмония", "Плеврит", "Пневмоторакс",
            "Фиброз", "Консолидация"
        ]

        top_indices = probs.argsort()[::-1][:4]
        details = [
            {"label": class_names[i], "confidence": f"{probs[i]*100:.1f}%"}
            for i in top_indices
        ]

        return jsonify({
            "image_url": f"{request.url_root}static/uploads/{filename}",
            "heatmap_url": f"{request.url_root}{heatmap_path}",
            "details": details,
            "detailed_analysis": {
                "Легочная ткань": gpt_analysis.get("lungs", ""),
                "Сердце и сосуды": gpt_analysis.get("heart", ""),
                "Сравнительный анализ": gpt_analysis.get("comparison", "")
            },
            "full_report": gpt_analysis.get("conclusion", "")
        })
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": f"Ошибка обработки: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
