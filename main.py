import os
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")

app = Flask(__name__)

CORS(app)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
MODEL_PATH = 'model.pth.tar'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY
)

@app.route("/")
def hello():
    return "Timeweb Cloud + Flask = ❤️"

if __name__ == "__main__":
    port = 5000
    app.run(debug=True,host='0.0.0.0',port=port)