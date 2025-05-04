import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from PIL import Image
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv

# load_dotenv()

app = Flask(__name__)

# OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
# if not OPENROUTER_KEY:
#     raise EnvironmentError("❌ Переменная окружения OPENROUTER_API_KEY не установлена")
#
# print("✅ OPENROUTER_API_KEY загружен")

@app.route("/")
def hello():
    return "Timeweb Cloud + Flask = ❤️"

if __name__ == "__main__":
    port = 5000
    app.run(debug=True,host='0.0.0.0',port=port)