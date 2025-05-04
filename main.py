import os
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

@app.route("/")
def hello():
    return "Timeweb Cloud + Flask = ❤️"

if __name__ == "__main__":
    port = 5000
    app.run(debug=True,host='0.0.0.0',port=port)