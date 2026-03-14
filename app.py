
from flask import Flask,render_template,request,jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    text = request.json["question"]
    pred = model.predict([text])[0]
    return jsonify({"prediction":pred})

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
