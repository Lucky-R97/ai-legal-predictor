
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

if __name__=="__main__":
    app.run(debug=True)
