from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = os.path.join("models", "model_joblib.pkl")
data = joblib.load(MODEL_PATH)
model = data["model"]
cols = data["columns"]

def prepare_input(df):
    proc = pd.get_dummies(df, drop_first=True).fillna(0)
    for c in cols:
        if c not in proc.columns:
            proc[c] = 0
    proc = proc[cols]
    return proc

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.form:
        data_in = {}
        for k, v in request.form.items():
            if k == "Weekend":
                data_in["Weekend"] = 1 if v in ["on", "1", "true", "True"] else 0
                continue
            try:
                data_in[k] = float(v)
            except:
                data_in[k] = v
        df = pd.DataFrame([data_in])
        X = prepare_input(df)
        prob = float(model.predict_proba(X)[0,1])
        pred = int(model.predict(X)[0])
        return render_template("result.html", prediction=pred, probability=prob)
    data_in = request.get_json(force=True)
    if isinstance(data_in, dict):
        df = pd.DataFrame([data_in])
    else:
        df = pd.DataFrame(data_in)
    X = prepare_input(df)
    probs = model.predict_proba(X)[:,1].tolist()
    preds = model.predict(X).astype(int).tolist()
    out = [{"prediction": int(p), "probability": float(pr)} for p, pr in zip(preds, probs)]
    return jsonify(out)

if __name__ == "__main__":
    app.run(debug=True)
