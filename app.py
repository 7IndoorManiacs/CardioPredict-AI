from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("cardiopredict_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    data = {
        "age": [float(request.form["age"])],
        "sex": [request.form["sex"]],
        "cp": [request.form["cp"]],
        "trestbps": [float(request.form["trestbps"])],
        "chol": [float(request.form["chol"])],
        "fbs": [request.form["fbs"] == "True"],
        "restecg": [request.form["restecg"]],
        "thalch": [float(request.form["thalch"])],
        "exang": [request.form["exang"] == "True"],
        "oldpeak": [float(request.form["oldpeak"])],
        "slope": [request.form["slope"]],
        "ca": [float(request.form["ca"])],
        "thal": [request.form["thal"]]
    }

    input_df = pd.DataFrame(data)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Risk Category + Color
    if probability < 0.30:
        risk = "Low Risk"
        color = "success"      # Green

    elif probability < 0.70:
        risk = "Moderate Risk"
        color = "warning"      # Yellow

    else:
        risk = "High Risk"
        color = "danger"       # Red

    result = (
        "Heart Disease Detected"
        if prediction == 1
        else "No Heart Disease Detected"
    )

    return render_template(
        "result.html",
        result=result,
        probability=round(probability * 100, 2),
        risk=risk,
        color=color
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)