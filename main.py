import os
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import base64
from io import BytesIO

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 載入模型
lstm_model = load_model("models/lstm_model.h5")
inception_model = load_model("models/inception_model.h5")
rf_model = joblib.load("models/rf_model.pkl")

# 範例資料載入（可以替換成即時抓取）
df_example = pd.read_csv("models/sample_data.csv")  # 假設有 sample_data.csv

def predict_model(model_name, X):
    X_input = np.expand_dims(X, axis=0)  # LSTM / Inception
    if model_name == "lstm":
        return lstm_model.predict(X_input).flatten()
    elif model_name == "inception":
        return inception_model.predict(X_input).flatten()
    elif model_name == "rf":
        return rf_model.predict(X)
    else:
        return []

def plot_prediction(y_true, y_pred):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_true, mode='lines+markers', name='True'))
    fig.add_trace(go.Scatter(y=y_pred, mode='lines+markers', name='Predicted'))
    buf = BytesIO()
    fig.write_image(buf, format="png")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction_img": None})

@app.post("/predict", response_class=HTMLResponse)
def predict(request: Request,
            model_name: str = Form(...),
            n_samples: int = Form(...)):
    
    # 從範例資料取最近 n_samples
    X = df_example.values[-n_samples:]
    y_true = X[:, 0]  # 假設第一欄為 Close
    y_pred = predict_model(model_name, X)
    
    pred_img = plot_prediction(y_true, y_pred)
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction_img": pred_img,
        "model_name": model_name
    })
