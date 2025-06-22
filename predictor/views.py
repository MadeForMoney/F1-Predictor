import os
import pandas as pd
import numpy as np
import joblib
from django.shortcuts import render
from predictor.feat_gen import generate_features_for_inference
from .encoders import le_driver, le_track, le_team  # if used

# Load model once (at server start)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "xgboost_model.pkl"))

def predict_position(request):
    prediction = None
    csv_path = os.path.join(BASE_DIR, "feat_eng_f1.csv")
    df = pd.read_csv(csv_path)
    drivers = sorted(df["Driver"].unique())
    tracks = sorted(df["Track"].unique())

    if request.method == "POST":
        driver = request.POST.get("driver")
        track = request.POST.get("track")
        grid = int(request.POST.get("starting_grid"))

        features = generate_features_for_inference(driver, track, grid, df, year=2025)
        prediction = model.predict(features)[0]
        prediction=np.floor(prediction)

    return render(request, "f1_form.html", {
        "drivers": drivers,
        "tracks": tracks,
        "prediction": prediction
    })


def about_us(request):
    return render(request,'about.html')

def home_page(request):
    return render(request,'f1_index.html')
