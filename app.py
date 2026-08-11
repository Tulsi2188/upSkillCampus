from flask import Flask, render_template, request
import joblib
import pandas as pd
from datetime import datetime

# Create Flask application
app = Flask(__name__)

# Load the trained model
model = joblib.load("traffic_model.pkl")


# Home Page
@app.route("/")
def home():
    return render_template("home.html")


# Dashboard Page
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# Prediction Page
@app.route("/prediction", methods=["GET", "POST"])
def prediction():

    prediction = None
    traffic_level = None
    summary = None

    if request.method == "POST":

        junction = int(request.form["junction"])

        date = request.form["date"]

        hour = int(request.form["hour"])

        dt = pd.to_datetime(date)

        year = dt.year
        month = dt.month
        day = dt.day
        dayofweek = dt.dayofweek

        isweekend = 1 if dayofweek >= 5 else 0
        quarter = dt.quarter
        weekofyear = dt.isocalendar().week
        dayofyear = dt.dayofyear

        X = pd.DataFrame([[
            junction,
            year,
            month,
            day,
            hour,
            dayofweek,
            isweekend,
            quarter,
            weekofyear,
            dayofyear
        ]],
        columns=[
            "Junction",
            "Year",
            "Month",
            "Day",
            "Hour",
            "DayOfWeek",
            "IsWeekend",
            "Quarter",
            "WeekOfYear",
            "DayOfYear"
        ])

        prediction = round(model.predict(X)[0], 2)

        if prediction <= 20:
            traffic_level = "🟢 Low Traffic"

        elif prediction <= 50:
            traffic_level = "🟡 Moderate Traffic"

        else:
            traffic_level = "🔴 Heavy Traffic"

        summary = {
            "junction": junction,
            "date": dt.strftime("%d %B %Y"),
            "hour": f"{hour:02d}:00"
        }

    return render_template(
        "prediction.html",
        prediction=prediction,
        traffic_level=traffic_level,
        summary=summary
    )


# Performance Page
@app.route("/performance")
def performance():

    mae = 14.52
    rmse = 20.55
    r2 = 0.37

    return render_template(
        "performance.html",
        mae=mae,
        rmse=rmse,
        r2=r2
    )


# About Page
@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)