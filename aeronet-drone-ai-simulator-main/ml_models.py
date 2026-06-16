from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

def load_bike_demand(path="data/raw/bike_train.csv"):
    p = Path(path)
    if not p.exists():
        rng = np.random.default_rng(42)
        n = 1000
        df = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=n, freq="h"),
            "season": rng.integers(1, 5, n),
            "holiday": rng.integers(0, 2, n),
            "workingday": rng.integers(0, 2, n),
            "weather": rng.integers(1, 4, n),
            "temp": rng.normal(25, 6, n),
            "humidity": rng.integers(30, 95, n),
            "windspeed": rng.normal(12, 5, n).clip(0),
            "count": rng.integers(10, 250, n)
        })
    else:
        df = pd.read_csv(
            p,
            on_bad_lines='skip',
            engine='python'
)

    df["datetime"] = pd.to_datetime(df["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    return df


def train_demand_model():
    df = load_bike_demand()
    features = ["season", "holiday", "workingday", "weather", "temp", "humidity", "windspeed", "hour", "dayofweek", "month"]
    X = df[features]
    y = df["count"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    return {
        "model": model,
        "mae": round(mean_absolute_error(y_test, pred), 3),
        "rmse": round(mean_squared_error(y_test, pred) ** 0.5, 3),
        "importance": importance,
        "sample": df.head(12),
        "mean_pred": round(float(np.mean(pred)), 3),
    }


def load_drone_log(path="data/raw/drone_operations_log.csv"):
    p = Path(path)
    if not p.exists():
        rng = np.random.default_rng(7)
        n = 700
        df = pd.DataFrame({
            "Battery Remaining (%)": rng.integers(5, 100, n),
            "GPS Accuracy (meters)": rng.normal(4, 2, n).clip(0),
            "Wind Speed (m/s)": rng.normal(5, 3, n).clip(0),
            "Altitude (meters)": rng.normal(70, 25, n).clip(10),
            "Flight Duration (minutes)": rng.normal(28, 10, n).clip(3),
            "Distance Flown (km)": rng.normal(9, 4, n).clip(1),
            "Obstacles Encountered": rng.choice(["Yes", "No"], n, p=[.25, .75]),
            "Flight Status": rng.choice(["Completed", "Warning", "Failed"], n, p=[.78, .15, .07]),
        })
    else:
        df = pd.read_csv(
        p,
        engine="python",
        on_bad_lines="skip"
)
    return df


def prepare_anomaly_labels(df):
    d = df.copy()

    for col in ["Battery Remaining (%)", "GPS Accuracy (meters)", "Wind Speed (m/s)", "Altitude (meters)", "Flight Duration (minutes)", "Distance Flown (km)"]:
        d[col] = pd.to_numeric(d[col], errors="coerce").fillna(d[col].median(numeric_only=True))

    obstacles = d["Obstacles Encountered"].astype(str).str.lower().str.contains("yes").astype(int)
    failed = d["Flight Status"].astype(str).str.lower().str.contains("fail|warning|incident").astype(int)

    labels = []
    for _, row in d.iterrows():
        if row["Battery Remaining (%)"] < 20:
            labels.append("battery_anomaly")
        elif row["GPS Accuracy (meters)"] > 8 or obstacles.loc[row.name] == 1:
            labels.append("route_anomaly")
        elif row["Wind Speed (m/s)"] > 5 or row["Altitude (meters)"] > 70:
            labels.append("sensor_weather_anomaly")
        elif failed.loc[row.name] == 1:
            labels.append("operational_anomaly")
        else:
            labels.append("normal")

    d["label"] = labels
    return d


def train_anomaly_model():
    df = prepare_anomaly_labels(load_drone_log())
    features = ["Battery Remaining (%)", "GPS Accuracy (meters)", "Wind Speed (m/s)", "Altitude (meters)", "Flight Duration (minutes)", "Distance Flown (km)"]
    X = df[features]
    y = df["label"]

    # If one class dominates, use stratify only if possible.
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42, stratify=stratify)

    models = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
    }

    results = []
    best = None
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        results.append({"name": name, "model": model, "accuracy": round(float(acc), 3), "pred": pred})
        if best is None or acc > best["accuracy"]:
            best = results[-1]

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, best["pred"], labels=labels)
    report = classification_report(y_test, best["pred"], zero_division=0)

    return {
        "df": df,
        "features": features,
        "results": pd.DataFrame([{"Model": r["name"], "Accuracy": r["accuracy"]} for r in results]),
        "best_name": best["name"],
        "accuracy": best["accuracy"],
        "labels": labels,
        "cm": cm,
        "report": report,
    }
