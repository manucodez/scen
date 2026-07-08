import os
import yaml
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib  # only needed if not already imported

# ========== Build base paths ==========
base_dir = os.path.dirname(__file__)
data_dir = os.path.abspath(os.path.join(base_dir, "..", "data"))

weather_file_path = os.environ.get("WEATHER_FILE", os.path.join(data_dir, "weather_batch.yaml"))  # can be overridden by external scripts
feedback_file_path = os.environ.get("FEEDBACK_FILE", os.path.join(data_dir, "feedback.csv"))

# === 1. Load weather parameters ===
weather_yaml = yaml.safe_load(open(weather_file_path, "r", encoding="utf-8")) if weather_file_path.endswith(".yaml") else None
weather_df = pd.DataFrame(weather_yaml["weather_configs"])
weather_df["weather_id"] = np.arange(len(weather_df))

# === 2. Load feedback data (e.g., collision_rate) ===
feedback_df = pd.read_csv(feedback_file_path)

# === 3. Merge data (align by weather_id) ===
merged_df = pd.merge(weather_df, feedback_df, on="weather_id", how="inner")

# === 4. Train a lightweight risk predictor ===
feature_cols = [c for c in merged_df.columns if c not in ["weather_id", "collision_rate"]]
X = merged_df[feature_cols].to_numpy(dtype=np.float32)
y = merged_df["collision_rate"].to_numpy(dtype=np.float32)

scaler_other = StandardScaler()
X_scaled = scaler_other.fit_transform(X)
joblib.dump(scaler_other, os.path.join(data_dir, "scaler_other.pkl"))
print("✅ Saved scaler_other.pkl (for inverse-standardizing wind, humidity, etc.)")

model = RandomForestRegressor(n_estimators=200, random_state=0)
model.fit(X_scaled, y)

joblib.dump(model, os.path.join(data_dir, "risk_predictor_rf.pkl"))
risk_pred = model.predict(X_scaled)

# === 5. Predict risk values and normalize ===
risk_scaler = MinMaxScaler()
risk_weight = risk_scaler.fit_transform(risk_pred.reshape(-1, 1)).flatten()

joblib.dump(risk_scaler, os.path.join(data_dir, "scaler_risk_weight.pkl"))

# === 6. Save .npz file for DDPM training ===
np.savez(
    os.path.join(data_dir, "radm_training_data.npz"),
    weather=X_scaled.astype(np.float32),
    risk_weight=risk_weight.astype(np.float32)
)
print("✅ Training data generated: data/radm_training_data.npz")
print("✅ Saved normalizers and risk predictor")