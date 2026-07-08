import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor

base_dir = os.path.dirname(__file__)
# Set data directory path
data_dir = os.path.abspath(os.path.join(base_dir, "..", "data"))

# 1. Load .npz file
npz_path = os.path.join(data_dir, "radm_training_data.npz")
data = np.load(npz_path)
X = data["weather"]
y = data["risk_weight"]

# 2. Train model
model = RandomForestRegressor(n_estimators=200, random_state=0)
model.fit(X, y)

# 3. Save model
model_path = os.path.join(data_dir, "risk_predictor_rf.pkl")
joblib.dump(model, model_path)
print("✅ Risk predictor saved to", model_path)

# 4. Optional: predict risks on training set and save
# pred = model.predict(X)
# df = pd.DataFrame(X)
# df["predicted_risk"] = pred
# df.to_csv(os.path.join(data_dir, "weather_with_predicted_risk.csv"), index=False)
# print("📄 Risk predictions written to", os.path.join(data_dir, "weather_with_predicted_risk.csv"))