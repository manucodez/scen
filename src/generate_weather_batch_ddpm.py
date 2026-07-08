import os
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# ==== Configuration ====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TXT_INPUT = os.path.join(DATA_DIR, "weather.txt")
YAML_OUTPUT = os.path.join(DATA_DIR, "weather_batch.yaml")

# ==== Step 1. Load and clean data ====
print("📂 Reading text file:", TXT_INPUT)
raw_df = pd.read_csv(TXT_INPUT, sep=r"\s+", encoding="utf-8")
raw_df.columns = raw_df.columns.str.strip()
raw_df.rename(columns={"TEM": "TMP"}, inplace=True)

# Convert wind direction angle to sin/cos
raw_df["WIN_D_Avg_2mi_sin"] = np.sin(np.deg2rad(raw_df["WIN_D_Avg_2mi"]))
raw_df["WIN_D_Avg_2mi_cos"] = np.cos(np.deg2rad(raw_df["WIN_D_Avg_2mi"]))
raw_df["WIN_D_S_Max_sin"] = np.sin(np.deg2rad(raw_df["WIN_D_S_Max"]))
raw_df["WIN_D_S_Max_cos"] = np.cos(np.deg2rad(raw_df["WIN_D_S_Max"]))

# Feature extraction
temperature_col = "TMP"
features = [
    "RHU", "RHU_Min", "PRE_3h", "WIN_S_Max", temperature_col,
    "WIN_D_Avg_2mi_sin", "WIN_D_Avg_2mi_cos",
    "WIN_D_S_Max_sin", "WIN_D_S_Max_cos"
]
X = raw_df[features].astype(np.float32).to_numpy()

print("✅ Feature example:")
print(raw_df[features].head())

# ==== Step 2. Standardization ====
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib_path = os.path.join(DATA_DIR, "scaler_other.pkl")

import joblib
joblib.dump(scaler, joblib_path)

dataset = TensorDataset(torch.tensor(X_scaled))
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# ==== Step 3. Model definition ====
class DDPM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim+1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, dim)
        )

    def forward(self, x, t):
        t_embed = t.float().unsqueeze(1) / T
        return self.net(torch.cat([x, t_embed], dim=1))

# ==== Step 4. Training ====
T = 1000
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

model = DDPM(X.shape[1])
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

num_epochs = 20
print("🧠 Starting DDPM training ...")

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch in dataloader:
        x0 = batch[0]
        t = torch.randint(0, T, (x0.size(0),))

        noise = torch.randn_like(x0)
        alpha_bar = alphas_cumprod[t].unsqueeze(1)
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise

        pred_noise = model(xt, t)
        loss = criterion(pred_noise, noise)

        if torch.isnan(loss):
            print(f"❌ NaN loss detected at epoch {epoch+1}; skipping this batch")
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.6f}")

# ==== Step 5. Sampling ====
def sample_ddpm(model, num_samples):
    x = torch.randn(num_samples, X.shape[1])
    for t in reversed(range(T)):
        t_tensor = torch.full((num_samples,), t, dtype=torch.long)
        pred_noise = model(x, t_tensor)

        if torch.isnan(pred_noise).any():
            print(f"⚠️ NaN predicted at step {t}; stopping sampling")
            break

        beta = betas[t]
        alpha = alphas[t]
        alpha_bar = alphas_cumprod[t]
        z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        x = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_bar)) * pred_noise) + torch.sqrt(beta) * z
    return x

print("🎨 Generating new samples...")
model.eval()
with torch.no_grad():
    samples = sample_ddpm(model, num_samples=100).cpu().numpy()
samples_inv = scaler.inverse_transform(samples)

# ==== Step 6. Map to weather parameters ====
samples_df = pd.DataFrame(samples_inv, columns=features)

def row_to_weather(row):
    return {
        "cloudiness": float(np.clip(row["RHU"], 0, 100)),
        "precipitation": float(np.clip(row["PRE_3h"] * 10.0, 0, 100)),
        "precipitation_deposits": float(np.clip(row["PRE_3h"] * 7.0, 0, 100)),
        "wetness": float(np.clip(row["RHU_Min"], 0, 100)),
        "wind_intensity": float(np.clip(row["WIN_S_Max"] * 4.0, 0, 100)),
        "sun_altitude_angle": float(np.clip(90 - row["RHU"], 0, 90)),
        "sun_azimuth_angle": float(180),
        "fog_density": float(np.clip((row["RHU"] - 70) * 2, 0, 100)),
        "fog_distance": float(np.clip(100 - row["RHU"], 0, 100)),
        "fog_falloff": float(np.clip(row["RHU"] / 100, 0, 1)),
        "scattering_intensity": float(np.clip(row["RHU"] / 100, 0, 1)),
        "mie_scattering_scale": float(np.clip(row["RHU"] / 100, 0, 1)),
        "rayleigh_scattering_scale": float(np.clip(1 - row["RHU"] / 100, 0, 1)),
    }

weather_list = [row_to_weather(row) for _, row in samples_df.iterrows()]
print(f"📦 Writing YAML file: {YAML_OUTPUT}")
with open(YAML_OUTPUT, "w") as f:
    yaml.dump({"weather_configs": weather_list}, f, sort_keys=False)

print("✅ Training finished. High-quality weather configurations generated ✅")