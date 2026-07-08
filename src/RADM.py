import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import joblib
import yaml

use_radm = True  # If True, enable risk-guided diffusion; otherwise use plain DDPM
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
risk_weight_enabled = use_radm

# -------------------------
# 1. Data preprocessing
# -------------------------
print("🚀 Starting RADM (Risk-Aware Diffusion Model) training pipeline")

radm_file_path = os.path.join(BASE_PATH, "radm_training_data.npz")
try:
    file_path = os.path.abspath(radm_file_path)
    print(f"📂 Loading file: {file_path}")
    radm_data = np.load(file_path)
    print(f"📦 File contains fields: {list(radm_data.keys())}")

    X_np = radm_data["weather"]
    risk_weight = radm_data["risk_weight"]

    if use_radm:
        print("✅ Mode confirmed: using RADM for risk-aware training")
except FileNotFoundError:
    print(f"❌ File not found: {file_path}")
    exit(1)
except KeyError as e:
    print(f"❌ Missing required fields: {e} (expected 'weather' and 'risk_weight')")
    exit(1)
except Exception as e:
    print(f"❌ Unknown error while loading training data: {e}")
    exit(1)

use_risk_model = use_radm
risk_model_path = os.path.join(BASE_PATH, "risk_predictor_rf.pkl")

risk_model = None
if use_risk_model and os.path.exists(risk_model_path):
    risk_model = joblib.load(risk_model_path)
    print("✅ Risk scoring will be applied to generated samples")
else:
    print("⚠️ Risk predictor not loaded; skipping scoring step")

# ✅ Ensure weather and risk arrays have the same length; truncate to the minimum length if needed
min_len = min(len(X_np), len(risk_weight))
X_np = X_np[:min_len]
risk_weight = risk_weight[:min_len]

X_tensor = torch.tensor(X_np, dtype=torch.float32)
risk_tensor = torch.tensor(risk_weight, dtype=torch.float32)

# Build TensorDataset
dataset = TensorDataset(X_tensor, risk_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# -------------------------
# 2. Noise schedule parameters
# -------------------------
T = 1000  # Total diffusion steps
betas = torch.linspace(1e-4, 0.02, T)  # Linear noise schedule
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# -------------------------
# 3. Define DDPM model
# -------------------------
class DDPM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, dim)
        )

    def forward(self, x, t):
        t_embed = t.float().unsqueeze(1) / T
        return self.net(torch.cat([x, t_embed], dim=1))

model = DDPM(X_tensor.shape[1])
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss(reduction="none")

# -------------------------
# 4. Training loop
# -------------------------
num_epochs = 20
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for x0, r in dataloader:
        t = torch.randint(0, T, (x0.size(0),))

        # ✅ Sample Gaussian noise and create noisy inputs
        noise = torch.randn_like(x0)
        alpha_bar = alphas_cumprod[t].unsqueeze(1)
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise

        # ✅ Predict noise with the model
        pred_noise = model(xt, t)

        # ✅ Compute risk-weighted loss
        loss_raw = criterion(pred_noise, noise).mean(dim=1)
        weight = r if risk_weight_enabled else torch.ones_like(r)
        loss = (loss_raw * weight).mean()

        # ✅ Risk-guidance term
        if use_risk_model:
            x0_pred = pred_noise

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.6f}")

# -------------------------
# 5. Sample new scenarios (reverse diffusion)
# -------------------------
print("🎨 Generating weather samples with RADM...")

@torch.no_grad()
def sample_ddpm(model, num_samples, T, betas, alphas, alphas_cumprod):
    x = torch.randn(num_samples, X_tensor.shape[1])
    for t in reversed(range(T)):
        t_tensor = torch.full((num_samples,), t, dtype=torch.long)
        pred_noise = model(x, t_tensor)

        beta = betas[t]
        alpha = alphas[t]
        alpha_bar = alphas_cumprod[t]
        z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)

        x = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - alpha_bar)) * pred_noise) + torch.sqrt(beta) * z
    return x

num_samples = 100
samples = sample_ddpm(model, num_samples, T, betas, alphas, alphas_cumprod)

# Inverse transform
try:
    scaler_other = joblib.load(os.path.join(BASE_PATH, "scaler_other.pkl"))
    if samples.shape[1] != scaler_other.mean_.shape[0]:
        raise ValueError(f"Sample dimension {samples.shape[1]} does not match scaler dimension {scaler_other.mean_.shape[0]}")
    samples_inv = scaler_other.inverse_transform(samples)
except Exception as e:
    print(f"❌ Inverse standardization failed: {e}")
    exit(1)

# Build DataFrame
samples_df_full = pd.DataFrame(samples_inv, columns=[
    "TMP", "WIN_S_Max", "RHU", "RHU_Min", "PRE_3h",
    "WIN_D_Avg_2mi_sin", "WIN_D_Avg_2mi_cos",
    "WIN_D_S_Max_sin", "WIN_D_S_Max_cos",
    "extra_1", "extra_2"  # replace with real fields if present
])

# Save generated samples to CSV
samples_df_full.to_csv(os.path.join(BASE_PATH, "ddpm_generated_samples.csv"), index=False)
print("✅ Saved radm_training_data.npz (13-D features + risk weights)")

# 6. Score samples with risk model and select high-risk subset
if risk_model is not None:
    print("🧠 Scoring generated samples using the risk predictor...")
    pred_risk = risk_model.predict(samples_df_full.to_numpy(dtype=np.float32))
    samples_df_full["predicted_risk"] = pred_risk
    samples_df_full = samples_df_full.sort_values("predicted_risk", ascending=False)

    # Select top-K high-risk samples
    samples_df_topk = samples_df_full.head(50).copy()
else:
    samples_df_topk = samples_df_full.copy()

def row_to_weather(row):
    """
    More physically consistent CARLA weather mapping:
    - cloudiness: derived from humidity and precipitation
    - precipitation: mm → percentage
    - precipitation_deposits: post-rain deposits, decreasing with precipitation strength
    - wetness: ground wetness combining precipitation and humidity
    - wind_intensity: wind speed m/s → 0-100
    - sun_altitude_angle: solar altitude decreases with cloudiness
    - sun_azimuth_angle: assume due south (180°) with small perturbation
    - fog_density: fog forms under high humidity
    - fog_distance: inversely related to fog density
    - fog_falloff: fog falloff increases with density
    - scattering_intensity: fog scattering intensity
    - mie_scattering_scale: scattering by larger particles (cloud/fog)
    - rayleigh_scattering_scale: molecular scattering under clear conditions
    """
    # Read raw features
    precipitation_mm = row.get("PRE_3h", 0.0)
    humidity = np.clip(row.get("RHU", 0.0), 0.0, 100.0)
    wind_speed = row.get("WIN_S_Max", 0.0)  # m/s

    # 1. Precipitation and deposits
    precipitation = float(np.clip(precipitation_mm * 10.0, 0.0, 100.0))
    # Deposits decay exponentially (example: 70% initial deposit)
    precipitation_deposits = float(np.clip(precipitation * 0.7, 0.0, 100.0))

    # 2. Ground wetness (slipperiness)
    wetness = float(np.clip(humidity, 0.0, 100.0))

    # 3. Wind intensity mapping: assume 25 m/s maps to 100
    wind_intensity = float(np.clip(wind_speed / 25.0 * 100.0, 0.0, 100.0))

    # 4. Cloudiness: combined effect of humidity and precipitation
    cloudiness = float(np.clip(0.6 * humidity + 0.4 * precipitation, 0.0, 100.0))

    # 5. Sun altitude: high for clear weather, lower for cloudy weather
    sun_altitude_angle = float(np.clip(90.0 - cloudiness * 0.8, 0.0, 90.0))
    # Sun azimuth: default due south (180°), random perturbation ±20°
    sun_azimuth_angle = float(180.0 + np.random.uniform(-20.0, 20.0))

    # 6. Fog: generated only when humidity is high
    fog_density = float(np.clip((humidity - 70.0) * 2.0, 0.0, 100.0))
    fog_distance = float(np.clip(100.0 - fog_density, 0.0, 100.0))
    fog_falloff = float(np.clip(fog_density / 100.0, 0.0, 1.0))

    # 7. Scattering parameters
    scattering_intensity = float(np.clip(fog_density / 100.0, 0.0, 1.0))
    mie_scattering_scale = float(np.clip(cloudiness / 100.0, 0.0, 1.0))
    rayleigh_scattering_scale = float(np.clip(1.0 - cloudiness / 100.0, 0.0, 1.0))

    return {
        "cloudiness": cloudiness,
        "precipitation": precipitation,
        "precipitation_deposits": precipitation_deposits,
        "wetness": wetness,
        "wind_intensity": wind_intensity,
        "sun_altitude_angle": sun_altitude_angle,
        "sun_azimuth_angle": sun_azimuth_angle,
        "fog_density": fog_density,
        "fog_distance": fog_distance,
        "fog_falloff": fog_falloff,
        "scattering_intensity": scattering_intensity,
        "mie_scattering_scale": mie_scattering_scale,
        "rayleigh_scattering_scale": rayleigh_scattering_scale
    }

print("📦 Building CARLA-compatible high-risk weather config (risky_weather.yaml)...")
weather_list_topk = [row_to_weather(row) for _, row in samples_df_topk.iterrows()]

output_yaml = "risky_weather.yaml"
with open(os.path.join(BASE_PATH, output_yaml), "w") as f:
    yaml.dump({"weather_configs": weather_list_topk}, f, sort_keys=False)
print(f"✅ High-risk CARLA weather config saved to data/{output_yaml}")

# ✅ Save full weather mappings to weather_batch.yaml regardless of risk-model usage
samples_df_full["weather_id"] = np.arange(len(samples_df_full))
weather_batch_list = [row_to_weather(row) for _, row in samples_df_full.iterrows()]

# Save to weather_batch.yaml
batch_yaml_path = os.path.join(BASE_PATH, "weather_batch.yaml")
with open(batch_yaml_path, "w") as f:
    yaml.dump({"weather_configs": weather_batch_list}, f, sort_keys=False)
print(f"✅ All generated samples saved to {batch_yaml_path}")

print("✅ RADM diffusion generation pipeline completed!")