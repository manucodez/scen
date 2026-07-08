import os
import yaml
import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# Configuration
# ============================================================

USE_RADM = True          # False -> plain DDPM
USE_RISK_MODEL = True    # Score generated samples

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))

RADM_DATA = os.path.join(DATA_DIR, "radm_training_data.npz")
RISK_MODEL = os.path.join(DATA_DIR, "risk_predictor_rf.pkl")
SCALER = os.path.join(DATA_DIR, "scaler_other.pkl")

OUTPUT_BATCH = os.path.join(DATA_DIR, "weather_batch.yaml")
OUTPUT_RISKY = os.path.join(DATA_DIR, "risky_weather.yaml")

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("RADM : Risk-Aware Diffusion Model")
print("=" * 60)

print(f"Device : {DEVICE}")
print(f"Data   : {RADM_DATA}")
# ============================================================
# Load training data
# ============================================================

print("\nLoading training dataset...")

try:
    data = np.load(RADM_DATA)

    X_np = data["weather"].astype(np.float32)
    risk_weight = data["risk_weight"].astype(np.float32)

    print(f"Weather samples : {X_np.shape}")
    print(f"Risk weights    : {risk_weight.shape}")

except Exception as e:
    print(f"❌ Failed to load training data")
    print(e)
    exit(1)

# ------------------------------------------------------------
# Safety check
# ------------------------------------------------------------

if len(X_np) != len(risk_weight):
    raise ValueError(
        f"Mismatch between weather ({len(X_np)}) "
        f"and risk weights ({len(risk_weight)})"
    )

# ------------------------------------------------------------
# Load Random Forest
# ------------------------------------------------------------

risk_model = None

if USE_RISK_MODEL:

    if not os.path.exists(RISK_MODEL):
        raise FileNotFoundError(RISK_MODEL)

    risk_model = joblib.load(RISK_MODEL)

    print("✅ Random Forest loaded")

# ------------------------------------------------------------
# Tensor Dataset
# ------------------------------------------------------------

X_tensor = torch.tensor(
    X_np,
    dtype=torch.float32,
    device=DEVICE
)

risk_tensor = torch.tensor(
    risk_weight,
    dtype=torch.float32,
    device=DEVICE
)

dataset = TensorDataset(
    X_tensor,
    risk_tensor
)

dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True
)

print(f"Mini-batches : {len(dataloader)}")
# ============================================================
# Diffusion Noise Schedule
# ============================================================

T = 1000

betas = torch.linspace(
    1e-4,
    2e-2,
    T,
    device=DEVICE
)

alphas = 1.0 - betas

alpha_bar = torch.cumprod(
    alphas,
    dim=0
)

# ============================================================
# DDPM Network
# ============================================================

class DDPM(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(input_dim + 1, 128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, input_dim)
        )

    def forward(self, x, t):

        t = t.float().unsqueeze(1) / T

        x = torch.cat([x, t], dim=1)

        return self.model(x)


# ============================================================
# Initialize
# ============================================================

input_dim = X_tensor.shape[1]

model = DDPM(input_dim).to(DEVICE)

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3
)

criterion = nn.MSELoss(
    reduction="none"
)

print(f"Input dimension : {input_dim}")
print("DDPM initialized")
# ============================================================
# Train RADM
# ============================================================

print("\nStarting RADM training...\n")

EPOCHS = 20

model.train()

for epoch in range(EPOCHS):

    epoch_loss = 0.0

    for x0, risk in dataloader:

        # ------------------------------------------
        # Random diffusion timestep
        # ------------------------------------------

        t = torch.randint(
            0,
            T,
            (x0.size(0),),
            device=DEVICE
        )

        # ------------------------------------------
        # Forward diffusion
        # ------------------------------------------

        noise = torch.randn_like(x0)

        alpha_bar_t = alpha_bar[t].unsqueeze(1)

        xt = (
            torch.sqrt(alpha_bar_t) * x0
            + torch.sqrt(1.0 - alpha_bar_t) * noise
        )

        # ------------------------------------------
        # Predict noise
        # ------------------------------------------

        predicted_noise = model(xt, t)

        # ------------------------------------------
        # DDPM loss
        # ------------------------------------------

        sample_loss = criterion(
            predicted_noise,
            noise
        ).mean(dim=1)

        # ------------------------------------------
        # RADM weighting
        # ------------------------------------------

        if USE_RADM:

            weights = 1.0 + risk

        else:

            weights = torch.ones_like(risk)

        loss = (sample_loss * weights).mean()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} "
        f"Loss = {avg_loss:.6f}"
    )

print("\nTraining finished.\n")
# ============================================================
# Reverse Diffusion Sampling
# ============================================================

print("Generating new weather samples...\n")

model.eval()


@torch.no_grad()
def sample_ddpm(model, num_samples):

    x = torch.randn(
        num_samples,
        input_dim,
        device=DEVICE
    )

    for t in reversed(range(T)):

        t_tensor = torch.full(
            (num_samples,),
            t,
            dtype=torch.long,
            device=DEVICE
        )

        pred_noise = model(x, t_tensor)

        beta = betas[t]
        alpha = alphas[t]
        alpha_bar_t = alpha_bar[t]

        if t > 0:

            noise = torch.randn_like(x)

        else:

            noise = torch.zeros_like(x)

        x = (
            (x - (beta / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise)
            / torch.sqrt(alpha)
        ) + torch.sqrt(beta) * noise

    return x


NUM_SAMPLES = 300

samples = sample_ddpm(
    model,
    NUM_SAMPLES
)

samples = samples.cpu().numpy()

print("Generated samples :", samples.shape)
# ============================================================
# Inverse Standardization
# ============================================================

print("\nRecovering original weather values...")

scaler = joblib.load(SCALER)

samples_original = scaler.inverse_transform(samples)

print("Recovered shape :", samples_original.shape)
# ============================================================
# Build DataFrame
# ============================================================

CARLA_COLUMNS = [

    "cloudiness",
    "precipitation",
    "precipitation_deposits",
    "wetness",
    "wind_intensity",
    "sun_altitude_angle",
    "sun_azimuth_angle",
    "fog_density",
    "fog_distance",
    "fog_falloff",
    "scattering_intensity",
    "mie_scattering_scale",
    "rayleigh_scattering_scale",
]

samples_df = pd.DataFrame(
    samples_original,
    columns=CARLA_COLUMNS
)

print(samples_df.head())
samples_original = scaler.inverse_transform(samples)
# ============================================================
# Post-process generated weather
# ============================================================

samples_df = pd.DataFrame(
    samples_original,
    columns=[
        "cloudiness",
        "precipitation",
        "precipitation_deposits",
        "wetness",
        "wind_intensity",
        "sun_altitude_angle",
        "sun_azimuth_angle",
        "fog_density",
        "fog_distance",
        "fog_falloff",
        "scattering_intensity",
        "mie_scattering_scale",
        "rayleigh_scattering_scale",
    ]
)

# Percentage values
percent_cols = [
    "cloudiness",
    "precipitation",
    "precipitation_deposits",
    "wetness",
    "wind_intensity",
    "fog_density",
]

for col in percent_cols:
    samples_df[col] = samples_df[col].clip(0, 100)

# Sun altitude
samples_df["sun_altitude_angle"] = (
    samples_df["sun_altitude_angle"]
    .clip(-90, 90)
)

# Sun azimuth
samples_df["sun_azimuth_angle"] = (
    samples_df["sun_azimuth_angle"] % 360
)

# Fog distance
samples_df["fog_distance"] = (
    samples_df["fog_distance"]
    .clip(lower=0)
)

# Small scattering parameters
samples_df["fog_falloff"] = (
    samples_df["fog_falloff"]
    .clip(0, 5)
)

samples_df["scattering_intensity"] = (
    samples_df["scattering_intensity"]
    .clip(0, 5)
)

samples_df["mie_scattering_scale"] = (
    samples_df["mie_scattering_scale"]
    .clip(0, 5)
)

samples_df["rayleigh_scattering_scale"] = (
    samples_df["rayleigh_scattering_scale"]
    .clip(0, 5)
)

print(samples_df.head())
samples_df["cloudiness"] = samples_df["cloudiness"].clip(0, 100)
samples_df["precipitation"] = samples_df["precipitation"].clip(0, 100)
samples_df["precipitation_deposits"] = samples_df["precipitation_deposits"].clip(0, 100)
samples_df["wetness"] = samples_df["wetness"].clip(0, 100)
samples_df["wind_intensity"] = samples_df["wind_intensity"].clip(0, 100)

samples_df["sun_altitude_angle"] = samples_df["sun_altitude_angle"].clip(-90, 90)
samples_df["sun_azimuth_angle"] = samples_df["sun_azimuth_angle"] % 360

samples_df["fog_density"] = samples_df["fog_density"].clip(0, 100)

# Avoid zero visibility
samples_df["fog_distance"] = samples_df["fog_distance"].clip(0.1, 100)

samples_df["fog_falloff"] = samples_df["fog_falloff"].clip(0, 5)
samples_df["scattering_intensity"] = samples_df["scattering_intensity"].clip(0, 5)
samples_df["mie_scattering_scale"] = samples_df["mie_scattering_scale"].clip(0, 5)
samples_df["rayleigh_scattering_scale"] = samples_df["rayleigh_scattering_scale"].clip(0, 5)
# ============================================================
# Score generated weather
# ============================================================

print("\nScoring generated weather...")

X_score = scaler.transform(samples_df.to_numpy(dtype=np.float32))

predicted_risk = risk_model.predict(X_score)

samples_df["predicted_risk"] = predicted_risk

samples_df = samples_df.sort_values(
    "predicted_risk",
    ascending=False
).reset_index(drop=True)

print(samples_df[["predicted_risk"]].head())
# ============================================================
# Save all generated weather
# ============================================================

weather_cols = [
    "cloudiness",
    "precipitation",
    "precipitation_deposits",
    "wetness",
    "wind_intensity",
    "sun_altitude_angle",
    "sun_azimuth_angle",
    "fog_density",
    "fog_distance",
    "fog_falloff",
    "scattering_intensity",
    "mie_scattering_scale",
    "rayleigh_scattering_scale",
]

weather_all = samples_df[weather_cols].to_dict(
    orient="records"
)

with open(OUTPUT_BATCH, "w") as f:

    yaml.safe_dump(
        {"weather_configs": weather_all},
        f,
        sort_keys=False
    )

print("Saved weather_batch.yaml")
# ============================================================
# Save Top-K risky weather
# ============================================================

TOP_K = 50

weather_top = (
    samples_df
    .head(TOP_K)[weather_cols]
    .to_dict(orient="records")
)

with open(OUTPUT_RISKY, "w") as f:

    yaml.safe_dump(
        {"weather_configs": weather_top},
        f,
        sort_keys=False
    )

print("Saved risky_weather.yaml")
print("\nRADM pipeline completed successfully.")
