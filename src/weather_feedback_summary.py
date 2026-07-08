import os
import pandas as pd

feedback_root = "log_sac"
records = []

for root, dirs, files in os.walk(feedback_root):
    if "feedback.csv" not in files:
        continue

    feedback_path = os.path.join(root, "feedback.csv")
    try:
        df = pd.read_csv(feedback_path)
    except Exception:
        continue

    # Extract weather index (parse numeric id from directory name)
    weather_id = None
    for part in root.split("/"):
        if "weather" in part:
            digits = "".join([c for c in part if c.isdigit()])
            if digits:
                weather_id = int(digits)
                break

    if weather_id is None:
        continue

    collision_rate = df["collision_rate"].mean() if "collision_rate" in df.columns else None
    records.append({"weather_id": weather_id, "collision_rate": collision_rate})

df = pd.DataFrame(records).sort_values("weather_id")
df.to_csv("weather_feedback_summary_radm.csv", index=False)
print("✅ Summary finished. Saved as weather_feedback_summary_radm.csv")