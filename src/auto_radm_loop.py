import os
import subprocess
import sys
import shutil

def run_script(script_name):
    print(f"🚀 Running script: {script_name}")
    script_path = os.path.join(os.path.dirname(__file__), f"{script_name}.py")
    subprocess.run([sys.executable, script_path], check=True, env=env)

safebench_run = "/home/wzj/SafeBench/scripts/run.py"

def run_safebench(round_idx):
    output_dir = f"log_sac/round{round_idx+1}"
    print(f"🚗 [4] Launching SafeBench evaluation risky_weather.yaml → {output_dir}")
    subprocess.run([
        "python", safebench_run,
        "--mode", "eval",
        "--agent_cfg", "sac.yaml",
        "--scenario_cfg", "standard.yaml",
        "--weather_file", "paper/data/weather_batch.yaml",
        "--output_dir", output_dir
    ], check=True, env=env)

# Set default environment variables
env = os.environ.copy()
env["WEATHER_FILE"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "weather_batch.yaml"))

feedback_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "feedback_rounds_agent"))
os.makedirs(feedback_dir, exist_ok=True)

weather_save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "generated_weather_rounds"))
os.makedirs(weather_save_dir, exist_ok=True)

NUM_ROUNDS = 3

for round_idx in range(NUM_ROUNDS):
    print(f"\n========== Round {round_idx+1}/{NUM_ROUNDS} ==========")

    weather_file_name = f"weather_batch_round{round_idx+1}.yaml"
    weather_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", weather_file_name))

    # Set the weather input file for this round
    env["WEATHER_FILE"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", weather_file_name))

    # ✅ From the second round onward, load previous feedback and train
    if round_idx > 0:
        pass
        # run_script("radm_training_data")
        # run_script("RADM")

    # # ✅ Train RAD-M and generate risky_weather.yaml
    # run_script("radm_training_data")
    # run_script("RADM")

    # ✅ Copy the generated weather config for this round
    # current_weather = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "risky_weather.yaml"))
    # shutil.copy(current_weather, weather_file_path)

    # ✅ Launch SafeBench evaluation for this round
    run_safebench(round_idx)

    # ✅ Copy feedback results to the designated feedback_rounds folder
    # src_feedback = os.path.abspath(os.path.join("log_sac", f"round{round_idx+1}", "feedback.csv"))
    # dst_feedback = os.path.join(feedback_dir, f"feedback_round{round_idx+1}.csv")
    # if os.path.exists(src_feedback):
    #     shutil.copy(src_feedback, dst_feedback)
    #     print(f"📁 Feedback file copied to: {dst_feedback}")
    # else:
    #     print(f"⚠️ Feedback file not found: {src_feedback}")
    #
    # # ✅ Update environment variables for the next round
    # env["FEEDBACK_FILE"] = dst_feedback

print("✅ RADM diffusion generation pipeline completed!")