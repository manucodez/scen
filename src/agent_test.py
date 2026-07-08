import subprocess
import os
import time
import json

def run_multi_agent_test(weather_file, weather_tag, safebench_run_path, env, agent_list=None):
    """
    Multi-policy automated evaluation module. By default it tests TD3, PPO, and SAC, and also supports additional agents.
    Each run outputs to log_transfer/{weather_tag}_{agent}
    """
    if agent_list is None:
        agent_list = ["td3", "ppo", "sac"]

    os.makedirs("log_transfer", exist_ok=True)
    summary = {}

    for agent in agent_list:
        output_dir = f"log_transfer/{weather_tag}_{agent}"
        log_file = f"{output_dir}/stdout.log"
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n🚗 Starting evaluation: {weather_tag} + {agent} → {output_dir}")
        start_time = time.time()

        try:
            with open(log_file, "w") as outlog:
                subprocess.run([
                    "python", safebench_run_path,
                    "--mode", "eval",
                    "--agent_cfg", f"{agent}.yaml",
                    "--scenario_cfg", "standard.yaml",
                    "--weather_file", weather_file,
                    "--output_dir", output_dir
                ], check=True, env=env, stdout=outlog, stderr=outlog)

            duration = round(time.time() - start_time, 2)
            summary[agent] = {"status": "success", "time": duration}

        except subprocess.CalledProcessError:
            summary[agent] = {"status": "fail", "time": None}
            print(f"❌ {agent} failed on {weather_tag}; skipping.")

    # Save summary information
    summary_path = f"log_transfer/{weather_tag}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ All agent evaluations for {weather_tag} finished. Results saved to {summary_path}")

# Example usage: you can call this from your main script as follows:
if __name__ == "__main__":
    safebench_run = "/home/youruser/SafeBench/scripts/run.py"  # replace with your local path
    env = os.environ.copy()

    run_multi_agent_test("data/weather_sampled_50.yaml", "baseline", safebench_run, env)
    run_multi_agent_test("data/ddpm_risky_weather.yaml", "ddpm", safebench_run, env)
    run_multi_agent_test("data/risky_weather.yaml", "radm", safebench_run, env)