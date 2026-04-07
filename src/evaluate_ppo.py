import csv
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from torcs_env_ppo import TorcsEnv


MODEL_PATH = "models/checkpoints/ppo_torcs_500000_steps.zip"
OUTPUT_CSV = "results/ppo_eval.csv"
EPISODES = 10


def make_env():
    return TorcsEnv(port=3001, max_steps=4000)


def main():
    env = DummyVecEnv([make_env])
    model = PPO.load(MODEL_PATH, env=env)

    rows = []

    for episode in range(1, EPISODES + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        last_info = None

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            total_reward += float(rewards[0])
            done = bool(dones[0])
            last_info = infos[0]
            steps += 1

        dist_raced = float(last_info.get("distRaced", np.nan)) if last_info else np.nan
        speed_x = float(last_info.get("speedX", np.nan)) if last_info else np.nan
        track_pos = float(last_info.get("trackPos", np.nan)) if last_info else np.nan
        angle = float(last_info.get("angle", np.nan)) if last_info else np.nan
        stuck = bool(last_info.get("stuck", False)) if last_info else False

        completed = dist_raced > 0 and not stuck and steps < 2500

        row = {
            "episode": episode,
            "steps": steps,
            "total_reward": total_reward,
            "distRaced": dist_raced,
            "final_speedX": speed_x,
            "final_trackPos": track_pos,
            "final_angle": angle,
            "stuck": stuck,
            "completed": completed,
        }
        rows.append(row)

        print(
            f"Episode {episode}: "
            f"steps={steps}, "
            f"reward={total_reward:.3f}, "
            f"distRaced={dist_raced:.3f}, "
            f"stuck={stuck}, "
            f"completed={completed}"
        )

    output_path = Path(OUTPUT_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "steps",
                "total_reward",
                "distRaced",
                "final_speedX",
                "final_trackPos",
                "final_angle",
                "stuck",
                "completed",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    completed_runs = [r for r in rows if r["completed"]]
    mean_dist = np.mean([r["distRaced"] for r in rows]) if rows else np.nan
    completion_rate = len(completed_runs) / len(rows) if rows else 0.0

    print("\nSummary")
    print(f"Episodes: {len(rows)}")
    print(f"Completion rate: {completion_rate:.2%}")
    print(f"Mean distRaced: {mean_dist:.3f}")
    print(f"Saved results to {output_path}")

    env.close()


if __name__ == "__main__":
    main()
