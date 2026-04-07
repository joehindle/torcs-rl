import argparse
import csv
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from torcs_env_sac import TorcsEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="models/sac_torcs_final.zip",
        help="Path to SAC model zip.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of eval episodes.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/sac_eval",
        help="Directory to save metrics and plots.",
    )
    parser.add_argument(
        "--max-complete-steps",
        type=int,
        default=3500,
        help="Heuristic step limit used in completion metric.",
    )
    parser.add_argument(
        "--completion-dist",
        type=float,
        default=None,
        help="If set, use this distRaced threshold for completion instead of heuristics.",
    )
    return parser.parse_args()


def make_env():
    return TorcsEnv(port=3001, max_steps=4000)


def _to_float(x, default=np.nan):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _safe_stats(values):
    arr = np.array(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "p10": np.nan,
            "p90": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _write_episode_csv(path, rows):
    fieldnames = [
        "episode",
        "steps",
        "total_reward",
        "distRaced",
        "dist_per_step",
        "final_speedX",
        "mean_speedX",
        "p90_speedX",
        "max_speedX",
        "final_trackPos",
        "mean_abs_trackPos",
        "max_abs_trackPos",
        "offtrack_ratio",
        "final_angle",
        "mean_abs_angle",
        "max_abs_angle",
        "final_rpm",
        "mean_rpm",
        "max_rpm",
        "final_damage",
        "damage_delta",
        "completed",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_timeseries_csv(path, rows):
    fieldnames = [
        "episode",
        "step",
        "speedX",
        "trackPos",
        "angle",
        "rpm",
        "damage",
        "distRaced",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_plots(output_dir, episode_rows, timeseries_rows):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Plotting skipped (matplotlib unavailable): {exc}")
        return

    rewards = np.array([r["total_reward"] for r in episode_rows], dtype=np.float32)
    dists = np.array([r["distRaced"] for r in episode_rows], dtype=np.float32)
    speeds = np.array([r["mean_speedX"] for r in episode_rows], dtype=np.float32)
    completed = np.array([1 if r["completed"] else 0 for r in episode_rows], dtype=np.int32)
    ep_ids = np.array([r["episode"] for r in episode_rows], dtype=np.int32)

    # Reward / distance histograms.
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    axs[0].hist(rewards[np.isfinite(rewards)], bins=12, color="#2D7FFF", alpha=0.85)
    axs[0].set_title("Reward Distribution")
    axs[0].set_xlabel("Total Reward")
    axs[0].set_ylabel("Count")
    axs[1].hist(dists[np.isfinite(dists)], bins=12, color="#2CA58D", alpha=0.85)
    axs[1].set_title("Distance Distribution")
    axs[1].set_xlabel("distRaced")
    axs[1].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_dir / "hist_reward_distance.png", dpi=140)
    plt.close(fig)

    # Episode bars.
    fig, ax = plt.subplots(figsize=(11, 4.8))
    colors = np.where(completed > 0, "#2CA58D", "#E36414")
    ax.bar(ep_ids, dists, color=colors, alpha=0.9)
    ax.set_title("Distance by Episode (Green=Completed)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("distRaced")
    fig.tight_layout()
    fig.savefig(output_dir / "episode_distance_bar.png", dpi=140)
    plt.close(fig)

    # Reward-distance correlation.
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    sc = ax.scatter(dists, rewards, c=speeds, cmap="viridis", alpha=0.85)
    ax.set_title("Reward vs Distance (color=mean speedX)")
    ax.set_xlabel("distRaced")
    ax.set_ylabel("total_reward")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("mean_speedX")
    fig.tight_layout()
    fig.savefig(output_dir / "scatter_reward_vs_distance.png", dpi=140)
    plt.close(fig)

    # Mean speed profile over steps.
    max_step = int(max((r["step"] for r in timeseries_rows), default=0))
    if max_step > 0:
        speed_by_step = [[] for _ in range(max_step + 1)]
        for row in timeseries_rows:
            step = int(row["step"])
            speed = _to_float(row["speedX"])
            if np.isfinite(speed):
                speed_by_step[step].append(speed)
        mean_speed = np.array(
            [np.mean(v) if len(v) > 0 else np.nan for v in speed_by_step],
            dtype=np.float32,
        )
        std_speed = np.array(
            [np.std(v) if len(v) > 0 else np.nan for v in speed_by_step],
            dtype=np.float32,
        )
        x = np.arange(mean_speed.shape[0], dtype=np.int32)
        valid = np.isfinite(mean_speed)
        if np.any(valid):
            fig, ax = plt.subplots(figsize=(10.5, 4.8))
            ax.plot(x[valid], mean_speed[valid], color="#2D7FFF", linewidth=1.8, label="mean speedX")
            lo = mean_speed - std_speed
            hi = mean_speed + std_speed
            std_ok = valid & np.isfinite(lo) & np.isfinite(hi)
            ax.fill_between(
                x[std_ok],
                lo[std_ok],
                hi[std_ok],
                color="#2D7FFF",
                alpha=0.18,
                label="±1 std",
            )
            ax.set_title("Mean Speed Profile by Step")
            ax.set_xlabel("Step")
            ax.set_ylabel("speedX")
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(output_dir / "mean_speed_profile.png", dpi=140)
            plt.close(fig)


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = DummyVecEnv([make_env])
    model = SAC.load(args.model, env=env)

    episode_rows = []
    timeseries_rows = []

    for episode in range(1, args.episodes + 1):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        last_info = None
        first_damage = None

        ep_speed = []
        ep_track = []
        ep_angle = []
        ep_rpm = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            total_reward += float(rewards[0])
            done = bool(dones[0])
            info = infos[0]
            last_info = info

            speed = _to_float(info.get("speedX", np.nan))
            track_pos = _to_float(info.get("trackPos", np.nan))
            angle = _to_float(info.get("angle", np.nan))
            rpm = _to_float(info.get("rpm", np.nan))
            damage = _to_float(info.get("damage", np.nan))
            dist = _to_float(info.get("distRaced", np.nan))

            if first_damage is None and np.isfinite(damage):
                first_damage = damage

            if np.isfinite(speed):
                ep_speed.append(speed)
            if np.isfinite(track_pos):
                ep_track.append(track_pos)
            if np.isfinite(angle):
                ep_angle.append(angle)
            if np.isfinite(rpm):
                ep_rpm.append(rpm)

            timeseries_rows.append(
                {
                    "episode": episode,
                    "step": steps,
                    "speedX": speed,
                    "trackPos": track_pos,
                    "angle": angle,
                    "rpm": rpm,
                    "damage": damage,
                    "distRaced": dist,
                }
            )
            steps += 1

        dist_raced = _to_float(last_info.get("distRaced", np.nan)) if last_info else np.nan
        final_speed = _to_float(last_info.get("speedX", np.nan)) if last_info else np.nan
        final_track = _to_float(last_info.get("trackPos", np.nan)) if last_info else np.nan
        final_angle = _to_float(last_info.get("angle", np.nan)) if last_info else np.nan
        final_rpm = _to_float(last_info.get("rpm", np.nan)) if last_info else np.nan
        final_damage = _to_float(last_info.get("damage", np.nan)) if last_info else np.nan
        if first_damage is None or not np.isfinite(first_damage) or not np.isfinite(final_damage):
            damage_delta = np.nan
        else:
            damage_delta = final_damage - first_damage

        mean_abs_track = float(np.mean(np.abs(ep_track))) if ep_track else np.nan
        max_abs_track = float(np.max(np.abs(ep_track))) if ep_track else np.nan
        offtrack_ratio = float(np.mean(np.abs(np.array(ep_track)) > 1.0)) if ep_track else np.nan
        mean_abs_angle = float(np.mean(np.abs(ep_angle))) if ep_angle else np.nan
        max_abs_angle = float(np.max(np.abs(ep_angle))) if ep_angle else np.nan
        mean_speed = float(np.mean(ep_speed)) if ep_speed else np.nan
        p90_speed = float(np.percentile(ep_speed, 90)) if ep_speed else np.nan
        max_speed = float(np.max(ep_speed)) if ep_speed else np.nan
        mean_rpm = float(np.mean(ep_rpm)) if ep_rpm else np.nan
        max_rpm = float(np.max(ep_rpm)) if ep_rpm else np.nan
        dist_per_step = float(dist_raced / max(steps, 1)) if np.isfinite(dist_raced) else np.nan

        if args.completion_dist is not None:
            completed = bool(np.isfinite(dist_raced) and dist_raced >= args.completion_dist)
        else:
            completed = bool(
                np.isfinite(dist_raced)
                and dist_raced > 0
                and steps < args.max_complete_steps
                and (not np.isfinite(damage_delta) or damage_delta <= 0.0)
            )

        row = {
            "episode": episode,
            "steps": steps,
            "total_reward": float(total_reward),
            "distRaced": dist_raced,
            "dist_per_step": dist_per_step,
            "final_speedX": final_speed,
            "mean_speedX": mean_speed,
            "p90_speedX": p90_speed,
            "max_speedX": max_speed,
            "final_trackPos": final_track,
            "mean_abs_trackPos": mean_abs_track,
            "max_abs_trackPos": max_abs_track,
            "offtrack_ratio": offtrack_ratio,
            "final_angle": final_angle,
            "mean_abs_angle": mean_abs_angle,
            "max_abs_angle": max_abs_angle,
            "final_rpm": final_rpm,
            "mean_rpm": mean_rpm,
            "max_rpm": max_rpm,
            "final_damage": final_damage,
            "damage_delta": damage_delta,
            "completed": completed,
        }
        episode_rows.append(row)

        print(
            f"Episode {episode}: "
            f"steps={steps}, "
            f"reward={total_reward:.3f}, "
            f"distRaced={dist_raced:.3f}, "
            f"mean_speed={mean_speed:.2f}, "
            f"offtrack_ratio={offtrack_ratio:.3f}, "
            f"completed={completed}"
        )

    # Persist outputs.
    episode_csv = output_dir / "episode_metrics.csv"
    timeseries_csv = output_dir / "timeseries.csv"
    _write_episode_csv(episode_csv, episode_rows)
    _write_timeseries_csv(timeseries_csv, timeseries_rows)

    # Summary metrics.
    completion_rate = float(np.mean([1.0 if r["completed"] else 0.0 for r in episode_rows])) if episode_rows else 0.0
    summary = {
        "episodes": len(episode_rows),
        "completion_rate": completion_rate,
        "reward": _safe_stats([r["total_reward"] for r in episode_rows]),
        "distRaced": _safe_stats([r["distRaced"] for r in episode_rows]),
        "dist_per_step": _safe_stats([r["dist_per_step"] for r in episode_rows]),
        "mean_speedX": _safe_stats([r["mean_speedX"] for r in episode_rows]),
        "p90_speedX": _safe_stats([r["p90_speedX"] for r in episode_rows]),
        "offtrack_ratio": _safe_stats([r["offtrack_ratio"] for r in episode_rows]),
        "mean_abs_trackPos": _safe_stats([r["mean_abs_trackPos"] for r in episode_rows]),
        "mean_abs_angle": _safe_stats([r["mean_abs_angle"] for r in episode_rows]),
        "damage_delta": _safe_stats([r["damage_delta"] for r in episode_rows]),
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (output_dir / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("SAC Evaluation Summary\n")
        f.write("======================\n")
        f.write(f"Episodes: {summary['episodes']}\n")
        f.write(f"Completion rate: {summary['completion_rate']:.2%}\n")
        f.write(f"Reward mean: {summary['reward']['mean']:.3f}\n")
        f.write(f"distRaced mean: {summary['distRaced']['mean']:.3f}\n")
        f.write(f"dist/step mean: {summary['dist_per_step']['mean']:.4f}\n")
        f.write(f"Mean speedX: {summary['mean_speedX']['mean']:.3f}\n")
        f.write(f"P90 speedX: {summary['p90_speedX']['mean']:.3f}\n")
        f.write(f"Offtrack ratio mean: {summary['offtrack_ratio']['mean']:.4f}\n")
        f.write(f"Mean |trackPos|: {summary['mean_abs_trackPos']['mean']:.4f}\n")
        f.write(f"Mean |angle|: {summary['mean_abs_angle']['mean']:.4f}\n")
        f.write(f"Damage delta mean: {summary['damage_delta']['mean']:.3f}\n")

    _save_plots(output_dir, episode_rows, timeseries_rows)

    print("\nSummary")
    print(f"Episodes: {summary['episodes']}")
    print(f"Completion rate: {summary['completion_rate']:.2%}")
    print(f"Mean distRaced: {summary['distRaced']['mean']:.3f}")
    print(f"Mean reward: {summary['reward']['mean']:.3f}")
    print(f"Saved outputs to: {output_dir}")
    print(f"- {episode_csv}")
    print(f"- {timeseries_csv}")
    print(f"- {output_dir / 'summary.json'}")
    print(f"- {output_dir / 'summary.txt'}")

    env.close()


if __name__ == "__main__":
    main()
