import argparse
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from torcs_env_ppo import TorcsEnv


def make_env():
    return TorcsEnv(port=3001, max_steps=4000)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint/model zip to resume from.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1000000,
        help="Timesteps to run for this training call.",
    )
    parser.add_argument(
        "--reset-num-timesteps",
        action="store_true",
        help="Reset SB3 timestep counter/logging (default: continue counter).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    # torcs_client parses sys.argv internally and exits on unknown flags.
    # Strip trainer CLI args so TORCS client init does not fail.
    sys.argv = [sys.argv[0]]
    env = DummyVecEnv([make_env])

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path="models/checkpoints",
        name_prefix="ppo_torcs",
    )

    if args.resume:
        print(f"Resuming from: {args.resume}")
        model = PPO.load(args.resume, env=env)
    else:
        print("Starting fresh model.")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            n_steps=8192,
            batch_size=512,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.001,
            clip_range=0.2,
            tensorboard_log="results/tb",
            policy_kwargs=dict(net_arch=[256, 256]),
        )

    model.learn(
        total_timesteps=args.timesteps,
        reset_num_timesteps=args.reset_num_timesteps,
        progress_bar=True,
        callback=checkpoint_callback,
    )

    model.save("models/ppo_torcs_first")
    env.close()


if __name__ == "__main__":
    main()
