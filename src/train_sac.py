import argparse
import sys
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from torcs_env_sac import TorcsEnv


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
        default=1_000_000,
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
        save_freq=25000,
        save_path="models/checkpoints",
        name_prefix="sac_torcs",
        save_replay_buffer=True,
    )

    if args.resume:
        print(f"Resuming from: {args.resume}")
        try:
            model = SAC.load(args.resume, env=env)
        except Exception as exc:
            raise RuntimeError(
                "Failed to resume checkpoint. This usually means env spaces changed "
                "(for example action size/observation size). Start a fresh run or "
                "use a checkpoint trained with the current env definition."
            ) from exc
        print(f"Checkpoint loaded: {args.resume}")
        # Try to restore replay buffer saved alongside checkpoint zip.
        resume_path = Path(args.resume)
        rb_candidates = [
            # Pattern used by final manual save:
            resume_path.with_name(f"{resume_path.stem}_replay_buffer.pkl"),
            # Pattern used by SB3 CheckpointCallback when save_replay_buffer=True:
            resume_path.with_name(
                f"{resume_path.stem.replace('sac_torcs_', 'sac_torcs_replay_buffer_')}.pkl"
            ),
        ]
        rb_path = next((p for p in rb_candidates if p.exists()), None)
        if rb_path is not None:
            model.load_replay_buffer(str(rb_path))
            print(f"Replay buffer loaded: {rb_path}")
        else:
            print("Replay buffer not found; resuming weights only.")
    else:
        print("Starting fresh SAC model.")
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=200_000,
            learning_starts=10_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef=0.02,
            tensorboard_log="results/tb",
            policy_kwargs=dict(net_arch=[256, 256]),
        )

    model.learn(
        total_timesteps=args.timesteps,
        reset_num_timesteps=args.reset_num_timesteps,
        progress_bar=True,
        callback=checkpoint_callback,
    )

    model.save("models/sac_torcs_final")
    model.save_replay_buffer("models/sac_torcs_final_replay_buffer.pkl")
    env.close()


if __name__ == "__main__":
    main()
