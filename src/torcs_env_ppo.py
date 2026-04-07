import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from torcs_client import Client


class TorcsEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, port=3001, max_steps=4000):
        super().__init__()
        self.port = port
        self.max_steps = max_steps
        self.client = None
        self.steps = 0
        self.stuck_steps = 0
        self.needs_restart = False
        self.last_steer = 0.0
        self.lap_done = False
        self.lap_armed = False

        # angle + trackPos + speedX + 19 track sensors
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(22,),
            dtype=np.float32,
        )

        # steering only
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def _obs_to_vec(self, obs):
        return np.concatenate([
            np.array([obs["angle"]], dtype=np.float32),
            np.array([obs["trackPos"]], dtype=np.float32),
            np.array([obs["speedX"]], dtype=np.float32),
            np.array(obs["track"], dtype=np.float32),
        ])

    def _reward(self, obs, prev_obs=None, steer=0.0, step_count=0):
        """
        Steering-focused reward:
        - reward forward motion
        - penalize sideways motion
        - penalize off-center driving
        - penalize jerky steering
        - penalize collisions
        - mild no-progress penalty after the start phase
        """
        angle = float(obs["angle"])
        speed_x = float(obs["speedX"])
        track_pos = float(obs["trackPos"])
        damage = float(obs.get("damage", 0.0))

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        forward = speed_x * cos_a
        side = abs(speed_x * sin_a)

        center_error = abs(track_pos)
        center_deadband = 0.15
        center_pen_error = max(0.0, center_error - center_deadband)
        reward = 0.0

        # Main driving reward
        reward += (1.8 * forward - 0.8 * side) / 100.0
        # Quadratic center penalty outside a gentle deadband.
        reward -= 0.18 * (center_pen_error ** 2)

        # Small step cost to discourage standing still
        reward -= 0.01

        # Steering smoothness penalty
        steer_change = abs(float(steer) - self.last_steer)
        reward -= 0.28 * steer_change
        # Mild penalties to reduce persistent wiggle at higher speeds.
        reward -= 0.02 * abs(float(steer))
        reward -= 0.05 * abs(angle)

        # Collision penalty
        if prev_obs is not None:
            prev_damage = float(prev_obs.get("damage", 0.0))
            damage_increase = max(0.0, damage - prev_damage)
            if damage_increase > 0:
                reward -= 1.0

        # No-progress penalty after the start phase
        if prev_obs is not None and step_count > 40:
            prev_dist = float(prev_obs.get("distRaced", 0.0))
            curr_dist = float(obs.get("distRaced", 0.0))
            progress = curr_dist - prev_dist
            if progress < 0.02:
                reward -= 1.0
            if abs(speed_x) < 5.0:
                reward -= 0.5

        # Strong off-track penalty
        if center_error > 1.0:
            reward -= 4.0

        # Keep reward magnitude in a stable range for PPO critic/policy updates.
        reward = float(np.clip(reward, -5.0, 5.0))
        return reward

    def _send_drive_command(self, steer, obs):
        """
        Steering-only setup with a simple adaptive speed controller.
        """
        speed = float(obs["speedX"])
        angle = abs(float(obs.get("angle", 0.0)))
        track_pos = abs(float(obs.get("trackPos", 0.0)))

        accel = 0.0
        brake = 0.0
        gear = 1

        # Higher speed on straights, slower when off-angle or off-center.
        target_speed = 60.0
        target_speed -= 20.0 * min(angle, 0.8)
        target_speed -= 12.0 * min(track_pos, 1.0)
        target_speed = float(np.clip(target_speed, 30.0, 60.0))

        # Very simple adaptive target-speed control
        if speed < target_speed:
            accel = 0.28
            brake = 0.0
        else:
            accel = 0.0
            brake = 0.04

        self.client.R.d["steer"] = float(np.clip(steer, -1.0, 1.0))
        self.client.R.d["accel"] = accel
        self.client.R.d["brake"] = brake
        self.client.R.d["gear"] = gear
        self.client.R.d["clutch"] = 0.0
        self.client.R.d["meta"] = 0
        self.client.respond_to_server()

    def _lap_completed(self, prev_obs, next_obs):
        """
        One-lap-per-episode trigger:
        mark complete once when TORCS reports a positive lastLapTime.
        """
        last_lap = float(next_obs.get("lastLapTime", 0.0))

        # Arm only after we observe a clean "no completed lap yet" state.
        # This avoids stale carry-over values right after reset.
        if last_lap <= 0.0:
            self.lap_armed = True

        if self.lap_done:
            return False
        return self.lap_armed and last_lap > 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.stuck_steps = 0
        self.last_steer = 0.0
        self.lap_done = False
        self.lap_armed = False

        if self.client is None:
            self.client = Client(p=self.port, vision=False)
            self.client.get_servers_input()
            obs = self.client.S.d
            return self._obs_to_vec(obs), {}

        if self.needs_restart:
            self.client.R.d["steer"] = 0.0
            self.client.R.d["accel"] = 0.0
            self.client.R.d["brake"] = 0.0
            self.client.R.d["gear"] = 1
            self.client.R.d["clutch"] = 0.0
            self.client.R.d["meta"] = 1
            self.client.respond_to_server()

            time.sleep(0.5)

            self.client = None
            time.sleep(0.5)
            self.client = Client(p=self.port, vision=False)

            self.client.get_servers_input()
            obs = self.client.S.d

            self.needs_restart = False
            return self._obs_to_vec(obs), {}

        self.client.get_servers_input()
        obs = self.client.S.d
        return self._obs_to_vec(obs), {}

    def step(self, action):
        self.steps += 1

        self.client.get_servers_input()
        obs = self.client.S.d

        steer = 0.85 * self.last_steer + 0.15 * float(action[0])
        self._send_drive_command(steer, obs)

        self.client.get_servers_input()
        next_obs = self.client.S.d

        progress = float(next_obs.get("distRaced", 0.0)) - float(obs.get("distRaced", 0.0))

        # Ignore stuck detection during start phase
        if self.steps > 40:
            if progress < 0.01 and abs(next_obs["speedX"]) < 3.0:
                self.stuck_steps += 1
            else:
                self.stuck_steps = 0
        else:
            self.stuck_steps = 0

        stuck = self.stuck_steps >= 25
        lap_completed = self._lap_completed(obs, next_obs)
        if lap_completed:
            self.lap_done = True
            # Pause briefly before handing control back for reset/restart.
            time.sleep(0.5)

        reward = self._reward(next_obs, obs, steer=steer, step_count=self.steps)
        if stuck:
            reward -= 5.0
        if lap_completed:
            reward += 5.0

        terminated = (
            abs(next_obs["trackPos"]) > 1.5
            or stuck
            or lap_completed
        )

        truncated = self.steps >= self.max_steps

        if terminated or truncated:
            self.needs_restart = True

        self.last_steer = steer

        info = {
            "speedX": float(next_obs["speedX"]),
            "trackPos": float(next_obs["trackPos"]),
            "angle": float(next_obs["angle"]),
            "rpm": float(next_obs["rpm"]),
            "distRaced": float(next_obs.get("distRaced", 0.0)),
            "lastLapTime": float(next_obs.get("lastLapTime", 0.0)),
            "curLapTime": float(next_obs.get("curLapTime", 0.0)),
            "stuck": stuck,
            "lap_completed": lap_completed,
            "reward_forward": float(next_obs["speedX"] * np.cos(next_obs["angle"])),
        }

        return self._obs_to_vec(next_obs), reward, terminated, truncated, info

    def close(self):
        pass
