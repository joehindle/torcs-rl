import copy
import math
import collections as col
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from torcs_client import Client


class TorcsEnv(gym.Env):
    """
    Analyse-style TORCS environment adapted to this project.
    """
#edit
    metadata = {"render_modes": []}

    def __init__(self, gui=False, infinite=False, port=3001, max_steps=4000):
        super().__init__()
        self.initial = True
        self.gui = gui
        self.port = port
        self.max_steps = max_steps
        self.needs_restart = False
        self.last_steer_cmd = 0.0
        self.last_pedal_cmd = 0.0
        self.last_accel_cmd = 0.0
        self.last_brake_cmd = 0.0
        self.current_gear = 1
        self.shift_hold_steps = 18
        self.shift_hold_timer = 0
        self.post_start_steps = 0
        self.spin_steps = 0

        # Keep early-stop checks lenient to avoid reset loops at race start.
        self.terminal_judge_start = 600
        self.termination_limit_progress = 0.0
        self.spawn_grace_steps = 50
        self.warmup_steps = 60
        self.spin_angle_threshold = 0.55  # normalized angle (angle/pi)
        self.spin_slip_threshold = 0.10   # normalized speedY (speedY/300)
        self.spin_terminate_steps = 8

        if infinite:
            self.terminal_judge_start = 1_000_000_000

        # Analyse action space: steer, accel, brake
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: base 23 + control context (accel, brake, steer, pedal) = 27
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(27,),
            dtype=np.float32,
        )

    def make_observaton(self, raw_obs):
        names = [
            "focus",
            "speedX",
            "speedY",
            "speedZ",
            "angle",
            "damage",
            "opponents",
            "rpm",
            "track",
            "trackPos",
            "wheelSpinVel",
        ]
        Observation = col.namedtuple("Observaion", names)
        return Observation(
            focus=np.array(raw_obs["focus"], dtype=np.float32) / 200.0,
            speedX=np.array(raw_obs["speedX"], dtype=np.float32) / 300.0,
            speedY=np.array(raw_obs["speedY"], dtype=np.float32) / 300.0,
            speedZ=np.array(raw_obs["speedZ"], dtype=np.float32) / 300.0,
            angle=np.array(raw_obs["angle"], dtype=np.float32) / math.pi,
            damage=np.array(raw_obs["damage"], dtype=np.float32),
            opponents=np.array(raw_obs["opponents"], dtype=np.float32) / 200.0,
            rpm=np.array(raw_obs["rpm"], dtype=np.float32) / 10000.0,
            track=np.array(raw_obs["track"], dtype=np.float32) / 200.0,
            trackPos=np.array(raw_obs["trackPos"], dtype=np.float32),
            wheelSpinVel=np.array(raw_obs["wheelSpinVel"], dtype=np.float32),
        )

    def _state_from_ob(self, ob, accel_cmd=0.0, brake_cmd=0.0, steer_cmd=0.0, pedal_cmd=0.0):
        return np.hstack(
            (ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, accel_cmd, brake_cmd, steer_cmd, pedal_cmd)
        ).astype(np.float32)

    def _reward(self, ob, steer, pedal, prev_steer, prev_pedal):
        """
        Normalized SAC-friendly dense reward:
        - reward forward aligned speed
        - penalize lateral speed, heading error, center error
        - penalize jerky control changes
        Final reward is clipped to keep updates stable.
        """
        speed_x = float(np.clip(ob.speedX, -1.0, 1.5))  # normalized by /300
        speed_y = float(np.clip(ob.speedY, -1.0, 1.0))  # normalized by /300
        angle_n = float(np.clip(ob.angle, -1.0, 1.0))   # angle / pi
        track_pos = float(np.clip(ob.trackPos, -2.0, 2.0))

        # Forward component in normalized units.
        forward = speed_x * float(np.cos(angle_n * np.pi))

        # Soft center deadband to avoid over-correcting tiny offsets.
        center_err = abs(track_pos)
        center_pen = max(0.0, center_err - 0.10)

        steer_change = abs(float(steer) - float(prev_steer))
        pedal_change = abs(float(pedal) - float(prev_pedal))

        reward = 0.0
        reward += 0.2 * forward
        reward -= 0.80 * abs(speed_y)
        reward -= 0.80 * abs(angle_n)
        reward -= 0.75 * (center_pen ** 2)
        reward -= 0.06 * abs(float(steer))
        reward -= 0.16 * steer_change
        reward -= 0.06 * pedal_change
        # Extra penalty for large steering at higher speed (helps reduce spins).
        reward -= 0.10 * max(0.0, speed_x) * abs(float(steer))
        reward -= 0.14 * max(0.0, speed_x) * steer_change

        # Small bonus for stable, centered alignment.
        if center_err < 0.10 and abs(angle_n) < 0.08:
            reward += 0.04
        reward -= 0.01  # mild step cost

        # Keep reward magnitude stable for critic/policy.
        return float(np.clip(reward, -1.0, 1.0))

    def step(self, action):
        # Action mapping with basic stabilization.
        prev_steer_cmd = self.last_steer_cmd
        prev_pedal_cmd = self.last_pedal_cmd
        prev_accel_cmd = self.last_accel_cmd
        prev_brake_cmd = self.last_brake_cmd
        prev_gear_cmd = int(self.current_gear)

        raw_steer = float(np.clip(action[0], -1.0, 1.0))
        raw_accel = float(np.clip(action[1], 0.0, 1.0))
        raw_brake = float(np.clip(action[2], 0.0, 1.0))

        # Steering smoothing + rate limit.
        steer = 0.88 * self.last_steer_cmd + 0.12 * raw_steer
        steer_delta = float(np.clip(steer - self.last_steer_cmd, -0.03, 0.03))
        steer = self.last_steer_cmd + steer_delta
        if abs(steer) < 0.02:
            steer = 0.0

        # Detect race start; countdown time should not consume warmup.
        speed_x = float(self.client.S.d.get("speedX", 0.0))
        cur_lap_time = float(self.client.S.d.get("curLapTime", 0.0))
        dist_raced = float(self.client.S.d.get("distRaced", 0.0))
        angle_raw = float(self.client.S.d.get("angle", 0.0))
        track_pos_raw = float(self.client.S.d.get("trackPos", 0.0))
        race_started = (cur_lap_time > 0.2) or (speed_x > 1.0) or (dist_raced > 1.0)
        if race_started:
            self.post_start_steps += 1

        # Straight-exit stabilization: damp recenter oscillation on straights.
        straight_mode = abs(angle_raw) < 0.10
        if straight_mode and speed_x > 45.0:
            # Blend toward a gentle center-seeking steer on straights.
            recenter = float(np.clip(-0.30 * track_pos_raw, -0.30, 0.30))
            steer = 0.75 * steer + 0.25 * recenter

            # If we are near center and trying to flip steering side, damp hard.
            if abs(track_pos_raw) < 0.20 and (steer * self.last_steer_cmd) < 0.0:
                steer *= 0.45

            # Inside a narrow center band on straights, avoid micro-corrections.
            if abs(track_pos_raw) < 0.08:
                steer *= 0.35

        self.last_steer_cmd = float(np.clip(steer, -1.0, 1.0))
        steer = self.last_steer_cmd

        # Smooth accel/brake separately to avoid cancellation deadlock.
        accel = 0.75 * prev_accel_cmd + 0.25 * raw_accel
        brake = 0.75 * prev_brake_cmd + 0.25 * raw_brake
        if accel < 0.03:
            accel = 0.0
        if brake < 0.03:
            brake = 0.0
        # Suppress simultaneous heavy accel/brake overlap.
        brake *= (1.0 - accel)

        self.last_accel_cmd = accel
        self.last_brake_cmd = brake
        pedal_cmd = accel - brake
        self.last_pedal_cmd = pedal_cmd

        self.client.R.d["steer"] = steer
        self.client.R.d["accel"] = accel
        self.client.R.d["brake"] = brake

        # Auto transmission with hysteresis + dwell to prevent gear hunting.
        speed_x = float(self.client.S.d["speedX"])
        reported_gear = int(float(self.client.S.d.get("gear", self.current_gear)))
        if reported_gear >= 1:
            self.current_gear = reported_gear
        else:
            self.current_gear = max(1, self.current_gear)

        upshift_threshold = {1: 60.0, 2: 90.0, 3: 120.0, 4: 150.0, 5: 180.0}
        downshift_threshold = {2: 48.0, 3: 78.0, 4: 108.0, 5: 138.0, 6: 168.0}

        if self.shift_hold_timer > 0:
            self.shift_hold_timer -= 1
        else:
            if self.current_gear < 6 and speed_x > upshift_threshold[self.current_gear]:
                self.current_gear += 1
                self.shift_hold_timer = self.shift_hold_steps
            elif self.current_gear > 1 and speed_x < downshift_threshold[self.current_gear]:
                self.current_gear -= 1
                self.shift_hold_timer = self.shift_hold_steps

        self.client.R.d["gear"] = int(self.current_gear)

        self.client.R.d["clutch"] = 0.0
        self.client.R.d["meta"] = 0

        obs_pre = copy.deepcopy(self.client.S.d)

        self.client.respond_to_server()
        self.client.get_servers_input()

        obs = self.client.S.d
        ob = self.make_observaton(obs)
        state = self._state_from_ob(
            ob,
            accel_cmd=float(accel),
            brake_cmd=float(brake),
            steer_cmd=float(steer),
            pedal_cmd=float(pedal_cmd),
        )

        sp = float(obs["speedX"])
        reward = self._reward(ob, steer, pedal_cmd, prev_steer_cmd, prev_pedal_cmd)
        # Small dense progress bonus: reward step-to-step forward distance.
        prev_dist = float(obs_pre.get("distRaced", 0.0))
        curr_dist = float(obs.get("distRaced", 0.0))
        progress = float(np.clip(curr_dist - prev_dist, -1.0, 1.0))
        reward += 0.60 * progress
        # Small anti-hunt penalty for each shift event.
        if int(self.current_gear) != prev_gear_cmd:
            reward -= 0.01
        reward = float(np.clip(reward, -1.0, 1.0))

        terminated = False
        truncated = False

        # collision
        if float(obs["damage"]) - float(obs_pre["damage"]) > 0.0:
            reward = -1.0
            terminated = True

        # out of track (after initial spawn grace)
        if self.timestep > self.spawn_grace_steps and abs(float(obs["trackPos"])) > 1.0:
            reward = -1.0
            terminated = True

        # backward (after initial spawn grace)
        if self.timestep > self.spawn_grace_steps and np.cos(float(obs["angle"])) < 0.0:
            reward = -1.0
            terminated = True

        # sustained spin/slide detection (after initial spawn grace).
        angle_n = abs(float(np.clip(ob.angle, -1.0, 1.0)))
        slip_n = abs(float(np.clip(ob.speedY, -1.0, 1.0)))
        if self.timestep > self.spawn_grace_steps and (
            angle_n > self.spin_angle_threshold and slip_n > self.spin_slip_threshold
        ):
            self.spin_steps += 1
        else:
            self.spin_steps = 0
        if self.spin_steps >= self.spin_terminate_steps:
            reward = -1.0
            terminated = True

        # too little progress after startup
        if (
            self.timestep > self.terminal_judge_start
            and reward < self.termination_limit_progress
            and sp < 10.0
        ):
            terminated = True

        # time limit
        if self.timestep >= self.max_steps:
            truncated = True

        if terminated or truncated:
            self.needs_restart = True

        self.timestep += 1

        info = {
            "speedX": float(obs["speedX"]),
            "trackPos": float(obs["trackPos"]),
            "angle": float(obs["angle"]),
            "damage": float(obs.get("damage", 0.0)),
            "distRaced": float(obs.get("distRaced", 0.0)),
            "rpm": float(obs.get("rpm", 0.0)),
        }

        return state, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.initial:
            self.client = Client(p=self.port, vision=bool(self.gui))
            self.initial = False
        elif self.needs_restart:
            self.client.R.d["meta"] = 1
            self.client.respond_to_server()
            time.sleep(0.5)
            self.client = None
            time.sleep(0.5)
            self.client = Client(p=self.port, vision=bool(self.gui))
            self.needs_restart = False

        self.timestep = 0
        self.last_steer_cmd = 0.0
        self.last_pedal_cmd = 0.0
        self.last_accel_cmd = 0.0
        self.last_brake_cmd = 0.0
        self.current_gear = 1
        self.shift_hold_timer = 0
        self.post_start_steps = 0
        self.spin_steps = 0
        self.client.get_servers_input()
        reported_gear = int(float(self.client.S.d.get("gear", 1)))
        self.current_gear = max(1, reported_gear)
        ob = self.make_observaton(self.client.S.d)
        state = self._state_from_ob(
            ob,
            accel_cmd=float(self.last_accel_cmd),
            brake_cmd=float(self.last_brake_cmd),
            steer_cmd=float(self.last_steer_cmd),
            pedal_cmd=float(self.last_pedal_cmd),
        )

        return state, {}

    def close(self):
        if hasattr(self, "client") and self.client is not None:
            self.client.shutdown()
