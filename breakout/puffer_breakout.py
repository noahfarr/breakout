import os

import gymnasium
import numpy as np

from raylib import rl, colors

import pufferlib
from pufferlib.environments.ocean import render
from breakout.c_breakout import CBreakout


class PufferBreakout(pufferlib.PufferEnv):
    def __init__(
        self,
        fps: float = 60,
        frameskip: int = 4,
        width: int = 576,
        height: int = 330,
        num_brick_rows: int = 6,
        num_brick_cols: int = 18,
        report_interval: int = 128,
        num_agents: int = 4096,
        render_mode: str = "rgb_array",
    ) -> None:
        self.fps = fps
        self.dt = 1 / fps
        self.frameskip = frameskip
        self.width = width
        self.height = height
        self.grid = [np.zeros((height, width), dtype=np.uint8)]
        self.num_brick_rows = num_brick_rows
        self.num_brick_cols = num_brick_cols
        self.num_bricks = num_brick_rows * num_brick_cols
        self.report_interval = report_interval

        self.paddle_positions = np.zeros((num_agents, 2), dtype=np.float32)
        self.ball_positions = np.zeros((num_agents, 2), dtype=np.float32)
        self.ball_velocities = np.zeros((num_agents, 2), dtype=np.float32)
        self.brick_states = np.zeros((num_agents, self.num_bricks), np.float32)

        self.c_env: CBreakout | None = None
        self.human_action = None
        self.tick = 0
        self.reward_sum = 0
        self.score_sum = 0
        self.episodic_returns = np.zeros(num_agents, dtype=np.float32)
        self.scores = np.zeros(num_agents, dtype=np.int32)
        self.num_balls = np.full(num_agents, 5, dtype=np.int32)
        self.ball_speeds = np.full(
            num_agents, 256, dtype=np.float32
        )  # n_pixels per second
        self.hit_counters = np.zeros(num_agents, dtype=np.int32)
        self.balls_fired = np.zeros(num_agents, dtype=np.float32)
        self.wall_positions = np.array(
            [[0.0, 0.0], [0.0, 0.0], [self.width, 0.0]], dtype=np.float32
        )

        self.paddle_widths = np.full(num_agents, 62, dtype=np.int32)
        self.paddle_heights = np.full(num_agents, 8, dtype=np.int32)
        self.ball_width: int = 6
        self.ball_height: int = 6
        self.brick_width: int = self.width // self.num_brick_cols
        self.brick_height: int = 12
        self.brick_positions = self.generate_brick_positions()

        # This block required by advanced PufferLib env spec
        self.obs_size = 2 + 2 + self.num_bricks + 1
        low = np.array(
            [0.0] * 2
            + [0.0] * 2
            + [0.0] * 2
            + [0.0]
            + [0.0]
            + [0.0]
            + [0.0] * self.num_bricks
        )
        high = np.array(
            [width, height]
            + [width, height]
            + [self.ball_speeds[0]] * 2
            + [1.0]
            + [5.0]
            + [self.paddle_widths[0]]
            + [1.0] * self.num_bricks
        )
        self.observation_space = gymnasium.spaces.Box(
            low=low, high=high, dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Discrete(4)
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space
        self.num_agents = num_agents
        self.render_mode = render_mode
        self.emulated = None
        self.done = False
        self.buf = pufferlib.namespace(
            observations=np.zeros(
                (self.num_agents, *self.observation_space.shape), dtype=np.float32
            ),
            rewards=np.zeros(self.num_agents, dtype=np.float32),
            terminals=np.zeros(self.num_agents, dtype=bool),
            truncations=np.zeros(self.num_agents, dtype=bool),
            masks=np.ones(self.num_agents, dtype=bool),
        )
        self.actions = np.zeros(self.num_agents, dtype=np.uint8)

        if render_mode == "ansi":
            self.client = render.AnsiRender()
        elif render_mode == "rgb_array":
            self.client = render.RGBArrayRender()
        elif render_mode == "human":
            self.client = RaylibClient(
                self.width,
                self.height,
                self.num_brick_rows,
                self.num_brick_cols,
                self.brick_positions,
                self.ball_width,
                self.ball_height,
                self.brick_width,
                self.brick_height,
                self.fps,
            )
        else:
            raise ValueError(f"Invalid render mode: {render_mode}")

    def step(self, actions):
        self.actions[:] = actions

        if self.render_mode == "human" and self.human_action is not None:
            self.actions[0] = self.human_action
        # elif self.render_mode == "human":
        #     self.actions[0] = 0

        self.c_env.step(self.actions)

        info = {}
        self.reward_sum += self.buf.rewards.mean()
        self.score_sum += self.scores.mean()

        if self.tick % self.report_interval == 0:
            info["episodic_return"] = self.episodic_returns.mean()
            info["reward"] = self.reward_sum / self.report_interval

            self.reward_sum = 0
            self.score_sum = 0
            self.tick = 0

        self.tick += 1

        return (
            self.buf.observations,
            self.buf.rewards,
            self.buf.terminals,
            self.buf.truncations,
            info,
        )

    def reset(self, seed=None):
        if self.c_env is None:
            self.c_env = CBreakout(
                dt=self.dt,
                frameskip=self.frameskip,
                observations=self.buf.observations,
                rewards=self.buf.rewards,
                scores=self.scores,
                episodic_returns=self.episodic_returns,
                num_balls=self.num_balls,
                paddle_positions=self.paddle_positions,
                ball_positions=self.ball_positions,
                ball_velocities=self.ball_velocities,
                brick_positions=self.brick_positions,
                brick_states=self.brick_states,
                balls_fired=self.balls_fired,
                hit_counters=self.hit_counters,
                wall_positions=self.wall_positions,
                num_agents=self.num_agents,
                width=self.width,
                height=self.height,
                obs_size=self.obs_size,
                num_bricks=self.num_bricks,
                num_brick_rows=self.num_brick_rows,
                num_brick_cols=self.num_brick_cols,
                paddle_widths=self.paddle_widths,
                paddle_heights=self.paddle_heights,
                ball_width=self.ball_width,
                ball_height=self.ball_height,
                ball_speeds=self.ball_speeds,
                brick_width=self.brick_width,
                brick_height=self.brick_height,
            )

        return self.buf.observations, {}

    def render(self):
        if self.render_mode == "ansi":
            return self.client.render(self.grid)
        elif self.render_mode == "rgb_array":
            return self.client.render(self.grid)
        elif self.render_mode == "raylib":
            return self.client.render(self.grid)
        elif self.render_mode == "human":
            action = None

            if rl.IsKeyDown(rl.KEY_UP) or rl.IsKeyDown(rl.KEY_W):
                action = 1
            if rl.IsKeyDown(rl.KEY_LEFT) or rl.IsKeyDown(rl.KEY_A):
                action = 2
            if rl.IsKeyDown(rl.KEY_RIGHT) or rl.IsKeyDown(rl.KEY_D):
                action = 3

            self.human_action = action

            return self.client.render(
                self.paddle_positions[0],
                self.paddle_widths[0],
                self.paddle_heights[0],
                self.ball_positions[0],
                self.brick_states[0],
                self.scores[0],
                self.num_balls[0],
            )
        else:
            raise ValueError(f"Invalid render mode: {self.render_mode}")

    def close(self):
        pass

    def _calculate_scores(self):
        score = 0
        for agent_idx in range(self.num_agents):
            self.scores[agent_idx] = 0
            for brick_idx in range(self.num_bricks):
                if self.brick_states[agent_idx][brick_idx] == 0:
                    continue
                row = brick_idx // self.num_brick_cols
                self.scores[agent_idx] += 7 - 3 * (row // 2)

    def generate_brick_positions(self):
        brick_positions = np.zeros((self.num_bricks, 2), dtype=np.float32)
        y_offset = 50
        for row in range(self.num_brick_rows):
            for col in range(self.num_brick_cols):
                idx = row * self.num_brick_cols + col
                x = col * self.brick_width
                y = row * self.brick_height + y_offset
                brick_positions[idx][0] = x
                brick_positions[idx][1] = y
        return brick_positions


class RaylibClient:
    def __init__(
        self,
        width: int,
        height: int,
        num_brick_rows: int,
        num_brick_cols: int,
        brick_positions: np.ndarray,
        ball_width: float,
        ball_height: float,
        brick_width: int,
        brick_height: int,
        fps: float,
        play_sounds: bool = False,
        use_pufferfish: bool = False,
    ) -> None:
        self.width = width
        self.height = height
        self.ball_width = ball_width
        self.ball_height = ball_height
        self.brick_width = brick_width
        self.brick_height = brick_height
        self.num_brick_rows = num_brick_rows
        self.num_brick_cols = num_brick_cols
        self.brick_positions = brick_positions

        self.running = False

        self.BRICK_COLORS = [
            colors.RED,
            colors.ORANGE,
            colors.YELLOW,
            colors.GREEN,
            colors.SKYBLUE,
            colors.BLUE,
        ]

        # Initialize raylib window
        rl.InitWindow(width, height, "PufferLib Ray Breakout".encode())
        rl.SetTargetFPS(fps)

        sprite_sheet_path = os.path.join(
            *self.__module__.split(".")[:-1], "puffer_chars.png"
        )
        self.puffer = rl.LoadTexture(sprite_sheet_path.encode())

        sound_path = os.path.join(*self.__module__.split(".")[:-1], "hit.wav")

        if play_sounds:
            rl.InitAudioDevice()
        self.hit_sound = rl.LoadSound(sound_path.encode())

        self.use_pufferfish = use_pufferfish

    def render(
        self,
        paddle_position: np.ndarray,
        paddle_width: int,
        paddle_height: int,
        ball_position: np.ndarray,
        brick_states: np.ndarray,
        score: float,
        num_balls: int,
    ) -> None:
        rl.BeginDrawing()
        rl.ClearBackground(render.PUFF_BACKGROUND)

        # Draw the paddle
        paddle_x, paddle_y = paddle_position
        rl.DrawRectangle(
            int(paddle_x),
            int(paddle_y),
            paddle_width,
            paddle_height,
            colors.DARKGRAY,
        )

        # Draw the ball
        ball_x, ball_y = ball_position

        source_rect = (128, 0, 128, 128)
        dest_rect = (ball_x, ball_y, 3 * self.ball_width, 3 * self.ball_height)

        if self.use_pufferfish:
            rl.DrawTexturePro(
                self.puffer, source_rect, dest_rect, (0, 0), 0.0, colors.WHITE
            )
        else:
            rl.DrawRectangle(
                int(ball_x),
                int(ball_y),
                self.ball_width,
                self.ball_height,
                colors.WHITE,
            )

        # Draw the bricks
        for row in range(self.num_brick_rows):
            for col in range(self.num_brick_cols):
                idx = row * self.num_brick_cols + col
                if brick_states[idx] == 1:
                    continue

                x, y = self.brick_positions[idx]
                brick_color = self.BRICK_COLORS[row]
                rl.DrawRectangle(
                    int(x),
                    int(y),
                    self.brick_width,
                    self.brick_height,
                    brick_color,
                )

        # Draw Score
        score_text = f"Score: {int(score)}".encode("ascii")
        rl.DrawText(score_text, 10, 10, 20, colors.WHITE)

        num_balls_text = f"Balls: {num_balls}".encode("ascii")
        rl.DrawText(num_balls_text, self.width - 80, 10, 20, colors.WHITE)

        rl.EndDrawing()

        ball_box = (
            int(ball_x),
            int(ball_y),
            self.ball_width,
            self.ball_height,
        )
        paddle_box = (
            int(paddle_x),
            int(paddle_y),
            paddle_width,
            paddle_height,
        )

        if rl.CheckCollisionRecs(ball_box, paddle_box):
            rl.PlaySound(self.hit_sound)

        for row in range(self.num_brick_rows):
            for col in range(self.num_brick_cols):
                idx = row * self.num_brick_cols + col
                if brick_states[idx] == 1:
                    continue

                x, y = self.brick_positions[idx]
                brick_color = self.BRICK_COLORS[row]
                brick_box = (int(x), int(y), self.brick_width, self.brick_height)
                if rl.CheckCollisionRecs(ball_box, brick_box):
                    rl.PlaySound(self.hit_sound)

        return render.cdata_to_numpy()

    def close(self) -> None:
        rl.close_window()


def test_performance(timeout=20, atn_cache=1024, num_envs=400):
    tick = 0

    import time

    start = time.time()
    while time.time() - start < timeout:
        atns = actions[tick % atn_cache]
        env.step(atns)
        tick += 1

    print(f"SPS: %f", num_envs * tick / (time.time() - start))


if __name__ == "__main__":
    # Run with c profile
    from cProfile import run

    num_envs = 100
    env = PufferBreakout(num_agents=num_envs)
    env.reset()
    actions = np.random.randint(0, 9, (1024, num_envs))
    test_performance(20, 1024, num_envs)
    # exit(0)

    run("test_performance(20)", "stats.profile")
    import pstats
    from pstats import SortKey

    p = pstats.Stats("stats.profile")
    p.sort_stats(SortKey.TIME).print_stats(25)
    exit(0)

    # test_performance(10)
