# cython: language_level=3
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: nonecheck=False
# cython: profile=False

cimport numpy as cnp
from libc.math cimport pi, sin, cos
from libc.stdlib cimport rand
import numpy as np

cdef:
    int NOOP = 0
    int FIRE = 1
    int LEFT = 2
    int RIGHT = 3
    int HALF_SCORE = 432
    int MAX_SCORE = 864
    int BALL_SPEED = 256
    int MAX_BALL_SPEED = 448
    int PADDLE_WIDTH = 62
    int HALF_PADDLE_WIDTH = 31

cdef class CBreakout:
    cdef:
        float[:, :] observations
        float[:] rewards
        int[:] scores
        float[:] episodic_returns
        float[:, :] paddle_positions
        float[:, :] ball_positions
        float[:, :] ball_velocities
        float[:, :] brick_positions
        float[:, :] brick_states
        float[:] balls_fired
        float[:, :] wall_positions
        int frameskip
        int width
        int height
        int obs_size
        int num_bricks
        int num_brick_rows
        int num_brick_cols
        int[:] paddle_widths
        int[:] paddle_heights
        int ball_width
        int ball_height
        int brick_width
        int brick_height
        int num_agents
        int[:] num_balls
        int[:] hit_counters
        float[:] ball_speeds
        unsigned char[:] check_brick_collisions
        float dt

    def __init__(self, float dt, int frameskip, cnp.ndarray observations, cnp.ndarray rewards, cnp.ndarray scores, cnp.ndarray episodic_returns, cnp.ndarray num_balls, cnp.ndarray paddle_positions, cnp.ndarray ball_positions, cnp.ndarray ball_velocities, cnp.ndarray brick_positions, cnp.ndarray brick_states, cnp.ndarray balls_fired, cnp.ndarray wall_positions, cnp.ndarray paddle_widths, cnp.ndarray paddle_heights, cnp.ndarray ball_speeds, cnp.ndarray hit_counters, int num_agents, int width, int height, int ball_width, int ball_height, int brick_width, int brick_height, int obs_size, int num_bricks, int num_brick_rows, int num_brick_cols):
        cdef int agent_idx

        self.dt = dt
        self.frameskip = frameskip
        self.observations = observations
        self.rewards = rewards
        self.scores = scores
        self.episodic_returns = episodic_returns
        self.num_balls = num_balls
        self.paddle_positions = paddle_positions
        self.ball_positions = ball_positions
        self.ball_velocities = ball_velocities
        self.brick_positions = brick_positions
        self.brick_states = brick_states
        self.wall_positions = wall_positions
        self.balls_fired = balls_fired
        self.width = width
        self.height = height
        self.paddle_widths = paddle_widths
        self.paddle_heights = paddle_heights
        self.ball_width = ball_width
        self.ball_height = ball_height
        self.brick_width = brick_width
        self.brick_height = brick_height
        self.obs_size = obs_size
        self.num_bricks = num_bricks
        self.num_brick_rows = num_brick_rows
        self.num_brick_cols = num_brick_cols
        self.ball_speeds = ball_speeds
        self.hit_counters = hit_counters
        self.num_agents = num_agents
        self.check_brick_collisions = np.zeros(self.num_agents, dtype=np.uint8)


        for agent_idx in range(self.num_agents):
            self.reset(agent_idx)

    cdef void compute_observations(self):
        cdef int agent_idx
        for agent_idx in range(self.num_agents):
            self.observations[agent_idx, 0] = self.paddle_positions[agent_idx, 0]
            self.observations[agent_idx, 1] = self.paddle_positions[agent_idx, 1]
            self.observations[agent_idx, 2] = self.ball_positions[agent_idx, 0]
            self.observations[agent_idx, 3] = self.ball_positions[agent_idx, 1]
            self.observations[agent_idx, 4] = self.ball_velocities[agent_idx, 0]
            self.observations[agent_idx, 5] = self.ball_velocities[agent_idx, 1]
            self.observations[agent_idx, 6] = self.balls_fired[agent_idx]
            self.observations[agent_idx, 7] = self.num_balls[agent_idx]
            self.observations[agent_idx, 8] = self.paddle_widths[agent_idx]
            self.observations[agent_idx, 9:] = self.brick_states[agent_idx, :]

    cdef void reset_paddle(self, int agent_idx):
        self.paddle_positions[agent_idx][0] = self.width / 2.0 - self.paddle_widths[agent_idx] // 2
        self.paddle_positions[agent_idx][1] = self.height - self.paddle_heights[agent_idx] - 10

    cdef void reset_ball(self, int agent_idx):
        self.ball_positions[agent_idx][0] = self.width // 2 - self.ball_width // 2
        self.ball_positions[agent_idx][1] = self.height // 2 - 30

        self.ball_velocities[agent_idx][0] = 0.0
        self.ball_velocities[agent_idx][1] = 0.0

        self.balls_fired[agent_idx] = 0.0

    cdef void reset_bricks(self, int agent_idx):
        self.brick_states[agent_idx][:] = 0.0

    cdef void reset(self, int agent_idx):

        self.scores[agent_idx] = 0
        self.num_balls[agent_idx] = 5
        self.hit_counters[agent_idx] = 0
        self.ball_speeds[agent_idx] = BALL_SPEED
        self.paddle_widths[agent_idx] = PADDLE_WIDTH
        self.reset_paddle(agent_idx)
        self.reset_ball(agent_idx)
        self.reset_bricks(agent_idx)


    cdef void handle_action(self, int agent_idx, int action):
        if action == NOOP:
            pass
        elif action == FIRE and self.balls_fired[agent_idx] == 0.0:
            self.balls_fired[agent_idx] = 1.0

            direction = pi / 3.25

            if rand() % 2 == 0:
                self.ball_velocities[agent_idx][0] = sin(direction) * self.ball_speeds[agent_idx] * self.dt
            else:
                self.ball_velocities[agent_idx][0] = -sin(direction) * self.ball_speeds[agent_idx] * self.dt

            self.ball_velocities[agent_idx][1] = cos(direction) * self.ball_speeds[agent_idx] * self.dt

        elif action == LEFT:
            self.paddle_positions[agent_idx][0] -= 512 * self.dt
            self.paddle_positions[agent_idx][0] = max(0, self.paddle_positions[agent_idx][0])
        elif action == RIGHT:
            self.paddle_positions[agent_idx][0] += 512 * self.dt
            self.paddle_positions[agent_idx][0] = min(self.width - self.paddle_widths[agent_idx], self.paddle_positions[agent_idx][0])

        self.ball_positions[agent_idx][0] += self.ball_velocities[agent_idx][0]
        self.ball_positions[agent_idx][1] += self.ball_velocities[agent_idx][1]

    cdef bint is_game_over(self, int agent_idx):
        return (
            self.num_balls[agent_idx] < 0
            or self.scores[agent_idx] == MAX_SCORE
        )

    def step(self, cnp.ndarray[unsigned char, ndim=1] actions):
        cdef int action, agent_idx
        cdef float direction

        self.rewards[:] = 0.0

        for agent_idx in range(self.num_agents):

            action = actions[agent_idx]

            for _ in range(self.frameskip):
                self.handle_action(agent_idx, action)
                self.handle_collisions(agent_idx)


            if self.ball_positions[agent_idx][1] >= self.paddle_positions[agent_idx][1] + self.paddle_heights[agent_idx]:
                self.num_balls[agent_idx] -= 1
                self.reset_ball(agent_idx)
                if self.is_game_over(agent_idx):
                    self.episodic_returns[agent_idx] = self.scores[agent_idx]
                    self.reset(agent_idx)

        self.compute_observations()

    cdef inline void handle_collisions(self, agent_idx):
        if self.handle_brick_ball_collisions(agent_idx):
            return
        if self.handle_paddle_ball_collisions(agent_idx):
            return
        if self.handle_wall_ball_collisions(agent_idx):
            return

    cdef inline bint check_collision_discrete(self, const float x, float y, int width, int height, const float other_x, float other_y, int other_width, int other_height):
        if x + width <= other_x or other_x + other_width <= x:
            return False
        if y + height <= other_y or other_y + other_height <= y:
            return False

        # If none of the above conditions are met, the rectangles must be overlapping
        return True

    cdef bint handle_paddle_ball_collisions(self, int agent_idx):
        cdef float base_angle = pi / 4
        cdef float angle, relative_intersection

        # Check if ball is above the paddle
        if self.ball_positions[agent_idx][1] + self.ball_height < self.paddle_positions[agent_idx][1]:
            return False

        # Check for collision
        if self.check_collision_discrete(self.paddle_positions[agent_idx][0], self.paddle_positions[agent_idx][1], self.paddle_widths[agent_idx], self.paddle_heights[agent_idx],
                                         self.ball_positions[agent_idx][0], self.ball_positions[agent_idx][1], self.ball_width, self.ball_height):

            relative_intersection = ((self.ball_positions[agent_idx][0] + self.ball_width//2) - self.paddle_positions[agent_idx][0]) / self.paddle_widths[agent_idx]
            angle = -base_angle + relative_intersection * 2 * base_angle

            self.ball_velocities[agent_idx][0] = sin(angle) * self.ball_speeds[agent_idx] * self.dt
            self.ball_velocities[agent_idx][1] = -cos(angle) * self.ball_speeds[agent_idx] * self.dt

            self.hit_counters[agent_idx] += 1
            self.check_brick_collisions[agent_idx] = 1

            if self.ball_speeds[agent_idx] < MAX_BALL_SPEED and self.hit_counters[agent_idx] % 4 == 0:
                self.ball_speeds[agent_idx] += 64
            if self.scores[agent_idx] == HALF_SCORE:
                self.brick_states[agent_idx][:] = 0.0

            return True

        return False


    cdef bint handle_wall_ball_collisions(self, int agent_idx):
        if self.ball_positions[agent_idx][0] > 0 and self.ball_positions[agent_idx][0] + self.ball_width < self.width and self.ball_positions[agent_idx][1] > 0:
            return False
        # Left Wall Collision
        if self.check_collision_discrete(self.wall_positions[0][0] - 50, self.wall_positions[0][1], 50, self.height,
                                        self.ball_positions[agent_idx][0], self.ball_positions[agent_idx][1],
                                        self.ball_width, self.ball_height):
            self.ball_positions[agent_idx][0] = 0
            self.ball_velocities[agent_idx][0] *= -1
            return True

        # Top Wall Collision
        if self.check_collision_discrete(self.wall_positions[1][0], self.wall_positions[1][1] - 50, self.width, 50,
                                        self.ball_positions[agent_idx][0], self.ball_positions[agent_idx][1],
                                        self.ball_width, self.ball_height):
            self.ball_positions[agent_idx][1] = 0.0
            self.ball_velocities[agent_idx][1] *= -1
            self.paddle_widths[agent_idx] = HALF_PADDLE_WIDTH
            self.check_brick_collisions[agent_idx] = 1
            return True

        # Right Wall Collision
        if self.check_collision_discrete(self.wall_positions[2][0], self.wall_positions[2][1], 50, self.height,
                                        self.ball_positions[agent_idx][0], self.ball_positions[agent_idx][1],
                                        self.ball_width, self.ball_height):
            self.ball_positions[agent_idx][0] = self.width - self.ball_width
            self.ball_velocities[agent_idx][0] *= -1
            return True

        return False


    cdef bint handle_brick_ball_collisions(self, int agent_idx):
        cdef int brick_idx, score, row

        if self.check_brick_collisions[agent_idx] == 0 or self.ball_positions[agent_idx][1] > self.brick_positions[self.num_bricks-1][1] + self.brick_height:
            return False

        # Loop over bricks in reverse to check lower bricks first
        for brick_idx in range(self.num_bricks - 1, -1, -1):
            if self.brick_states[agent_idx][brick_idx] == 1.0:
                continue

            if self.check_collision_discrete(self.brick_positions[brick_idx][0], self.brick_positions[brick_idx][1], self.brick_width, self.brick_height, self.ball_positions[agent_idx][0], self.ball_positions[agent_idx][1], self.ball_width, self.ball_height):
                self.brick_states[agent_idx][brick_idx] = 1.0
                score = 7 - 3 * (brick_idx // self.num_brick_cols // 2)
                self.rewards[agent_idx] += score
                self.scores[agent_idx] += score

                row =  brick_idx // self.num_brick_cols
                if row < 3:
                    self.ball_speeds[agent_idx] = MAX_BALL_SPEED


                self.ball_velocities[agent_idx][1] *= -1
                self.check_brick_collisions[agent_idx] = 0

                return True

        return False
