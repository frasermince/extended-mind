from gymnasium.core import ActType, ObsType, WrapperObsType
from gymnasium import spaces
from minigrid.core.constants import OBJECT_TO_IDX
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Box, Goal, Lava, Wall, WorldObj
from minigrid.minigrid_env import MiniGridEnv
from gymnasium.wrappers import TransformObservation
from minigrid.utils.rendering import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_circle,
    point_in_rect,
    point_in_triangle,
    rotate_fn,
)

import gymnasium as gym
import numpy as np
import heapq

from minigrid.wrappers import ImgObsWrapper
from typing import Any, SupportsFloat

from enum import IntEnum

TILE_PIXELS = 8


class PartialAndTotalRecordVideo(gym.wrappers.RecordVideo):
    def _capture_frame(self):
        assert self.recording, "Cannot capture a frame, recording wasn't started."

        frame = self.render()
        if isinstance(frame, list):
            if len(frame) == 0:  # render was called
                return
            self.render_history += frame
            frame = frame[-1]

        if isinstance(frame, np.ndarray):
            self.recorded_frames.append(frame)
        else:
            self.stop_recording()
            logger.warn(
                f"Recording stopped: expected type of frame returned by render to be a numpy array, got instead {type(frame)}."
            )

    @property
    def enabled(self):
        return self.recording

    def render(self):
        img_total, img_partial = self.env.unwrapped.render_path_visualizations()
        # img_partial has 1 channel; repeat to make it 3 channels
        # Handles the fact that the partial view is grayscale
        if img_partial.ndim == 3 and img_partial.shape[2] == 1:
            img_partial = np.repeat(img_partial, 3, axis=2)

        # INSERT_YOUR_CODE
        # 8x the size of img_total and img_partial using nearest neighbor upsampling
        def upsample(img, scale):
            h, w, c = img.shape
            return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)

        img_total = upsample(img_total, 8)
        img_partial = upsample(img_partial, 8)
        # Stack img_total and img_partial vertically with a padding in between

        # Ensure both images have the same width; if not, pad the smaller one
        h1, w1, c1 = img_total.shape
        h2, w2, c2 = img_partial.shape
        pad_color = 255  # white padding

        # Determine the max width and number of channels
        max_w = max(w1, w2)
        max_c = max(c1, c2)

        def pad_img(img, target_w, target_c):
            h, w, c = img.shape
            # Pad width if needed, split between left and right
            if w < target_w:
                total_pad = target_w - w
                pad_left = total_pad // 2
                pad_right = total_pad - pad_left
                pad_width = ((0, 0), (pad_left, pad_right), (0, 0))
                img = np.pad(img, pad_width, mode="constant", constant_values=pad_color)
            # Pad channels if needed
            if c < target_c:
                total_pad = target_c - c
                pad_left = total_pad // 2
                pad_right = total_pad - pad_left
                pad_channels = ((0, 0), (0, 0), (pad_left, pad_right))
                img = np.pad(
                    img, pad_channels, mode="constant", constant_values=pad_color
                )
            return img

        img_total_padded = pad_img(img_total, max_w, max_c)
        img_partial_padded = pad_img(img_partial, max_w, max_c)

        # Create a padding row (e.g., 10 pixels high)
        pad_height = 10
        padding = np.full((pad_height, max_w, max_c), pad_color, dtype=img_total.dtype)
        padding_bottom = np.full((30, max_w, max_c), pad_color, dtype=img_total.dtype)

        # Concatenate: total on top, then padding, then partial
        render_out = [
            np.concatenate(
                [img_total_padded, padding, img_partial_padded, padding_bottom], axis=0
            )
        ]
        if self.recording and isinstance(render_out, list):
            self.recorded_frames += render_out

        if len(self.render_history) > 0:
            tmp_history = self.render_history
            self.render_history = []
            return tmp_history + render_out
        else:
            return render_out


class DirectionlessGrid(Grid):
    def __init__(self, *args, **kwargs):
        self.unique_tiles = kwargs.pop("unique_tiles", None)
        self.padded_unique_tiles = kwargs.pop("padded_unique_tiles", None)
        self.tile_global_indices = kwargs.pop("tile_global_indices", None)
        self.show_grid_lines = kwargs.pop("show_grid_lines", False)
        self.show_walls_pov = kwargs.pop("show_walls_pov", False)
        self.pad_width = kwargs.pop("pad_width", None)
        self.seed = kwargs.pop("seed", None)
        super().__init__(*args, **kwargs)

    @classmethod
    def render_tile(
        cls,
        grid: "DirectionlessGrid",
        obj: WorldObj | None,
        agent_dir: int | None = None,
        highlight: bool = False,
        tile_size: int = TILE_PIXELS,
        subdivs: int = 1,
        i: int = 0,
        j: int = 0,
        reveal_all: bool = False,
    ) -> np.ndarray:
        """
        Render a tile and cache the result
        """

        if reveal_all:
            key: tuple[Any, ...] = (
                tile_size,
                obj,
                grid.tile_global_indices[i, j][0],
                grid.tile_global_indices[i, j][1],
                reveal_all,
                grid.seed,
                agent_dir,
            )
        else:
            key: tuple[Any, ...] = (
                tile_size,
                grid.tile_global_indices[i, j][0],
                grid.tile_global_indices[i, j][1],
                reveal_all,
                grid.seed,
            )

        key = obj.encode() + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        if isinstance(obj, Wall):
            if reveal_all:
                img = np.zeros(
                    shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8
                )
            else:
                img = np.zeros(
                    shape=(tile_size * subdivs, tile_size * subdivs, 1), dtype=np.uint8
                )
        else:
            if reveal_all:
                img = grid.unique_tiles[i, j].copy()
            else:
                img = np.expand_dims(grid.unique_tiles[i, j][:, :, 0].copy(), axis=-1)

        # Draw the grid lines (top and left edges)
        if grid.show_grid_lines or reveal_all:
            if reveal_all:
                fill_coords(img, point_in_rect(0, 0.0625, 0, 1), (100, 100, 100))
                fill_coords(img, point_in_rect(0, 1, 0, 0.0625), (100, 100, 100))
            else:
                fill_coords(img, point_in_rect(0, 0.0625, 0, 1), (100,))
                fill_coords(img, point_in_rect(0, 1, 0, 0.0625), (100,))

        if obj is not None and obj.type != "wall" and reveal_all:
            obj.render(img)

        # Overlay the agent on top
        if agent_dir is not None and reveal_all:
            tri_fn = point_in_circle(
                0.5,
                0.5,
                0.3,
            )

            # Rotate the agent based on its direction
            # tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * agent_dir)
            fill_coords(img, tri_fn, (255, 0, 0))

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    def render(
        self,
        tile_size: int,
        agent_pos: tuple[int, int],
        agent_dir: int | None = None,
        highlight_mask: np.ndarray | None = None,
        reveal_all: bool = False,
        path: list[tuple[int, int]] | None = None,
    ) -> np.ndarray:
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        if reveal_all:
            img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)
        else:
            img = np.zeros(shape=(height_px, width_px, 1), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)

                agent_here = np.array_equal(agent_pos, (i, j))
                assert highlight_mask is not None

                if isinstance(cell, Goal) and cell.color == "green" and not reveal_all:
                    cell = None
                if (
                    isinstance(cell, Wall)
                    and not reveal_all
                    and not self.show_walls_pov
                ):
                    cell = None
                tile_img = DirectionlessGrid.render_tile(
                    self,
                    cell,
                    agent_dir=agent_dir if agent_here else None,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size,
                    i=i,
                    j=j,
                    reveal_all=reveal_all,
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size

                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def slice(self, topX: int, topY: int, width: int, height: int) -> Grid:
        """
        Get a subset of the grid
        """
        relevant_unique = self.padded_unique_tiles[
            self.pad_width + topX : self.pad_width + topX + width,
            self.pad_width + topY : self.pad_width + topY + height,
            :,
            :,
        ]

        relevant_unique_global_indices = self.tile_global_indices[
            topX : topX + width, topY : topY + height, :
        ]

        grid = DirectionlessGrid(
            width,
            height,
            tile_global_indices=relevant_unique_global_indices,
            unique_tiles=relevant_unique,
            padded_unique_tiles=self.padded_unique_tiles,
            show_grid_lines=self.show_grid_lines,
            show_walls_pov=self.show_walls_pov,
            pad_width=self.pad_width,
            seed=self.seed,
        )

        for j in range(0, height):
            for i in range(0, width):
                x = topX + i
                y = topY + j

                if 0 <= x < self.width and 0 <= y < self.height:
                    v = self.get(x, y)
                else:
                    v = Wall()

                grid.set(i, j, v)

        return grid


class ImgObsPositionWrapper(gym.ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.

    Example:
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import ImgObsWrapper
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> obs, _ = env.reset()
        >>> obs.keys()
        dict_keys(['image', 'direction', 'mission'])
        >>> env = ImgObsWrapper(env)
        >>> obs, _ = env.reset()
        >>> obs.shape
        (7, 7, 3)
    """

    def __init__(self, env):
        """A wrapper that makes image the only observation.

        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.observation_space = env.observation_space.spaces["image"]

    def observation(self, obs):
        return obs["image"], obs


class Actions(IntEnum):
    left = 0
    forward = 1
    right = 2
    backward = 3


class SaltAndPepper(MiniGridEnv):
    """
    ## Description

    Depending on the `obstacle_type` parameter:
    - `Lava` - The agent has to reach the green goal square on the other corner
        of the room while avoiding rivers of deadly lava which terminate the
        episode in failure. Each lava stream runs across the room either
        horizontally or vertically, and has a single crossing point which can be
        safely used; Luckily, a path to the goal is guaranteed to exist. This
        environment is useful for studying safety and safe exploration.
    - otherwise - Similar to the `LavaCrossing` environment, the agent has to
        reach the green goal square on the other corner of the room, however
        lava is replaced by walls. This MDP is therefore much easier and maybe
        useful for quickly testing your algorithms.

    ## Mission Space
    Depending on the `obstacle_type` parameter:
    - `Lava` - "avoid the lava and get to the green goal square"
    - otherwise - "find the opening and get to the green goal square"

    ## Action Space

    | Num | Name         | Action        |
    |-----|--------------|---------------|
    | 0   | left         | Move left     |
    | 1   | forward      | Move forward  |
    | 2   | right        | Move right    |
    | 3   | backward     | Move backward |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1.0' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. The agent falls into lava.
    3. Timeout (see `max_steps`).

    ## Registered Configurations

    S: size of the map SxS.
    N: number of valid crossings across lava or walls from the starting position
    to the goal

    - `Lava` :
        - `MiniGrid-LavaCrossingS9N1-v0`
        - `MiniGrid-LavaCrossingS9N2-v0`
        - `MiniGrid-LavaCrossingS9N3-v0`
        - `MiniGrid-LavaCrossingS11N5-v0`

    - otherwise :
        - `MiniGrid-SimpleCrossingS9N1-v0`
        - `MiniGrid-SimpleCrossingS9N2-v0`
        - `MiniGrid-SimpleCrossingS9N3-v0`
        - `MiniGrid-SimpleCrossingS11N5-v0`

    """

    def __init__(
        self,
        size=9,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.seed = kwargs.pop("seed", None)
        # self.num_crossings = num_crossings
        # self.obstacle_type = obstacle_type
        self.goal_position = None
        self.path_episode_threshold = 4000
        self.num_episodes = 0

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = int(size**2.0)

        show_grid_lines = kwargs.pop("show_grid_lines", False)
        show_walls_pov = kwargs.pop("show_walls_pov", False)
        agent_view_size = kwargs.pop("agent_view_size", 5)
        # self.invisible_goal = kwargs.pop("invisible_goal", False)
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,  # Set this to True for maximum speed
            max_steps=max_steps,
            agent_pov=True,
            agent_view_size=agent_view_size,
            tile_size=TILE_PIXELS,
            **kwargs,
        )
        self.show_grid_lines = show_grid_lines
        self.show_walls_pov = show_walls_pov
        self.actions = Actions
        image_observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(
                self.agent_view_size * TILE_PIXELS,
                self.agent_view_size * TILE_PIXELS,
                1,
            ),
            dtype="uint8",
        )
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Dict(
            {
                "image": image_observation_space,
                "direction": gym.spaces.Discrete(4),
                "mission": mission_space,
                "position": gym.spaces.Box(
                    low=0,
                    high=size - 1,
                    shape=(2,),
                    dtype="int64",
                ),
            }
        )

        self.tile_size = TILE_PIXELS
        self.size = size

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.num_episodes += 1
        return super().reset(seed=seed, options=options)

    def _reward(self) -> float:
        return 1.0

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    def _gen_unique_tiles(self):
        # subdivs = 3
        # Generate all unique random black and white pixels for all cells at once
        # Calculate section size (8x8 pixels)
        # section_size = 1
        num_sections_per_tile = TILE_PIXELS

        # Generate random black/white sections for all cells
        # Generate random black/white sections for all cells
        # Each section will be either all black (0,0,0) or all white (255,255,255)
        pad_width = int(np.ceil(self.agent_view_size / 2))
        section_colors = self.np_random.choice(
            [0, 255],
            size=(
                self.size + pad_width * 2,
                self.size + pad_width * 2,
                num_sections_per_tile,
                num_sections_per_tile,
            ),
            p=[0.10, 0.90],
        )
        # Expand to 3 channels - all channels get the same value
        padded_tiles = np.stack([section_colors] * 3, axis=-1)

        # Expand sections to full tile size
        # expanded_tiles = np.repeat(
        #     np.repeat(all_tile_pixels, section_size, axis=2), section_size, axis=3
        # )
        # Pad by agent_view_size on each side
        x_indices = np.arange(pad_width, self.size + pad_width)
        y_indices = np.arange(pad_width, self.size + pad_width)
        tile_global_indices = np.stack(
            np.meshgrid(x_indices, y_indices, indexing="ij"), axis=-1
        )
        return (
            padded_tiles[pad_width:-pad_width, pad_width:-pad_width, :, :],
            padded_tiles,
            pad_width,
            tile_global_indices,
        )

    def _gen_grid(self, width, height):
        assert width % 2 == 1 and height % 2 == 1  # odd size

        # Create an empty grid
        (
            self.unique_tiles,
            self.padded_unique_tiles,
            self.pad_width,
            self.tile_global_indices,
        ) = self._gen_unique_tiles()

        self.grid = DirectionlessGrid(
            width,
            height,
            show_grid_lines=self.show_grid_lines,
            show_walls_pov=self.show_walls_pov,
            unique_tiles=self.unique_tiles,
            padded_unique_tiles=self.padded_unique_tiles,
            pad_width=self.pad_width,
            tile_global_indices=self.tile_global_indices,
            seed=self.seed,
        )

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        # Place the agent in the bottom middle
        self.agent_pos = np.array((width // 2, height - 2))
        self.agent_dir = 3

        self.goal_position = (2, 3)

        self.put_obj(Goal(), *self.goal_position)

        self.mission = "get to the green goal square"
        self.path = self.compute_dijkstra_path()

        img = self.grid.render(
            TILE_PIXELS,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=None,
            reveal_all=False,
            path=self.path,
        )

    
    def compute_dijkstra_path(self):
        start = tuple(self.agent_pos)
        goal = self.goal_position
        width, height = self.grid.width, self.grid.height

        visited = set()
        came_from = {}
        cost_so_far = {start: 0}

        heap = [(0, start)]
        
        while heap:
            current_cost, current = heapq.heappop(heap)

            if current == goal:
                break

            if current in visited:
                continue
            visited.add(current)

            x, y = current
            neighbors = [
                (x-1, y), (x+1, y),
                (x, y-1), (x, y+1)
            ]
            for nx, ny in neighbors:
                if 0 <= nx < width and 0 <= ny < height:
                    cell = self.grid.get(nx, ny)
                    if isinstance(cell, Wall) or isinstance(cell, Lava):
                        continue
                    new_cost = current_cost + 1
                    if (nx, ny) not in cost_so_far or new_cost < cost_so_far[(nx, ny)]:
                        cost_so_far[(nx, ny)] = new_cost
                        heapq.heappush(heap, (new_cost, (nx, ny)))
                        came_from[(nx, ny)] = current

        # Reconstruct path
        current = goal
        path = [current]
        while current != start:
            current = came_from.get(current)
            if current is None:
                return []  # No path
            path.append(current)

        return path[::-1]  # Start to goal


    def draw_path_on_grid(self, img: np.ndarray, path: list[tuple[int, int]], color=(0, 0, 255)) -> np.ndarray:
        """Overlay a path on a full grid image"""
        img = img.copy()
        for (x, y) in path:
            px = x * self.tile_size
            py = y * self.tile_size
            img[py:py+self.tile_size, px:px+self.tile_size] = color
        return img
    

    def draw_path_on_agent_view(self, img: np.ndarray, path: list[tuple[int, int]], color=(0, 0, 255)) -> np.ndarray:
        """Overlay the visible path on the agent's POV image"""
        img = img.copy()

        # If grayscale (1 channel), repeat to RGB
        if img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)

        topX, topY, botX, botY = self.get_view_exts()

        for (x, y) in path:
            if topX <= x < botX and topY <= y < botY:
                vx = x - topX
                vy = y - topY
                px = vx * self.tile_size
                py = vy * self.tile_size
                img[py:py+self.tile_size, px:px+self.tile_size] = color
        return img
    

    def render_path_visualizations(self):
        path = self.compute_dijkstra_path()

        full_img = self.get_full_render(highlight=False, tile_size=self.tile_size, reveal_all=True)
        partial_img = self.get_pov_render(tile_size=self.tile_size)

        full_img_with_path = self.draw_path_on_grid(full_img, path)
        partial_img_with_path = self.draw_path_on_agent_view(partial_img, path)

        return full_img_with_path, partial_img_with_path


    def get_view_exts(self, agent_view_size=None):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        if agent_view_size is None, use self.agent_view_size
        """

        agent_view_size = agent_view_size or self.agent_view_size

        assert self.agent_dir == 3

        topX = self.agent_pos[0] - agent_view_size // 2
        topY = self.agent_pos[1] - agent_view_size // 2

        botX = topX + agent_view_size
        botY = topY + agent_view_size

        return topX, topY, botX, botY

    def get_pov_render(self, tile_size):
        """
        Render an agent's POV observation for visualization
        """
        grid, vis_mask = self.gen_obs_grid()

        # Render the whole grid
        img = grid.render(
            tile_size,
            agent_pos=(self.agent_view_size // 2, self.agent_view_size - 1),
            agent_dir=3,
            highlight_mask=vis_mask,
        )

        return img

    def gen_obs_grid(self, agent_view_size=None):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        if agent_view_size is None, self.agent_view_size is used
        """

        topX, topY, botX, botY = self.get_view_exts(agent_view_size)

        agent_view_size = agent_view_size or self.agent_view_size

        grid = self.grid.slice(topX, topY, agent_view_size, agent_view_size)

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(
                agent_pos=(agent_view_size // 2, agent_view_size - 1)
            )
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        agent_pos = grid.width // 2, grid.height // 2
        if self.carrying:
            grid.set(*agent_pos, self.carrying)
        else:
            grid.set(*agent_pos, None)

        return grid, vis_mask

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """
        grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = grid.render(
            TILE_PIXELS,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=None,
            reveal_all=False,
        )

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)

        obs = {
            "image": image,
            "direction": self.agent_dir,
            "mission": self.mission,
            "position": self.agent_pos,
        }
        return obs

    # Customized to remove highlight mask
    def get_full_render(self, highlight, tile_size, reveal_all=False):
        """
        Render a non-paratial observation for visualization
        """
        # Compute which cells are visible to the agent
        _, vis_mask = self.gen_obs_grid()

        # Compute the world coordinates of the bottom-left corner
        # of the agent's view area
        f_vec = self.dir_vec
        r_vec = self.right_vec
        top_left = (
            self.agent_pos
            + f_vec * (self.agent_view_size - 1)
            - r_vec * (self.agent_view_size // 2)
        )

        # For each cell in the visibility mask
        for vis_j in range(0, self.agent_view_size):
            for vis_i in range(0, self.agent_view_size):
                # If this cell is not visible, don't highlight it
                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                if abs_i < 0 or abs_i >= self.width:
                    continue
                if abs_j < 0 or abs_j >= self.height:
                    continue

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agent_pos,
            self.agent_dir,
            highlight_mask=None,
            reveal_all=reveal_all,
        )

        return img

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Move left
        if action == self.actions.left:
            fwd_pos = np.array((self.agent_pos[0] - 1, self.agent_pos[1]))
            fwd_cell = self.grid.get(*fwd_pos)
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self._reward()
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        # Move right
        elif action == self.actions.right:
            fwd_pos = np.array((self.agent_pos[0] + 1, self.agent_pos[1]))
            fwd_cell = self.grid.get(*fwd_pos)
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self._reward()
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        # Move forward
        elif action == self.actions.forward:
            fwd_pos = np.array((self.agent_pos[0], self.agent_pos[1] - 1))
            fwd_cell = self.grid.get(*fwd_pos)
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self._reward()
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        # Move backward
        elif action == self.actions.backward:
            fwd_pos = np.array((self.agent_pos[0], self.agent_pos[1] + 1))
            fwd_cell = self.grid.get(*fwd_pos)
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self._reward()
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}
