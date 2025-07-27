from gymnasium.core import ActType, ObsType
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall, WorldObj
from minigrid.minigrid_env import MiniGridEnv
from minigrid.utils.rendering import (
    fill_coords,
    point_in_circle,
    point_in_rect,
)

import gymnasium as gym
import numpy as np

from typing import Any, SupportsFloat

from enum import IntEnum

from pathfinding import compute_pixel_dijkstra_path

TILE_PIXELS = 8


class DirectionlessGrid(Grid):
    def __init__(self, *args, **kwargs):
        self.unique_tiles = kwargs.pop("unique_tiles", None)
        self.padded_unique_tiles = kwargs.pop("padded_unique_tiles", None)
        self.tile_global_indices = kwargs.pop("tile_global_indices", None)
        self.show_grid_lines = kwargs.pop("show_grid_lines", False)
        self.show_walls_pov = kwargs.pop("show_walls_pov", False)
        self.generate_optimal_path = kwargs.pop("generate_optimal_path", True)
        self.show_optimal_path = kwargs.pop("show_optimal_path", True)
        self.pad_width = kwargs.pop("pad_width", None)
        self.seed = kwargs.pop("seed", None)
        self.path_pixels = kwargs.pop("path_pixels", set())  # Store pixel-level path coordinates
        self.tile_cache = {}
        super().__init__(*args, **kwargs)

    def render_tile(
        self,
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
                self.tile_global_indices[i, j][0],
                self.tile_global_indices[i, j][1],
                reveal_all,
                grid.seed,
                agent_dir,
                tuple(sorted(grid.path_pixels))  # Include path pixels in cache key
            )
        else:
            key: tuple[Any, ...] = (
                tile_size,
                self.tile_global_indices[i, j][0],
                self.tile_global_indices[i, j][1],
                reveal_all,
                grid.seed,
                tuple(sorted(grid.path_pixels))  # Include path pixels in cache key
            )

        key = obj.encode() + key if obj else key

        if key in self.tile_cache:
            return self.tile_cache[key]

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
                img = self.unique_tiles[i, j].copy()
            else:
                img = np.expand_dims(self.unique_tiles[i, j][:, :, 0].copy(), axis=-1)

        # Draw path pixels in blue (same size as wall pixels - individual pixels)
        tile_x_start = i * tile_size
        tile_y_start = j * tile_size
        
        for px in range(tile_size):
            for py in range(tile_size):
                global_px = tile_x_start + px
                global_py = tile_y_start + py
                
                if (global_px, global_py) in grid.path_pixels:
                    if reveal_all:
                        img[py, px] = [0, 0, 255]  # Blue in RGB
                    else:
                        img[py, px] = [128]  # Gray in grayscale (visible but distinct)

        # Draw the grid lines (top and left edges)
        if grid.show_grid_lines or reveal_all:
            line_thickness = 0.0625
            if reveal_all:
                fill_coords(img, point_in_rect(0, line_thickness, 0, 1), (100, 100, 100))
                fill_coords(img, point_in_rect(0, 1, 0, line_thickness), (100, 100, 100))
            else:
                fill_coords(img, point_in_rect(0, line_thickness, 0, 1), (100,))
                fill_coords(img, point_in_rect(0, 1, 0, line_thickness), (100,))

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
        self.tile_cache[key] = img

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
                    # TODO: Check if wall == pepper == black cells
                    isinstance(cell, Wall)
                    and not reveal_all
                    and not self.show_walls_pov
                ):
                    cell = None
                tile_img = self.render_tile(
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
            generate_optimal_path=self.generate_optimal_path,
            show_optimal_path=self.show_optimal_path,
            pad_width=self.pad_width,
            seed=self.seed,
            path_pixels=self.path_pixels,  # Pass path pixels to sliced grid
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


class Actions(IntEnum):
    left = 0
    forward = 1
    right = 2
    backward = 3


class SaltAndPepper(MiniGridEnv):
    """
    ## Description
    - `Salt` - Accessible white tiles
    - `Pepper` - Inaccessible black tiles

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
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    S: size of the map SxS.
    N: number of valid crossings across walls from the starting position
    to the goal

    - `SaltAndPepper` :
        - `MiniGrid-SaltAndPepper-v0-custom`

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
        generate_optimal_path = kwargs.pop("generate_optimal_path", True)
        show_optimal_path = kwargs.pop("show_optimal_path", True)
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
        self.generate_optimal_path = generate_optimal_path
        self.show_optimal_path = show_optimal_path
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
            generate_optimal_path=self.generate_optimal_path,
            show_optimal_path=self.show_optimal_path,
            unique_tiles=self.unique_tiles,
            padded_unique_tiles=self.padded_unique_tiles,
            pad_width=self.pad_width,
            tile_global_indices=self.tile_global_indices,
            seed=self.seed,
            path_pixels=set(),  # Initialize empty path pixels
        )

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        # Place the agent in the bottom middle
        self.agent_pos = np.array((width // 2, height - 2))
        self.agent_dir = 3

        self.goal_position = (2, 3)

        self.put_obj(Goal(), *self.goal_position)

        self.mission = "get to the green goal square"

        if self.generate_optimal_path:
            self.path = compute_pixel_dijkstra_path(self, TILE_PIXELS)

            # Update grid with path pixels
            if self.show_optimal_path and self.path:
                self.grid.path_pixels = set(self.path)

    def render_path_visualizations(self):
        """Render both full and partial views with path visualization"""
        full_img = self.get_full_render(highlight=False, tile_size=self.tile_size, reveal_all=True)
        partial_img = self.get_pov_render(tile_size=self.tile_size)

        return full_img, partial_img

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

        grid.path_pixels = self.grid.path_pixels

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

        # Move right
        elif action == self.actions.right:
            fwd_pos = np.array((self.agent_pos[0] + 1, self.agent_pos[1]))
            fwd_cell = self.grid.get(*fwd_pos)
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self._reward()

        # Move forward
        elif action == self.actions.forward:
            fwd_pos = np.array((self.agent_pos[0], self.agent_pos[1] - 1))
            fwd_cell = self.grid.get(*fwd_pos)
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self._reward()

        # Move backward
        elif action == self.actions.backward:
            fwd_pos = np.array((self.agent_pos[0], self.agent_pos[1] + 1))
            fwd_cell = self.grid.get(*fwd_pos)
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and fwd_cell.type == "goal":
                terminated = True
                reward = self._reward()

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