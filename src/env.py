from gymnasium.core import ActType, ObsType
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Wall, WorldObj
from minigrid.minigrid_env import MiniGridEnv
from minigrid.utils.rendering import (
    fill_coords,
    point_in_circle,
    point_in_rect,
    point_in_triangle,
)

import gymnasium as gym
import numpy as np

from typing import Any, SupportsFloat

from enum import IntEnum, StrEnum


TILE_PIXELS = 8


class PathDirection(IntEnum):
    UP = 1
    LEFT = 2
    RIGHT = 3
    DOWN = 4


# Local helper: Manhattan ball (a diamond / rotated square)
def point_in_diamond(cx: float, cy: float, r: float):
    def fn(x, y):
        return abs(x - cx) + abs(y - cy) <= r

    return fn


class Landmark(WorldObj):
    def __init__(
        self, shape="circle", render_color=(0, 100, 0), *, size=1, tile_offset=(0, 0)
    ):
        super().__init__("box", "green")
        self.shape = shape
        self.render_color = render_color  # RGB for full render
        self.size = size  # 1 = normal (1×1), 2 = big (2×2)
        self.tile_offset = tile_offset  # (ox, oy) ∈ {(0,0),(1,0),(0,1),(1,1)} for 2×2

    def can_overlap(self):
        # Landmarks are visual only and should not block movement.
        # Allow agents to overlap / walk through landmarks.
        return True

    def _apply_big_transform(self, fn):
        """
        Wrap a predicate fn(x, y) defined on [0,1]×[0,1] (global big shape space),
        so it can be applied to the local tile’s [0,1]×[0,1] by mapping:
            (x_local, y_local) -> ((ox + x_local)/2, (oy + y_local)/2)
        where (ox, oy) is this tile’s offset in the 2×2 big shape.
        """
        ox, oy = self.tile_offset

        def wrapped(x, y):
            return fn((ox + x) / 2.0, (oy + y) / 2.0)

        return wrapped

    def render(self, img):
        # grayscale POV vs full RGB
        if img.shape[-1] == 1:
            color = (160,)  # mid-gray so it stands out against black path
        else:
            color = self.render_color

        # define big-shape predicates in global [0,1] space (for the 2×2 as a whole)
        if self.shape == "circle":
            # same relative radius as before (0.4 of the *big* shape width)
            base_fn = point_in_circle(0.5, 0.5, 0.4)
        elif self.shape == "triangle":
            base_fn = point_in_triangle((0.5, 0.1), (0.1, 0.9), (0.9, 0.9))
        elif self.shape == "square":
            base_fn = point_in_rect(0.3, 0.9, 0.3, 0.9)
        elif self.shape == "diamond":
            base_fn = point_in_diamond(0.5, 0.5, 0.4)
        elif self.shape == "crescent":
            # Crescent made from outer circle minus an offset inner circle
            outer = point_in_circle(0.5, 0.5, 0.4)
            inner = point_in_circle(0.62, 0.48, 0.28)
            base_fn = lambda x, y: outer(x, y) and not inner(x, y)
        elif self.shape == "ring":
            # Ring: outer circle with a smaller concentric inner hole
            outer = point_in_circle(0.5, 0.5, 0.45)
            inner = point_in_circle(0.5, 0.5, 0.25)
            base_fn = lambda x, y: outer(x, y) and not inner(x, y)
        elif self.shape == "cross":
            # Cross (plus sign) composed of a vertical and a horizontal bar
            vbar = point_in_rect(0.4, 0.6, 0.15, 0.85)
            hbar = point_in_rect(0.15, 0.85, 0.4, 0.6)
            base_fn = lambda x, y: vbar(x, y) or hbar(x, y)
        elif self.shape == "rectangle":
            # Rectangle covers a wide central band but half the height
            base_fn = point_in_rect(0.15, 0.85, 0.375, 0.625)
        else:
            return

        # if size==2, render the appropriate quadrant by remapping coords;
        # if size==1, draw as usual.
        if self.size == 2:
            fn = self._apply_big_transform(base_fn)
            fill_coords(img, fn, color)
        else:
            fill_coords(img, base_fn, color)


class Path:
    def __init__(self, fragments):
        self.fragments = fragments

    def to_pixel_list(self):
        pixel_list = []
        path_widths = []
        for fragment in self.fragments:
            fragment_pixel_list, fragment_path_width = fragment.to_pixel_list()
            pixel_list.extend(fragment_pixel_list)
            path_widths.extend(fragment_path_width)
        return np.array(pixel_list), np.array(path_widths)


class PathFragment:
    def __init__(self, starting_point, direction, length):
        self.starting_point = starting_point
        self.direction = direction
        self.length = length

    def to_pixel_list(self):
        pixel_list = []
        if (
            self.direction == PathDirection.LEFT
            or self.direction == PathDirection.RIGHT
        ):
            path_width = [0, 1]
        else:
            path_width = [1, 0]
        path_width = [path_width for i in range(self.length)]
        for i in range(self.length):
            if self.direction == PathDirection.LEFT:
                pixel_list.append((self.starting_point[0] - i, self.starting_point[1]))
            elif self.direction == PathDirection.RIGHT:
                pixel_list.append((self.starting_point[0] + i, self.starting_point[1]))
            elif self.direction == PathDirection.UP:
                pixel_list.append((self.starting_point[0], self.starting_point[1] - i))
            elif self.direction == PathDirection.DOWN:
                pixel_list.append((self.starting_point[0], self.starting_point[1] + i))
        return pixel_list, path_width


shortest_path = Path(
    [
        PathFragment((60, 108), PathDirection.LEFT, 40),
        PathFragment((20, 108), PathDirection.UP, 81),
    ]
)

slightly_suboptimal_path = Path(
    [
        PathFragment((60, 108), PathDirection.RIGHT, 16),
        PathFragment((76, 108), PathDirection.UP, 8),
        PathFragment((76, 100), PathDirection.LEFT, 16),
        PathFragment((60, 100), PathDirection.UP, 72),
        PathFragment((60, 28), PathDirection.LEFT, 40),
    ]
)

moderately_suboptimal_path = Path(
    [
        PathFragment((60, 108), PathDirection.RIGHT, 32),
        PathFragment((92, 108), PathDirection.UP, 16),
        PathFragment((92, 92), PathDirection.LEFT, 32),
        PathFragment((60, 92), PathDirection.UP, 16),
        PathFragment((60, 76), PathDirection.LEFT, 16),
        PathFragment((44, 76), PathDirection.UP, 16),
        PathFragment((44, 60), PathDirection.LEFT, 16),
        PathFragment((28, 60), PathDirection.UP, 16),
        PathFragment((28, 44), PathDirection.LEFT, 8),
        PathFragment((20, 44), PathDirection.UP, 16),
    ]
)

misleading_path = Path(
    [
        PathFragment((60, 108), PathDirection.RIGHT, 24),
        PathFragment((84, 108), PathDirection.UP, 40),
        PathFragment((84, 68), PathDirection.LEFT, 56),
        PathFragment((28, 68), PathDirection.DOWN, 32),
        PathFragment((28, 100), PathDirection.LEFT, 8),
        PathFragment((20, 100), PathDirection.UP, 56),
        PathFragment((20, 44), PathDirection.RIGHT, 64),
        PathFragment((84, 44), PathDirection.UP, 32),
    ]
)


class DirectionlessGrid(Grid):
    def __init__(self, *args, **kwargs):
        self.seed = kwargs.pop("seed", 42)
        self.unique_tiles = kwargs.pop("unique_tiles", None)
        self.padded_unique_tiles = kwargs.pop("padded_unique_tiles", None)
        self.tile_global_indices = kwargs.pop("tile_global_indices", None)
        self.show_grid_lines = kwargs.pop("show_grid_lines", False)
        self.show_walls_pov = kwargs.pop("show_walls_pov", False)
        self.show_optimal_path = kwargs.pop("show_optimal_path", True)
        self.show_landmarks = kwargs.pop("show_landmarks", True)
        self.pad_width = kwargs.pop("pad_width", None)
        self.path_widths = kwargs.pop("path_widths", np.array([]))
        self.path_pixels = kwargs.pop("path_pixels", set())
        self.path_pixels_array = kwargs.pop("path_pixels_array", np.array([]))
        self.render_objects = kwargs.pop("render_objects", True)
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

        tile_x_start = i * tile_size
        tile_y_start = j * tile_size

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

        if grid.show_optimal_path or reveal_all:
            # Draw path pixels
            for px in range(tile_size):
                for py in range(tile_size):
                    global_px = tile_x_start + px
                    global_py = tile_y_start + py

                    if (global_px, global_py) in grid.path_pixels:
                        # Find the index in the first dimension where the inner array matches [global_px, global_py]
                        path_index = np.where(
                            np.all(
                                grid.path_pixels_array == [global_px, global_py], axis=1
                            )
                        )[0]
                        x_width, y_width = grid.path_widths[path_index][0]

                        if reveal_all:
                            img[
                                py - y_width : py + y_width + 1,
                                px - x_width : px + x_width + 1,
                            ] = [
                                0,
                                0,
                                255,
                            ]  # Blue in RGB
                        else:
                            img[
                                py - y_width : py + y_width + 1,
                                px - x_width : px + x_width + 1,
                                0,
                            ] = 0  # Black in grayscale

        # Draw the grid lines (top and left edges)
        if (grid.show_grid_lines or reveal_all) and grid.render_objects:
            line_thickness = 0.0625
            if reveal_all:
                fill_coords(
                    img, point_in_rect(0, line_thickness, 0, 1), (100, 100, 100)
                )
                fill_coords(
                    img, point_in_rect(0, 1, 0, line_thickness), (100, 100, 100)
                )
            else:
                fill_coords(img, point_in_rect(0, line_thickness, 0, 1), (100,))
                fill_coords(img, point_in_rect(0, 1, 0, line_thickness), (100,))

        if obj is not None and obj.type != "wall" and grid.render_objects:
            obj.render(img)

        # Overlay the agent on top
        if agent_dir is not None and reveal_all and grid.render_objects:
            tri_fn = point_in_circle(
                0.5,
                0.5,
                0.3,
            )
            # Rotate the agent based on its direction
            # tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * agent_dir)
            fill_coords(img, tri_fn, (255, 0, 0))

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

        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)
                agent_here = np.array_equal(agent_pos, (i, j))
                assert highlight_mask is not None

                if isinstance(cell, Goal) and cell.color == "green" and not reveal_all:
                    cell = None
                if isinstance(cell, Wall) and (
                    (not reveal_all and not self.show_walls_pov)
                    or not self.render_objects
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

        # Transform path pixels to local coordinates for the sliced grid
        local_path_pixels = set()
        local_path_widths = []
        local_path_pixels_array = []
        for index, (global_px, global_py) in enumerate(self.path_pixels_array):
            # Convert global pixel coordinates to tile coordinates
            tile_x = global_px // TILE_PIXELS
            tile_y = global_py // TILE_PIXELS

            x_width, y_width = self.path_widths[index]

            # Check if this tile is within the slice bounds
            if topX <= tile_x < topX + width and topY <= tile_y < topY + height:
                # Convert to local pixel coordinates within the slice
                local_tile_x = tile_x - topX
                local_tile_y = tile_y - topY
                local_px = local_tile_x * TILE_PIXELS + (global_px % TILE_PIXELS)
                local_py = local_tile_y * TILE_PIXELS + (global_py % TILE_PIXELS)
                local_path_pixels.add((local_px, local_py))
                local_path_widths.append((x_width, y_width))
                local_path_pixels_array.append((local_px, local_py))
        local_path_pixels_array = np.array(local_path_pixels_array)
        local_path_widths = np.array(local_path_widths)

        grid = DirectionlessGrid(
            width,
            height,
            tile_global_indices=relevant_unique_global_indices,
            unique_tiles=relevant_unique,
            padded_unique_tiles=self.padded_unique_tiles,
            show_grid_lines=self.show_grid_lines,
            show_walls_pov=self.show_walls_pov,
            show_optimal_path=self.show_optimal_path,
            show_landmarks=self.show_landmarks,
            pad_width=self.pad_width,
            seed=self.seed,
            path_pixels=local_path_pixels,
            path_widths=local_path_widths,
            path_pixels_array=local_path_pixels_array,
            render_objects=self.render_objects,
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


class PathMode(StrEnum):
    NONE = "NONE"
    SHORTEST_PATH = "SHORTEST_PATH"
    SLIGHTLY_SUBOPTIMAL_PATH = "SLIGHTLY_SUBOPTIMAL_PATH"
    SUBOPTIMAL_PATH = "SUBOPTIMAL_PATH"
    MISLEADING_PATH = "MISLEADING_PATH"
    VISITED_CELLS = "VISITED_CELLS"


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
            max_steps = 1000

        show_grid_lines = kwargs.pop("show_grid_lines", False)
        show_walls_pov = kwargs.pop("show_walls_pov", False)
        show_optimal_path = kwargs.pop("show_optimal_path", True)
        show_landmarks = kwargs.pop("show_landmarks", False)
        agent_view_size = kwargs.pop("agent_view_size", 5)
        path_mode = kwargs.pop("path_mode", "NONE")
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
        self.show_optimal_path = show_optimal_path
        self.show_landmarks = show_landmarks
        self.path_mode = PathMode[path_mode]
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
        (
            self.unique_tiles,
            self.padded_unique_tiles,
            self.pad_width,
            self.tile_global_indices,
        ) = self._gen_unique_tiles()
        self.cell_visitation = np.zeros(shape=(size, size, 4))
        self.cell_visitation_indices_stack = []
        self.max_visitation_count = 1000000
        self.segments_in_visitation_path = 25
        self.path = np.array([])
        self.path_widths = np.array([])

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.num_episodes += 1
        return super().reset(seed=seed, options=options)

    def _reward(self) -> float:
        return 1.0

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    def put_big_landmark(self, shape, rgb, x, y):
        """
        Place a single logical landmark that spans 2×2 tiles with top-left at (x, y).
        We actually place 4 Landmark instances, each told which quadrant (tile_offset)
        to render, so visually they compose one large shape.
        """
        for ox in (0, 1):
            for oy in (0, 1):
                self.put_obj(
                    Landmark(shape, rgb, size=2, tile_offset=(ox, oy)), x + ox, y + oy
                )

    def _gen_unique_tiles(self):
        # Use a fixed random state for generating tiles to ensure consistency
        tile_rng = np.random.RandomState(self.seed if self.seed is not None else 42)

        # subdivs = 3
        # Generate all unique random black and white pixels for all cells at once
        # Calculate section size (8x8 pixels)
        # section_size = 1
        num_sections_per_tile = TILE_PIXELS

        # Generate random black/white sections for all cells
        # Each section will be either all black (0,0,0) or all white (255,255,255)
        pad_width = int(np.ceil(self.agent_view_size / 2))
        # Generate the same pattern every time with the fixed seed
        section_colors = tile_rng.choice(
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

        self.grid = DirectionlessGrid(
            width,
            height,
            show_grid_lines=self.show_grid_lines,
            show_walls_pov=self.show_walls_pov,
            show_optimal_path=self.show_optimal_path,
            show_landmarks=self.show_landmarks,
            unique_tiles=self.unique_tiles,
            padded_unique_tiles=self.padded_unique_tiles,
            pad_width=self.pad_width,
            tile_global_indices=self.tile_global_indices,
            seed=self.seed,
            path_pixels=set(),
        )

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        # Place the agent in the bottom middle
        self.agent_pos = np.array((width // 2, height - 2))
        self.agent_dir = 3

        self.goal_position = (2, 3)

        self.put_obj(Goal(), *self.goal_position)

        # --- Place landmarks ---
        if self.show_landmarks:
            # Dark green circle (2×2)
            self.put_big_landmark("circle", (0, 100, 0), 4, 4)
            # Orange triangle (2×2)
            self.put_big_landmark("triangle", (255, 165, 0), 11, 11)
            # Purple square (2×2)
            self.put_big_landmark("square", (128, 0, 128), 10, 2)
            # Pink diamond (2×2)
            self.put_big_landmark("diamond", (255, 105, 180), 3, 11)
            # Teal crescent (2×2)
            self.put_big_landmark("crescent", (0, 128, 128), 7, 8)
            # Yellow cross (2×2)
            self.put_big_landmark("cross", (255, 255, 0), 9, 5)
            # Brown rectangle (2×2)
            self.put_big_landmark("rectangle", (165, 42, 42), 2, 1)
            # Dark blue ring (2×2)
            self.put_big_landmark("ring", (0, 0, 139), 2, 8)

        self.mission = "get to the green goal square"

        if self.path_mode == PathMode.SHORTEST_PATH:
            self.path, self.path_widths = shortest_path.to_pixel_list()
        elif self.path_mode == PathMode.SLIGHTLY_SUBOPTIMAL_PATH:
            self.path, self.path_widths = slightly_suboptimal_path.to_pixel_list()
        elif self.path_mode == PathMode.SUBOPTIMAL_PATH:
            self.path, self.path_widths = moderately_suboptimal_path.to_pixel_list()
        elif self.path_mode == PathMode.MISLEADING_PATH:
            self.path, self.path_widths = misleading_path.to_pixel_list()

        # Update grid with path pixels
        if self.show_optimal_path and not self.path_mode == PathMode.VISITED_CELLS:
            self.grid.path_pixels = set(map(tuple, self.path.tolist()))
            self.grid.path_pixels_array = self.path
            self.grid.path_widths = self.path_widths

    def render_path_visualizations(self):
        """Render both full and partial views with path visualization"""
        full_img = self.get_full_render(
            highlight=False, tile_size=self.tile_size, reveal_all=True
        )
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
        grid, vis_mask, agent_pos = self.gen_obs_grid()

        img = grid.render(
            tile_size,
            agent_pos=agent_pos,
            agent_dir=3,
            highlight_mask=vis_mask,
        )

        return img

    def generate_visitation_path(self):
        if self.step_count % 1000 == 0:
            segment_count = np.minimum(
                self.segments_in_visitation_path, np.sum(self.cell_visitation > 0)
            )
            flat = self.cell_visitation.flatten()
            order = np.lexsort(
                (np.arange(flat.size), -flat)
            )  # primary: -flat, secondary: index
            top_indices = order[:segment_count]
            # Vectorized construction of per-tile vertical path pixels

            r, c, action = np.unravel_index(top_indices, self.cell_visitation.shape)

            left_x_pix = (
                (r * TILE_PIXELS)[:, None]
                + np.arange(-TILE_PIXELS // 2, TILE_PIXELS // 2 + 1)
            ).ravel()
            left_y_pix = (c * TILE_PIXELS).repeat(TILE_PIXELS + 1) + TILE_PIXELS // 2

            up_x_pix = (r * TILE_PIXELS).repeat(TILE_PIXELS + 1) + TILE_PIXELS // 2
            up_y_pix = (
                (c * TILE_PIXELS)[:, None]
                + np.arange(-TILE_PIXELS // 2, TILE_PIXELS // 2 + 1)
            ).ravel()

            right_x_pix = (
                (r * TILE_PIXELS)[:, None]
                + np.arange(TILE_PIXELS + 1)
                + TILE_PIXELS // 2
                - 1
            ).ravel()
            right_y_pix = (c * TILE_PIXELS).repeat(TILE_PIXELS + 1) + TILE_PIXELS // 2

            down_x_pix = (r * TILE_PIXELS).repeat(TILE_PIXELS + 1) + TILE_PIXELS // 2
            down_y_pix = (
                (c * TILE_PIXELS)[:, None]
                + np.arange(TILE_PIXELS + 1)
                + TILE_PIXELS // 2
                - 1
            ).ravel()

            action = action.repeat(TILE_PIXELS + 1)

            x_pix = np.choose(
                action,
                [left_x_pix, up_x_pix, right_x_pix, down_x_pix],
            )
            y_pix = np.choose(action, [left_y_pix, up_y_pix, right_y_pix, down_y_pix])

            self.path = np.column_stack((x_pix, y_pix))

            path_list = list(map(tuple, self.path.tolist()))
            self.path_widths = np.where(
                np.logical_or(action[:, None] == 1, action[:, None] == 3),
                np.tile([1, 0], (len(self.path), 1)),
                np.tile([0, 1], (len(self.path), 1)),
            )

            self.grid.path_pixels = set(path_list)
            self.grid.path_pixels_array = self.path
            self.grid.path_widths = self.path_widths

    def gen_obs_grid(self, agent_view_size=None):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        if agent_view_size is None, self.agent_view_size is used
        """

        topX, topY, botX, botY = self.get_view_exts(agent_view_size)

        agent_view_size = agent_view_size or self.agent_view_size

        if self.path_mode == PathMode.VISITED_CELLS:
            self.generate_visitation_path()

        grid = self.grid.slice(topX, topY, agent_view_size, agent_view_size)

        # The path pixels are already properly transformed to local coordinates in the slice method
        # No additional processing needed here

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(
                agent_pos=(agent_view_size // 2, agent_view_size - 1)
            )
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)

        agent_pos = grid.width // 2, grid.height // 2

        return grid, vis_mask, agent_pos

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """
        grid, vis_mask, agent_pos = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = grid.render(
            TILE_PIXELS,
            agent_pos,
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
        _, vis_mask, _ = self.gen_obs_grid()

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

    def update_visitation_count(self, pos, action):
        self.cell_visitation[pos[0], pos[1], action] += 1
        self.cell_visitation_indices_stack.append([pos[0], pos[1], action])
        if len(self.cell_visitation_indices_stack) > self.max_visitation_count:
            index = self.cell_visitation_indices_stack.pop(0)
            self.cell_visitation[index[0], index[1], index[2]] -= 1

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Map actions to movement deltas
        action_deltas = {
            self.actions.left: (-1, 0),
            self.actions.right: (1, 0),
            self.actions.forward: (0, -1),
            self.actions.backward: (0, 1),
        }

        try:
            if action.item() in action_deltas:
                dx, dy = action_deltas[action.item()]
                fwd_pos = np.array((self.agent_pos[0] + dx, self.agent_pos[1] + dy))
                fwd_cell = self.grid.get(*fwd_pos)
                if fwd_cell is None or fwd_cell.can_overlap():
                    self.update_visitation_count(
                        self.agent_pos,
                        action.item(),
                    )
                    self.agent_pos = tuple(fwd_pos)

                if fwd_cell is not None and fwd_cell.type == "goal":
                    terminated = True
                    reward = self._reward()

            # Done action (not used by default)
            elif action == self.actions.done:
                pass

            else:
                raise ValueError(f"Unknown action: {action}")
        except:
            import pdb

            pdb.set_trace()

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()
        # plt.imshow(obs["image"], cmap="gray", vmin=0, vmax=255)
        # plt.savefig("step.png")
        # plt.close()

        return obs, reward, terminated, truncated, {}
