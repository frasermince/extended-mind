"""
Pixel-level Dijkstra pathfinding for the SaltAndPepper environment.

Each tile is treated as a grid of pixels (e.g., 8x8). This module computes the
shortest path between the agent and the goal using Dijkstra's algorithm, operating
at pixel-level granularity to support fine-grained traversability decisions.
"""

import heapq
from minigrid.core.world_object import Wall
from typing import List, Tuple

PIXEL_PASSABLE_THRESHOLD = 127


def compute_pixel_dijkstra_path(env, tile_pixels) -> List[Tuple[int, int]]:
    """
    Compute Dijkstra's path at the pixel level (8x8 pixels per tile)
    """
    # Convert tile coordinates to pixel coordinates
    start_tile = tuple(env.agent_pos)
    goal_tile = env.goal_position
    
    # Agent starts at center of its tile
    start_px = (start_tile[0] * tile_pixels + tile_pixels // 2, 
                start_tile[1] * tile_pixels + tile_pixels // 2)
    goal_px = (goal_tile[0] * tile_pixels + tile_pixels // 2, 
                goal_tile[1] * tile_pixels + tile_pixels // 2)
    
    width_px = env.grid.width * tile_pixels
    height_px = env.grid.height * tile_pixels

    visited = set()
    came_from = {}
    cost_so_far = {start_px: 0}

    heap = [(0, start_px)]
    
    while heap:
        current_cost, current = heapq.heappop(heap)

        if current == goal_px:
            break

        if current in visited:
            continue
        visited.add(current)

        px, py = current
        # 4-directional movement at pixel level
        neighbors = [
            (px-1, py), (px+1, py),
            (px, py-1), (px, py+1)
        ]
        
        for npx, npy in neighbors:
            if 0 <= npx < width_px and 0 <= npy < height_px:
                # Check if this pixel is passable
                if is_pixel_passable(env, npx, npy, tile_pixels):
                    new_cost = current_cost + 1
                    if (npx, npy) not in cost_so_far or new_cost < cost_so_far[(npx, npy)]:
                        cost_so_far[(npx, npy)] = new_cost
                        heapq.heappush(heap, (new_cost, (npx, npy)))
                        came_from[(npx, npy)] = current

    # Reconstruct path
    if goal_px not in came_from and goal_px != start_px:
        return []  # No path found
        
    current = goal_px
    path = [current]
    while current != start_px:
        current = came_from.get(current)
        if current is None:
            return []  # No path
        path.append(current)

    return path[::-1]  # Start to goal


def is_pixel_passable(env, px, py, tile_pixels) -> bool:
    """
    Determines whether a pixel is passable (not a wall or pepper).

    Args:
        env: SaltAndPepper environment instance.
        px (int): X-coordinate of the pixel.
        py (int): Y-coordinate of the pixel.
        tile_pixels (int): Number of pixels per tile.

    Returns:
        bool: True if pixel is traversable, False otherwise.
    """
    # Convert pixel coordinates to tile coordinates
    tile_x = px // tile_pixels
    tile_y = py // tile_pixels
    
    # Check bounds
    if tile_x < 0 or tile_x >= env.grid.width or tile_y < 0 or tile_y >= env.grid.height:
        return False
        
    # Check if tile contains a wall
    cell = env.grid.get(tile_x, tile_y)
    if isinstance(cell, Wall):
        return False
        
    # Check if the specific pixel within the tile is passable (white/salt)
    pixel_in_tile_x = px % tile_pixels
    pixel_in_tile_y = py % tile_pixels
    
    # Get the pixel value from unique_tiles
    pixel_value = env.unique_tiles[tile_x, tile_y, pixel_in_tile_y, pixel_in_tile_x, 0]
    
    # Return True if pixel is white (salt), False if black (pepper)
    return pixel_value > PIXEL_PASSABLE_THRESHOLD  # Threshold for white vs black