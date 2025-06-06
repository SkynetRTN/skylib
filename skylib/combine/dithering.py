"""Implementation of the various dithering patterns."""

import numpy as np


__all__ = [
    "generate_grid_pattern", "generate_snake_pattern", "generate_spiral_pattern", "generate_jitter_pattern",
    "generate_cross_pattern",
]

def generate_grid_pattern(step_size: float, n_steps: int) -> list[tuple[float, float]]:
    """
    Generate a centered grid dithering pattern using row-wise ordering. The grid is centered at (0, 0), starting from
    bottom-left, moving left-to-right.

    :param step_size: The size of each step in the pattern.
    :param n_steps: The total number of steps to generate.

    :return: A list of (x, y) offsets for the dithering pattern.
    """
    if n_steps < 1:
        return []

    half_side = int(np.ceil(np.sqrt(n_steps)))//2

    grid = []
    for y in range(-half_side, half_side + 1):
        for x in range(-half_side, half_side + 1):
            grid.append((x*step_size, y*step_size))
            if len(grid) == n_steps:
                return grid
    return grid[:n_steps]


def generate_snake_pattern(step_size: float, n_steps: int) -> list[tuple[float, float]]:
    """
    Generate a centered grid dithering pattern using row-wise snake ordering. The grid is centered at (0, 0), starting
    from bottom-left, moving left-to-right in even-numbered rows (0-based), and right-to-left in odd-numbered rows.

    :param step_size: The size of each step in the pattern.
    :param n_steps: The total number of steps to generate.

    :return: A list of (x, y) offsets for the dithering pattern.
    """
    if n_steps < 1:
        return []

    half_side = int(np.ceil(np.sqrt(n_steps)))//2

    grid = []
    for row_index, y in enumerate(range(-half_side, half_side + 1)):
        x_coords = range(-half_side, half_side + 1)
        if row_index % 2 == 1:
            x_coords = reversed(x_coords)
        for x in x_coords:
            grid.append((x*step_size, y*step_size))
            if len(grid) == n_steps:
                return grid
    return grid


def generate_spiral_pattern(step_size: float, n_steps: int) -> list[tuple[float, float]]:
    """
    Generate a square spiral dithering pattern starting at (0, 0).

    :param step_size: The size of each step in the pattern.
    :param n_steps: The total number of steps to generate.

    :return: A list of (x, y) offsets for the dithering pattern.
    """
    offsets = [(0, 0)]
    dx, dy = 0, 0
    direction = [(1, 0), (0, -1), (-1, 0), (0, 1)]  # right, down, left, up
    d = 0
    steps_per_side = 1

    while len(offsets) < n_steps:
        for _ in range(2):  # Two directions per spiral level
            for _ in range(steps_per_side):
                if len(offsets) >= n_steps:
                    break
                curdir = direction[d % 4]
                dx += curdir[0]*step_size
                dy += curdir[1]*step_size
                offsets.append((dx, dy))
                if len(offsets) >= n_steps:
                    return offsets
            d += 1
        steps_per_side += 1
    return offsets


def generate_jitter_pattern(step_size: float, n_steps: int) -> list[tuple[float, float]]:
    """
    Generate a random (jitter) dithering pattern. Offsets are within a circle of radius = step_size.

    :param step_size: The size of each step in the pattern.
    :param n_steps: The total number of steps to generate.

    :return: A list of (x, y) offsets for the dithering pattern.
    """
    offsets = []
    for _ in range(n_steps):
        r = np.random.uniform(0, step_size)
        theta = np.random.uniform(0, 2*np.pi)
        offsets.append((r*np.cos(theta), r*np.sin(theta)))
    return offsets


def generate_cross_pattern(step_size: float, n_steps: int) -> list[tuple[float, float]]:
    """
    Generate a cross dithering pattern for arbitrary n_steps. Distributes steps symmetrically along 4 directions
    centered at (0, 0). The order is left, right, down, up, and then increases the step size for each direction.

    :param step_size: The size of each step in the pattern.
    :param n_steps: The total number of steps to generate.

    :return: A list of (x, y) offsets for the dithering pattern.
    """
    if n_steps < 1:
        return []

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    step = 1

    offsets = [(0, 0)]
    while len(offsets) < n_steps:
        for dx, dy in directions:
            offsets.append((dx*step*step_size, dy*step*step_size))
            if len(offsets) == n_steps:
                return offsets
        step += 1
    return offsets
