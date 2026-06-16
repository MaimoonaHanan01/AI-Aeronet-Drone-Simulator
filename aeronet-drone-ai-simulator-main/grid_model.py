from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Callable
from matplotlib.pyplot import grid
import pandas as pd
import numpy as np
from pathlib import Path

Coord = Tuple[int, int]


@dataclass
class Cell:
    row: int
    col: int
    zone: str
    density: float
    density_level: str
    demand: float
    is_hub: bool = False
    is_charging: bool = False
    is_medical_pickup: bool = False
    no_fly: bool = False

    @property
    def coord(self) -> Coord:
        return (self.row, self.col)


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_neighbors(row: int, col: int, size: int = 10) -> List[Coord]:
    candidates = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
    return [(r, c) for r, c in candidates if 0 <= r < size and 0 <= c < size]


def load_density_values(path: str = "data/raw/uscitypopdensity.csv", n: int = 100) -> np.ndarray:
    """Load public city density values. Falls back to synthetic values if file is missing."""
    p = Path(path)
    if p.exists():
        df = pd.read_csv(p)
        col = "Population Density (Persons/Square Mile)"
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").dropna().values
            if len(values) >= 10:
                return values[:n]
    rng = np.random.default_rng(42)
    return rng.integers(300, 30000, size=n)


def density_level(value: float, q1: float, q2: float) -> str:
    if value <= q1:
        return "Low"
    if value <= q2:
        return "Medium"
    return "High"


def create_advanced_grid(seed: int = 42) -> List[List[Cell]]:
    """
    Creates a 10x10 city grid using population density values.
    The zone layout is manually controlled so CSP validation remains explainable.
    """
    rng = np.random.default_rng(seed)
    density_values = load_density_values(n=100).copy()
    rng.shuffle(density_values)
    q1, q2 = np.quantile(density_values, [0.33, 0.66])

    zone_layout = [
        ["Residential","Residential","Commercial","Commercial","Open Field","Open Field","School","School","Open Field","Open Field"],
        ["Residential","Residential","Commercial","Open Field","Open Field","Open Field","School","Open Field","Open Field","Open Field"],
        ["Residential","Commercial","Commercial","Open Field","Open Field","Open Field","Open Field","Open Field","Industrial","Industrial"],
        ["Open Field","Open Field","Open Field","Open Field","Open Field","Open Field","Open Field","Open Field","Industrial","Industrial"],
        ["Hospital","Hospital","Open Field","Open Field","Residential","Commercial","Commercial","Open Field","Open Field","Open Field"],
        ["Hospital","Open Field","Open Field","Residential","Residential","Commercial","Open Field","Commercial","Open Field","Open Field"],
        ["Open Field","Open Field","Open Field","Open Field","Residential","Residential","Commercial","Commercial","Open Field","School"],
        ["Open Field","Open Field","Open Field","Open Field","Open Field","Residential","Residential","Open Field","Open Field","School"],
        ["Industrial","Industrial","Open Field","Commercial","Commercial","Open Field","Open Field","Open Field","Open Field","Open Field"],
        ["Industrial","Industrial","Open Field","Commercial","Commercial","Open Field","Open Field","Hospital","Hospital","Open Field"],
    ]

    grid = []
    idx = 0
    for r in range(10):
        row = []
        for c in range(10):
            zone = zone_layout[r][c]
            den = float(density_values[idx])
            lvl = density_level(den, q1, q2)
            zone_weight = {
                "Residential": 1.00,
                "Commercial": 1.25,
                "Hospital": 1.05,
                "School": 0.75,
                "Industrial": 0.55,
                "Open Field": 0.15,
            }[zone]
            demand = round((den / 1000.0) * zone_weight, 2)
            row.append(Cell(r, c, zone, den, lvl, demand))
            idx += 1
        grid.append(row)

# Hubs and charging pads
# Hubs and charging pads
    grid[1][1].is_hub = True      # covers top-left residential cells
    grid[5][6].is_hub = True      # covers middle/commercial zone
    grid[4][4].is_hub = True      # covers residential cell (4,3)

# Charging pads within 2 Manhattan cells of every hub
    grid[1][2].is_charging = True
    grid[5][7].is_charging = True
    grid[4][5].is_charging = True
    # Medical pickup
    grid[5][1].is_medical_pickup = True

    # No fly
    grid[3][6].no_fly = True
    grid[4][6].no_fly = True
    grid[5][5].no_fly = True
    grid[6][5].no_fly = True
    grid[2][7].no_fly = True

    return grid


def find_cells(grid: List[List[Cell]], predicate: Callable[[Cell], bool]) -> List[Cell]:
    return [cell for row in grid for cell in row if predicate(cell)]


def hubs(grid: List[List[Cell]]) -> List[Coord]:
    return [c.coord for c in find_cells(grid, lambda x: x.is_hub)]


def charging_pads(grid: List[List[Cell]]) -> List[Coord]:
    return [c.coord for c in find_cells(grid, lambda x: x.is_charging)]


def medical_pickups(grid: List[List[Cell]]) -> List[Coord]:
    return [c.coord for c in find_cells(grid, lambda x: x.is_medical_pickup)]
