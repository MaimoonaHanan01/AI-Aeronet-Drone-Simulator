from dataclasses import dataclass
from typing import List
from grid_model import Cell, get_neighbors, manhattan, hubs, charging_pads, medical_pickups, find_cells
@dataclass
class RuleResult:
    rule_id: str
    passed: bool
    messages: List[str]


def check_industrial_safety(grid: List[List[Cell]]) -> RuleResult:
    errors = []
    for row in grid:
        for cell in row:
            if cell.zone == "Industrial":
                for nr, nc in get_neighbors(cell.row, cell.col, len(grid)):
                    nz = grid[nr][nc].zone
                    if nz in {"School", "Hospital"}:
                        errors.append(f"Industrial cell {cell.coord} is adjacent to {nz} cell {(nr,nc)}.")
    return RuleResult("R1", not errors, errors or ["Industrial safety passed."])


def check_residential_coverage(grid: List[List[Cell]]) -> RuleResult:
    errors = []
    hs = hubs(grid)
    for cell in find_cells(grid, lambda x: x.zone == "Residential"):
        nearest = min(manhattan(cell.coord, h) for h in hs)
        if nearest > 3:
            errors.append(f"Residential cell {cell.coord} is {nearest} cells away from nearest hub.")
    return RuleResult("R2", not errors, errors or ["Residential hub coverage passed."])


def check_hub_charging(grid: List[List[Cell]]) -> RuleResult:
    errors = []
    cps = charging_pads(grid)
    for h in hubs(grid):
        nearest = min(manhattan(h, c) for c in cps)
        if nearest > 2:
            errors.append(f"Hub {h} has nearest charging pad {nearest} cells away.")
    return RuleResult("R3", not errors, errors or ["Hub charging access passed."])


def check_medical_access(grid: List[List[Cell]]) -> RuleResult:
    hospitals = find_cells(grid, lambda x: x.zone == "Hospital")
    pickups = medical_pickups(grid)
    for h in hospitals:
        for p in pickups:
            if manhattan(h.coord, p) <= 1:
                return RuleResult("R4", True, [f"Hospital {h.coord} has medical pickup nearby at {p}."])
    return RuleResult("R4", False, ["No hospital has medical pickup within 1 cell."])


def validate_layout(grid):
    return [
        check_industrial_safety(grid),
        check_residential_coverage(grid),
        check_hub_charging(grid),
        check_medical_access(grid),
    ]