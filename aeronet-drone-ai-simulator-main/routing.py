from __future__ import annotations

import heapq
from typing import Dict, List, Tuple, Optional
from grid_model import Coord, get_neighbors, manhattan
from config import ZONE_COST


def reconstruct(came_from: Dict[Coord, Coord], current: Coord) -> List[Coord]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return list(reversed(path))


def cell_cost(grid, coord: Coord, dense_penalty: bool = True) -> float:
    cell = grid[coord[0]][coord[1]]
    base = ZONE_COST.get(cell.zone, 1.0)
    if dense_penalty and cell.density_level == "High":
        base += 2.5
    elif dense_penalty and cell.density_level == "Medium":
        base += 1.2
    return base


def shortest_path(
    start: Coord,
    goal: Coord,
    grid,
    algorithm: str = "A*",
    dense_penalty: bool = True,
) -> Tuple[Optional[List[Coord]], float, str]:
    """
    Supports:
    - A*
    - Dijkstra
    - Weighted A*
    """
    if grid[start[0]][start[1]].no_fly:
        return None, float("inf"), "Start cell is no-fly."
    if grid[goal[0]][goal[1]].no_fly:
        return None, float("inf"), "Goal cell is no-fly."

    def heuristic(a, b):
        if algorithm == "Dijkstra":
            return 0
        if algorithm == "Weighted A*":
            return 1.5 * manhattan(a, b)
        return manhattan(a, b)

    open_heap = [(heuristic(start, goal), start)]
    came_from = {}
    g = {start: 0.0}
    visited = set()
    expanded_nodes = 0

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            return reconstruct(came_from, current), round(g[current], 2), f"{algorithm} route found. Expanded nodes={expanded_nodes}"
            

        if current in visited:
            continue
        visited.add(current)
        expanded_nodes += 1

        for nr, nc in get_neighbors(current[0], current[1], len(grid)):
            if grid[nr][nc].no_fly:
                continue

            nxt = (nr, nc)
            tentative = g[current] + cell_cost(grid, nxt, dense_penalty=dense_penalty)

            if tentative < g.get(nxt, float("inf")):
                came_from[nxt] = current
                g[nxt] = tentative
                f = tentative + heuristic(nxt, goal)
                heapq.heappush(open_heap, (f, nxt))

    return None, float("inf"), "No safe path found."


def full_delivery_route(hub: Coord, pickup: Coord, dropoff: Coord, grid, algorithm="A*"):
    route = []
    total = 0.0
    messages = []
    total_expanded_nodes = 0

    for s, g in [(hub, pickup), (pickup, dropoff), (dropoff, hub)]:
        path, cost, msg = shortest_path(s, g, grid, algorithm=algorithm)

        if path is None:
            return None, float("inf"), msg

        route.extend(path if not route else path[1:])
        total += cost
        messages.append(msg)

        if "Expanded nodes=" in msg:
            expanded = int(msg.split("Expanded nodes=")[-1].split()[0])
            total_expanded_nodes += expanded

    final_msg = (
        f"Full delivery route planned using {algorithm}. "
        f"Total expanded nodes={total_expanded_nodes}. "
        + " | ".join(messages)
    )

    return route, round(total, 2), final_msg