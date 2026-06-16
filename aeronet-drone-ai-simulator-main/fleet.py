from dataclasses import dataclass
from typing import Tuple, List
from grid_model import hubs


@dataclass
class Drone:
    drone_id: str
    drone_type: str
    hub: Tuple[int, int]
    position: Tuple[float, float, float]
    battery: float
    payload_kg: float
    max_payload: float
    speed: float
    status: str = "idle"
    current_route: list = None


def select_fleet(grid, budget=12000):
    """
    Advanced but explainable heuristic:
    - demand controls fleet size
    - hospitals require at least one heavy drone
    """
    total_demand = sum(c.demand for row in grid for c in row)
    light_cost, heavy_cost = 1000, 1800

    heavy = 2 if total_demand > 800 else 1
    light = max(3, min(8, int(total_demand // 160)))

    while light * light_cost + heavy * heavy_cost > budget and light > 1:
        light -= 1
    while light * light_cost + heavy * heavy_cost > budget and heavy > 1:
        heavy -= 1

    return {"light": light, "heavy": heavy, "cost": light * light_cost + heavy * heavy_cost, "demand": round(total_demand, 2)}


def build_drones(grid, fleet_selection) -> List[Drone]:
    hs = hubs(grid)
    drones = []
    count = 1

    for i in range(fleet_selection["light"]):
        hub = hs[i % len(hs)]
        drones.append(Drone(
            drone_id=f"D{count}",
            drone_type="Light",
            hub=hub,
            position=(hub[1], hub[0], 1.5),
            battery=100,
            payload_kg=1.2,
            max_payload=2,
            speed=1.0,
            current_route=[]
        ))
        count += 1

    for i in range(fleet_selection["heavy"]):
        hub = hs[(i + fleet_selection["light"]) % len(hs)]
        drones.append(Drone(
            drone_id=f"D{count}",
            drone_type="Heavy",
            hub=hub,
            position=(hub[1], hub[0], 1.8),
            battery=100,
            payload_kg=3.5,
            max_payload=5,
            speed=0.85,
            current_route=[]
        ))
        count += 1

    return drones
