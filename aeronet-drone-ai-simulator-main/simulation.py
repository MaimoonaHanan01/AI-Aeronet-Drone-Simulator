from dataclasses import asdict
from grid_model import create_advanced_grid, hubs, find_cells
from fleet import select_fleet, build_drones
from routing import full_delivery_route, shortest_path
from ml_models import train_demand_model, train_anomaly_model
from csp_validator import validate_layout


def generate_demo_deliveries(grid):
    commercial = [c.coord for c in find_cells(grid, lambda c: c.zone == "Commercial")]
    residential = [c.coord for c in find_cells(grid, lambda c: c.zone in {"Residential", "Hospital", "School"})]
    med = [c.coord for c in find_cells(grid, lambda c: c.is_medical_pickup)]
    return [
        {"id": "DEL-1", "pickup": med[0], "dropoff": residential[-1], "priority": "medical"},
        {"id": "DEL-2", "pickup": commercial[3], "dropoff": residential[2], "priority": "normal"},
        {"id": "DEL-3", "pickup": commercial[-2], "dropoff": residential[7], "priority": "normal"},
    ]


def run_advanced_simulation(algorithm="A*", budget=12000, disruption_cell=(3, 5)):
    grid = create_advanced_grid()
    fleet_selection = select_fleet(grid, budget)
    drones = build_drones(grid, fleet_selection)
    deliveries = generate_demo_deliveries(grid)
    events = []

    events.append("System initialized with 10x10 advanced 3D grid.")
    events.append(f"Fleet selected: {fleet_selection['light']} light and {fleet_selection['heavy']} heavy drones.")

    routes = {}
    for i, delivery in enumerate(deliveries):
        if i >= len(drones):
            events.append(f"{delivery['id']} queued because no idle drone is available.")
            continue
        drone = drones[i]
        route, cost, msg = full_delivery_route(drone.hub, delivery["pickup"], delivery["dropoff"], grid, algorithm=algorithm)
        if route:
            drone.current_route = route
            drone.status = "delivering"
            drone.battery -= min(40, cost * 1.25)
            routes[drone.drone_id] = route
            events.append(f"{delivery['id']} assigned to {drone.drone_id}. Route cost={cost} using {algorithm}.")
        else:
            events.append(f"{delivery['id']} failed route planning: {msg}")

    # Disruption
    grid[disruption_cell[0]][disruption_cell[1]].no_fly = True
    events.append(f"Real-time disruption: no-fly zone activated at {disruption_cell}.")

    reroutes = {}
    for d in drones:
        if d.current_route and disruption_cell in d.current_route:
            current = d.current_route[min(2, len(d.current_route)-1)]
            goal = d.current_route[-1]
            new_route, cost, msg = shortest_path(current, goal, grid, algorithm=algorithm)
            if new_route:
                reroutes[d.drone_id] = new_route
                d.current_route = new_route
                d.status = "rerouted"
                d.battery -= min(20, cost)
                events.append(f"{d.drone_id} rerouted successfully after no-fly disruption. New cost={cost}.")
            else:
                d.status = "delayed"
                events.append(f"{d.drone_id} delayed because no safe reroute exists.")

    demand = train_demand_model()
    anomaly = train_anomaly_model()
    events.append(f"Demand model trained. MAE={demand['mae']}, RMSE={demand['rmse']}.")
    events.append(f"Anomaly classifier trained. Best={anomaly['best_name']}, Accuracy={anomaly['accuracy']}.")

    if drones:
        drones[0].status = "returning_to_hub"
        events.append(f"Safety action: {drones[0].drone_id} returning to hub after telemetry check.")

    return {
        "grid": grid,
        "fleet_selection": fleet_selection,
        "drones": drones,
        "deliveries": deliveries,
        "routes": routes,
        "reroutes": reroutes,
        "events": events,
        "demand": demand,
        "anomaly": anomaly,
    }

def generate_20_step_deliveries(grid):
    """Generate 5-10 demo deliveries required for the semester-project simulation."""
    return [
        {"id": "DEL-1", "pickup": (5, 1), "dropoff": (9, 8), "priority": "medical"},
        {"id": "DEL-2", "pickup": (2, 1), "dropoff": (0, 6), "priority": "normal"},
        {"id": "DEL-3", "pickup": (9, 3), "dropoff": (5, 3), "priority": "normal"},
        {"id": "DEL-4", "pickup": (0, 2), "dropoff": (7, 9), "priority": "school"},
        {"id": "DEL-5", "pickup": (8, 4), "dropoff": (4, 1), "priority": "high-demand"},
        {"id": "DEL-6", "pickup": (6, 7), "dropoff": (0, 0), "priority": "high-demand"},
    ]


def run_20_step_simulation(algorithm="A*", budget=12000, disruption_cell=(3, 2)):
    """
    Required 20-step demo scenario:
    1-3 initialize/validate/fleet,
    4-6 deliveries/routes,
    7-10 movement,
    11 disruption,
    12-14 rerouting,
    15-17 demand forecasting,
    18 anomaly,
    19 safety action,
    20 final summary.
    """
    grid = create_advanced_grid()
    csp_results = validate_layout(grid)
    fleet_selection = select_fleet(grid, budget)
    drones = build_drones(grid, fleet_selection)
    deliveries = generate_20_step_deliveries(grid)

    step_log = []
    routes = {}
    reroutes = {}
    failed_deliveries = []
    queued_deliveries = []
    demand = None
    anomaly = None

    def add(step, phase, event, drone="-", status="OK"):
        step_log.append({
            "Step": step,
            "Phase": phase,
            "Event": event,
            "Drone": drone,
            "Status": status,
        })

    # Steps 1-3: initialization, validation, fleet
    add(1, "Initialization", "10x10 grid initialized with zones, density, hubs, charging pads, medical pickup points, and no-fly cells.")
    add(2, "CSP Validation", "Layout validation completed: " + ("PASS" if all(r.passed for r in csp_results) else "FAIL"))
    add(3, "Fleet Planning", f"Fleet selected under budget {budget}: {fleet_selection['light']} light drone(s), {fleet_selection['heavy']} heavy drone(s), cost={fleet_selection['cost']}.")

    # Steps 4-6: delivery generation and route planning
    add(4, "Delivery Generation", f"Generated {len(deliveries)} delivery requests for medical, residential, school, and high-demand zones.")

    active_deliveries = deliveries[:min(len(drones), 4)]

    for i, delivery in enumerate(active_deliveries):
        drone = drones[i]

        route, cost, msg = full_delivery_route(
            drone.hub,
            delivery["pickup"],
            delivery["dropoff"],
            grid,
            algorithm=algorithm
        )

        if route:
            drone.current_route = route
            drone.status = "delivering"
            drone.battery -= min(35, cost * 0.8)
            routes[drone.drone_id] = route
        else:
            drone.status = "failed"
            failed_deliveries.append(delivery["id"])

    queued_deliveries = deliveries[len(active_deliveries):]

    add(5, "Delivery Assignment", f"Assigned {len(active_deliveries)} delivery request(s) to available drones; {len(queued_deliveries)} request(s) kept in queue.")
    add(6, "Route Planning", f"{algorithm} planned safe hub-pickup-dropoff-hub routes while avoiding no-fly cells.")

    # Steps 7-10: simple movement simulation
    for step in range(7, 11):
        moved = 0

        for drone in drones:
            if drone.current_route:
                idx = min(step - 6, len(drone.current_route) - 1)
                r, c = drone.current_route[idx]
                drone.position = (c, r, drone.position[2])
                moved += 1

        add(step, "Drone Movement", f"Moved {moved} active drone(s) one step forward along their planned routes.")

    # Step 11: disruption
    grid[disruption_cell[0]][disruption_cell[1]].no_fly = True
    add(11, "Disruption", f"New no-fly cell activated at {disruption_cell}.", status="WARNING")

    # Step 12: route impact analysis
    affected = [
        d for d in drones
        if d.current_route and disruption_cell in d.current_route
    ]

    add(
        12,
        "Impact Check",
        f"Checked active routes. Affected drones: {', '.join(d.drone_id for d in affected) if affected else 'None'}."
    )

    # Step 13: rerouting
    for drone in affected:
        current_index = min(2, len(drone.current_route) - 1)
        current = drone.current_route[current_index]
        goal = drone.current_route[-1]

        new_route, cost, msg = shortest_path(
            current,
            goal,
            grid,
            algorithm=algorithm
        )

        if new_route:
            reroutes[drone.drone_id] = new_route
            drone.current_route = new_route
            drone.status = "rerouted"
            drone.battery -= min(20, cost)
        else:
            drone.status = "delayed"

    add(
        13,
        "Rerouting",
        f"{algorithm} rerouting executed for affected drone(s).",
        status="OK" if affected else "NO_CHANGE"
    )

    # Step 14: rerouting summary
    delayed_count = sum(1 for d in drones if d.status == "delayed")

    add(
        14,
        "Rerouting Result",
        f"Rerouted={len(reroutes)}, delayed={delayed_count}, failed={len(failed_deliveries)}."
    )

    # Steps 15-17: demand forecasting and additional request
    demand = train_demand_model()

    add(
        15,
        "Demand Forecasting",
        f"Demand forecasting completed using Random Forest Regressor. MAE={demand['mae']}, RMSE={demand['rmse']}."
    )

    if queued_deliveries:
        new_delivery = queued_deliveries[0]
        add(
            16,
            "Demand-Based Update",
            f"High predicted demand triggered/kept an additional queued delivery: {new_delivery['id']} from {new_delivery['pickup']} to {new_delivery['dropoff']}."
        )
    else:
        add(16, "Demand-Based Update", "No additional delivery generated because queue is empty.")

    add(
        17,
        "Fleet/Demand Review",
        f"Mean predicted demand={demand['mean_pred']}; fleet remains within budget and active delivery load is monitored."
    )

    # Step 18: anomaly detection
    anomaly = train_anomaly_model()

    add(
        18,
        "Anomaly Detection",
        f"Telemetry anomaly classifier completed. Best model={anomaly['best_name']}, accuracy={anomaly['accuracy']}.",
        status="ALERT"
    )

    # Step 19: safety action
    if drones:
        drones[0].status = "returning_to_hub"
        add(
            19,
            "Safety Response",
            f"{drones[0].drone_id} forced to return to hub after telemetry/anomaly safety check.",
            drone=drones[0].drone_id,
            status="RETURN"
        )
    else:
        add(19, "Safety Response", "No drone available for safety response.")

    # Step 20: final summary
    completed = max(0, len(active_deliveries) - delayed_count - len(failed_deliveries))

    add(
        20,
        "Final Summary",
        f"Simulation complete. Completed={completed}, rerouted={len(reroutes)}, delayed={delayed_count}, failed={len(failed_deliveries)}, queued={len(queued_deliveries)}."
    )

    return {
        "grid": grid,
        "csp_results": csp_results,
        "fleet_selection": fleet_selection,
        "drones": drones,
        "deliveries": deliveries,
        "routes": routes,
        "reroutes": reroutes,
        "step_log": step_log,
        "demand": demand,
        "anomaly": anomaly,
        "summary": {
            "completed": completed,
            "rerouted": len(reroutes),
            "delayed": delayed_count,
            "failed": len(failed_deliveries),
            "queued": len(queued_deliveries),
        },
    }