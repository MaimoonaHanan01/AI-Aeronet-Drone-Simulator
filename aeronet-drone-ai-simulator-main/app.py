import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from grid_model import create_advanced_grid
from csp_validator import validate_layout
from fleet import select_fleet, build_drones
from routing import full_delivery_route
from visual3d import make_city_3d, create_animation_frames
from simulation import run_advanced_simulation, run_20_step_simulation
from ml_models import train_demand_model, train_anomaly_model

st.set_page_config(page_title="AeroNet Pro", page_icon="🚁", layout="wide")

st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg, #061826, #12395A, #2A9D8F);
    color: white;
    padding: 28px;
    border-radius: 24px;
    margin-bottom: 18px;
}
.hero h1 {font-size: 42px; margin: 0;}
.hero p {font-size: 17px; opacity: .92;}
.card {
    background: white;
    border-radius: 18px;
    padding: 18px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 8px 20px rgba(0,0,0,.05);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🚁 AeroNet Pro</h1>
<p>Advanced 3D Real-Time Autonomous Drone Delivery AI Simulator</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("Simulation Controls")
algorithm = st.sidebar.selectbox("Routing Algorithm", ["A*", "Dijkstra", "Weighted A*"])
budget = st.sidebar.slider("Fleet Budget", 6000, 20000, 12000, 500)
disruption_row = st.sidebar.slider("No-Fly Row", 0, 9, 3)
disruption_col = st.sidebar.slider("No-Fly Col", 0, 9, 2)
show_density_penalty = st.sidebar.checkbox("Explain Dense Area Penalty", True)

grid = create_advanced_grid()
results = validate_layout(grid)
passed = all(r.passed for r in results)
fleet = select_fleet(grid, budget)
drones = build_drones(grid, fleet)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Grid", "10 × 10")
m2.metric("CSP", "PASS" if passed else "FAIL")
m3.metric("Fleet", f"{fleet['light'] + fleet['heavy']} drones")
m4.metric("Budget Used", fleet["cost"])
m5.metric("Demand", fleet["demand"])

tabs = st.tabs([
    "3D City",
    "CSP + Grid Intelligence",
    "Routing Algorithms",
    "Live Simulation",
    "ML Intelligence",
    "Drone Telemetry",
    "Validation Checks",
    "20-Step Simulation",
    "Explanation",
])

with tabs[0]:
    st.subheader("3D City Grid")
    st.write("Height represents zone type and demand intensity. No-fly cells appear dark.")
    fig = make_city_3d(grid, drones=drones, title="AeroNet Pro 3D City Grid")
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("CSP Layout Validation")
    for r in results:
        if r.passed:
            st.success(f"{r.rule_id}: Passed")
        else:
            st.error(f"{r.rule_id}: Failed")
        with st.expander(f"Details for {r.rule_id}"):
            for msg in r.messages:
                st.write("-", msg)

    st.subheader("Population Density Integration")
    density_rows = []
    for row in grid:
        for c in row:
            density_rows.append({
                "cell": c.coord,
                "zone": c.zone,
                "density": round(c.density, 0),
                "density_level": c.density_level,
                "demand": c.demand,
                "hub": c.is_hub,
                "charging": c.is_charging,
                "no_fly": c.no_fly,
            })
    st.dataframe(pd.DataFrame(density_rows), use_container_width=True)

with tabs[2]:
    st.subheader("Routing Algorithm Comparison")
    hub = (1, 3)
    pickup = (5, 1)
    dropoff = (9, 8)

    comparison = []
    route_map = {}
    for alg in ["A*", "Dijkstra", "Weighted A*"]:
        route, cost, msg = full_delivery_route(hub, pickup, dropoff, grid, algorithm=alg)
        expanded_nodes = None
        if "Expanded nodes=" in msg:
           expanded_nodes = int(msg.split("Expanded nodes=")[-1])
        comparison.append({
        "Algorithm": alg,
        "Status": "Success" if route else "Failed",
        "Cost": cost if route else None,
        "Route Length": len(route) if route else None,
        "Expanded Nodes": expanded_nodes,
        "Message": msg,
      })
    
        if route:
            route_map[alg] = route

    print("\n================ ROUTING DEBUG RESULTS ================\n")

    for row in comparison:
        print(f"Algorithm       : {row['Algorithm']}")
        print(f"Status          : {row['Status']}")
        print(f"Cost            : {row['Cost']}")
        print(f"Route Length    : {row['Route Length']}")
        print(f"Expanded Nodes  : {row['Expanded Nodes']}")
        print(f"Message         : {row['Message']}")
        print("------------------------------------------------------")

    print("\n=======================================================\n")
    st.dataframe(pd.DataFrame(comparison), use_container_width=True)

    selected_route = route_map.get(algorithm)
    if selected_route:
        st.plotly_chart(make_city_3d(grid, routes={algorithm: selected_route}, drones=drones), use_container_width=True)
        st.subheader("Animated Route Preview")
        anim = create_animation_frames(grid, selected_route, drone_id="D1")
        st.plotly_chart(anim, use_container_width=True)

    if show_density_penalty:
        st.info("Dense areas increase route cost slightly, representing safety and congestion concerns.")

with tabs[3]:
    st.subheader("Live Advanced Simulation")

    if st.button("Run Advanced Simulation"):
        sim = run_advanced_simulation(algorithm=algorithm, budget=budget, disruption_cell=(disruption_row, disruption_col))

        st.plotly_chart(make_city_3d(sim["grid"], routes=sim["routes"], drones=sim["drones"], title="Live Simulation Routes"), use_container_width=True)

        st.subheader("Event Log")
        progress = st.progress(0)
        log_box = st.empty()
        shown = []
        for i, event in enumerate(sim["events"], start=1):
            shown.append(event)
            progress.progress(i / len(sim["events"]))
            log_box.text("\\n".join(shown))

        st.subheader("Drone Status")
        st.dataframe(pd.DataFrame([d.__dict__ for d in sim["drones"]]), use_container_width=True)

with tabs[4]:
    st.subheader("Demand Forecasting")
    demand = train_demand_model()
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", demand["mae"])
    c2.metric("RMSE", demand["rmse"])
    c3.metric("Mean Predicted Demand", demand["mean_pred"])
    st.dataframe(demand["importance"], use_container_width=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(demand["importance"]["feature"], demand["importance"]["importance"])
    ax.invert_yaxis()
    ax.set_title("Demand Feature Importance")
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Anomaly Detection")
    anomaly = train_anomaly_model()
    st.metric("Best Model", anomaly["best_name"])
    st.metric("Accuracy", anomaly["accuracy"])
    st.dataframe(anomaly["results"], use_container_width=True)
    st.text(anomaly["report"])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(anomaly["cm"])
    ax.set_xticks(range(len(anomaly["labels"])))
    ax.set_yticks(range(len(anomaly["labels"])))
    ax.set_xticklabels(anomaly["labels"], rotation=35, ha="right")
    ax.set_yticklabels(anomaly["labels"])
    ax.set_title("Anomaly Confusion Matrix")
    for i in range(len(anomaly["labels"])):
        for j in range(len(anomaly["labels"])):
            ax.text(j, i, anomaly["cm"][i, j], ha="center", va="center")
    st.pyplot(fig)

with tabs[5]:
    st.subheader("Drone Telemetry Panel")
    telemetry = []
    for d in drones:
        telemetry.append({
            "Drone ID": d.drone_id,
            "Type": d.drone_type,
            "Hub": d.hub,
            "Battery": d.battery,
            "Payload": d.payload_kg,
            "Max Payload": d.max_payload,
            "Speed": d.speed,
            "Status": d.status,
        })
    st.dataframe(pd.DataFrame(telemetry), use_container_width=True)

    st.info("In the simulation, route cost reduces battery. If anomaly is detected, drone status changes to returning_to_hub.")

with tabs[6]:
    st.subheader("Automatic Validation Checks")

    st.write("This tab verifies whether the AI results are logically correct and defendable.")

    from grid_model import manhattan
    from routing import full_delivery_route

    validation_rows = []

    # 1. CSP validation
    for result in results:
        validation_rows.append({
            "Check": f"CSP {result.rule_id}",
            "Status": "PASS" if result.passed else "FAIL",
            "Details": "; ".join(result.messages)
        })

    # 2. Residential coverage manual check
    hub_cells = []
    residential_cells = []
    charging_cells = []

    for row in grid:
        for cell in row:
            if cell.is_hub:
                hub_cells.append(cell.coord)
            if cell.zone == "Residential":
                residential_cells.append(cell.coord)
            if cell.is_charging:
                charging_cells.append(cell.coord)

    residential_ok = True
    bad_residential = []

    for res_cell in residential_cells:
        nearest = min(manhattan(res_cell, hub) for hub in hub_cells)
        if nearest > 3:
            residential_ok = False
            bad_residential.append(f"{res_cell} distance={nearest}")

    validation_rows.append({
        "Check": "Residential cells within 3 cells of hub",
        "Status": "PASS" if residential_ok else "FAIL",
        "Details": "All residential cells covered" if residential_ok else ", ".join(bad_residential)
    })

    # 3. Hub charging manual check
    charging_ok = True
    bad_hubs = []

    for hub in hub_cells:
        nearest = min(manhattan(hub, chg) for chg in charging_cells)
        if nearest > 2:
            charging_ok = False
            bad_hubs.append(f"{hub} distance={nearest}")

    validation_rows.append({
        "Check": "Every hub has charging pad within 2 cells",
        "Status": "PASS" if charging_ok else "FAIL",
        "Details": "All hubs have nearby charging pads" if charging_ok else ", ".join(bad_hubs)
    })

    # 4. Route validation
    hub = (1, 1)
    pickup = (5, 1)
    dropoff = (9, 8)

    route, route_cost, route_msg = full_delivery_route(
        hub,
        pickup,
        dropoff,
        grid,
        algorithm=algorithm
    )

    if route:
        route_starts_correct = route[0] == hub
        route_ends_correct = route[-1] == hub

        no_fly_violations = [
            cell for cell in route
            if grid[cell[0]][cell[1]].no_fly
        ]

        movement_valid = True
        bad_moves = []

        for i in range(len(route) - 1):
            r1, c1 = route[i]
            r2, c2 = route[i + 1]
            dist = abs(r1 - r2) + abs(c1 - c2)

            if dist != 1:
                movement_valid = False
                bad_moves.append(f"{route[i]} -> {route[i + 1]}")

        validation_rows.append({
            "Check": f"{algorithm} route exists",
            "Status": "PASS",
            "Details": f"Route cost={route_cost}, length={len(route)}"
        })

        validation_rows.append({
            "Check": "Route starts and ends at hub",
            "Status": "PASS" if route_starts_correct and route_ends_correct else "FAIL",
            "Details": f"Start={route[0]}, End={route[-1]}, Expected Hub={hub}"
        })

        validation_rows.append({
            "Check": "Route avoids no-fly cells",
            "Status": "PASS" if len(no_fly_violations) == 0 else "FAIL",
            "Details": "No no-fly violation" if len(no_fly_violations) == 0 else str(no_fly_violations)
        })

        validation_rows.append({
            "Check": "Route moves one cell at a time",
            "Status": "PASS" if movement_valid else "FAIL",
            "Details": "All moves are valid 4-direction moves" if movement_valid else ", ".join(bad_moves)
        })

    else:
        validation_rows.append({
            "Check": f"{algorithm} route exists",
            "Status": "FAIL",
            "Details": route_msg
        })

    validation_df = pd.DataFrame(validation_rows)

    st.dataframe(validation_df, width="stretch")

    pass_count = (validation_df["Status"] == "PASS").sum()
    fail_count = (validation_df["Status"] == "FAIL").sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Checks", len(validation_df))
    c2.metric("Passed", pass_count)
    c3.metric("Failed", fail_count)

    if fail_count == 0:
        st.success("All validation checks passed. Results are logically consistent.")
    else:
        st.warning("Some validation checks failed. Review the failed rows before final demo.")

with tabs[7]:
    st.subheader("20-Step Simulation Scenario")

    st.write(
        "This tab runs the required 20-step demo scenario: initialization, CSP validation, "
        "fleet planning, route planning, drone movement, no-fly disruption, rerouting, "
        "demand forecasting, anomaly detection, safety action, and final summary."
    )

    if st.button("Run 20-Step Simulation"):
        sim20 = run_20_step_simulation(
            algorithm=algorithm,
            budget=budget,
            disruption_cell=(disruption_row, disruption_col)
        )

        st.subheader("20-Step Event Log")
        step_df = pd.DataFrame(sim20["step_log"])
        st.dataframe(step_df, use_container_width=True)

        log_text = "\n".join(
            [
                f"Step {row['Step']}: {row['Phase']} - {row['Event']} [{row['Status']}]"
                for row in sim20["step_log"]
            ]
        )

        st.download_button(
            label="Download 20-Step Simulation Log",
            data=log_text,
            file_name="simulation_20_step_log.txt",
            mime="text/plain"
        )

        st.subheader("Final Simulation Summary")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Completed", sim20["summary"]["completed"])
        c2.metric("Rerouted", sim20["summary"]["rerouted"])
        c3.metric("Delayed", sim20["summary"]["delayed"])
        c4.metric("Failed", sim20["summary"]["failed"])
        c5.metric("Queued", sim20["summary"]["queued"])

        st.subheader("Planned Routes")
        st.plotly_chart(
            make_city_3d(
                sim20["grid"],
                routes=sim20["routes"],
                drones=sim20["drones"],
                title="20-Step Simulation: Planned Routes"
            ),
            use_container_width=True
        )

        if sim20["reroutes"]:
            st.subheader("Rerouted Paths After No-Fly Disruption")
            st.plotly_chart(
                make_city_3d(
                    sim20["grid"],
                    routes=sim20["reroutes"],
                    drones=sim20["drones"],
                    title="20-Step Simulation: Rerouted Paths"
                ),
                use_container_width=True
            )

        st.subheader("Drone Status After 20 Steps")
        st.dataframe(
            pd.DataFrame([d.__dict__ for d in sim20["drones"]]),
            use_container_width=True
        )

        st.subheader("Delivery Requests")
        st.dataframe(
            pd.DataFrame(sim20["deliveries"]),
            use_container_width=True
        )
with tabs[8]:
    st.subheader("Viva Defense Points")
    st.markdown("""
### Why this is advanced
This version upgrades the original 10x10 simulator into a 3D real-time AI dashboard.

### Algorithms used
- CSP for layout validation
- A*, Dijkstra, Weighted A* for route planning
- Heuristic fleet selection
- Random Forest Regressor for demand forecasting
- Decision Tree and Random Forest for anomaly classification

### Dataset usage
- Bike Sharing Demand is used as a proxy for hourly delivery demand.
- US population density maps cells into demand intensity.
- Drone telemetry log is used for anomaly classification with rule-based labels.

### Why not one dataset?
Because each AI module solves a different problem:
- routing needs a grid
- forecasting needs time/weather demand data
- anomaly detection needs telemetry data
- CSP needs rule-based city layout data
""")
