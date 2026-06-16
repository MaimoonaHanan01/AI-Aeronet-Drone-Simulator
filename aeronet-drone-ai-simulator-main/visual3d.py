import plotly.graph_objects as go
import pandas as pd
from config import ZONE_COLORS, ZONE_HEIGHT


def add_cell_cube(fig, x, y, height, color, opacity=0.72, name=None, hovertext=None):
    # cuboid vertices
    z0, z1 = 0, height
    vertices = [
        (x, y, z0), (x+1, y, z0), (x+1, y+1, z0), (x, y+1, z0),
        (x, y, z1), (x+1, y, z1), (x+1, y+1, z1), (x, y+1, z1),
    ]
    xs, ys, zs = zip(*vertices)
    i = [0, 0, 0, 1, 2, 3, 4, 4, 5, 6, 7, 4]
    j = [1, 2, 3, 2, 3, 0, 5, 6, 6, 7, 4, 7]
    k = [2, 3, 0, 6, 7, 4, 6, 7, 1, 2, 3, 0]
    fig.add_trace(go.Mesh3d(
        x=xs, y=ys, z=zs,
        i=i, j=j, k=k,
        color=color,
        opacity=opacity,
        name=name or "cell",
        hovertext=hovertext,
        hoverinfo="text",
        showscale=False
    ))


def make_city_3d(grid, routes=None, drones=None, title="AeroNet Pro 3D City"):
    fig = go.Figure()

    for row in grid:
        for cell in row:
            height = ZONE_HEIGHT.get(cell.zone, 0.5) + min(cell.demand / 80, 1.5)
            color = "#263238" if cell.no_fly else ZONE_COLORS.get(cell.zone, "#CCCCCC")
            hover = (
                f"Cell: {cell.coord}<br>"
                f"Zone: {cell.zone}<br>"
                f"Density: {cell.density:.0f}<br>"
                f"Density Level: {cell.density_level}<br>"
                f"Demand: {cell.demand}<br>"
                f"No-Fly: {cell.no_fly}"
            )
            add_cell_cube(fig, cell.col, cell.row, height, color, name=cell.zone, hovertext=hover)

            label = None
            if cell.is_hub:
                label = "HUB"
            elif cell.is_charging:
                label = "CHG"
            elif cell.is_medical_pickup:
                label = "MED"
            elif cell.no_fly:
                label = "NO-FLY"

            if label:
                fig.add_trace(go.Scatter3d(
                    x=[cell.col + .5], y=[cell.row + .5], z=[height + .35],
                    mode="text",
                    text=[label],
                    textfont=dict(size=12, color="black"),
                    showlegend=False,
                    hoverinfo="skip"
                ))

    if routes:
        for rid, route in routes.items():
            if route:
                xs = [c[1] + .5 for c in route]
                ys = [c[0] + .5 for c in route]
                zs = [2.6 for _ in route]
                fig.add_trace(go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode="lines+markers",
                    line=dict(width=7),
                    marker=dict(size=4),
                    name=f"Route {rid}",
                ))

    if drones:
        for d in drones:
            x, y, z = d.position
            fig.add_trace(go.Scatter3d(
                x=[x + .5], y=[y + .5], z=[z + 1.8],
                mode="markers+text",
                marker=dict(size=8, symbol="diamond"),
                text=[d.drone_id],
                textposition="top center",
                name=f"Drone {d.drone_id}",
                hovertext=f"{d.drone_id}<br>Battery: {d.battery:.1f}%<br>Status: {d.status}",
                hoverinfo="text"
            ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Column",
            yaxis_title="Row",
            zaxis_title="Height / Density",
            aspectmode="manual",
            aspectratio=dict(x=1.2, y=1.2, z=.55),
            camera=dict(eye=dict(x=1.35, y=1.45, z=1.05)),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=740,
        legend=dict(x=.01, y=.99)
    )
    return fig


def create_animation_frames(grid, route, drone_id="D1"):
    fig = make_city_3d(grid, routes={"planned": route}, drones=[])
    frames = []

    for idx, coord in enumerate(route):
        x, y, z = coord[1] + .5, coord[0] + .5, 3.2
        trail = route[:idx+1]
        frames.append(go.Frame(
            data=[
                go.Scatter3d(
                    x=[c[1]+.5 for c in trail],
                    y=[c[0]+.5 for c in trail],
                    z=[3.0 for _ in trail],
                    mode="lines+markers",
                    line=dict(width=8),
                    marker=dict(size=4),
                    name="Live trail",
                ),
                go.Scatter3d(
                    x=[x], y=[y], z=[z],
                    mode="markers+text",
                    marker=dict(size=10, symbol="diamond"),
                    text=[drone_id],
                    textposition="top center",
                    name=drone_id,
                )
            ],
            name=str(idx)
        ))

    fig.frames = frames
    fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="lines+markers", name="Live trail"))
    fig.add_trace(go.Scatter3d(x=[], y=[], z=[], mode="markers+text", name=drone_id))

    fig.update_layout(
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(label="Play", method="animate", args=[None, {"frame": {"duration": 450, "redraw": True}, "fromcurrent": True}]),
                dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]),
            ],
            x=0.05, y=0.05
        )]
    )
    return fig
