from pathlib import Path
from grid_model import create_advanced_grid
from routing import full_delivery_route
from visual3d import create_animation_frames

grid = create_advanced_grid()
hub = (1, 1)
pickup = (5, 1)
dropoff = (9, 8)

route, cost, msg = full_delivery_route(hub, pickup, dropoff, grid, algorithm="A*")
if route is None:
    raise RuntimeError(msg)

fig = create_animation_frames(grid, route, drone_id="D1")

out = Path("outputs/aeronet_3d_animation.html")
out.parent.mkdir(exist_ok=True, parents=True)
fig.write_html(out)

print(f"3D animation saved to: {out.resolve()}")
