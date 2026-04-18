"""One-off script to generate cluster_map.png from saved best_model.pt."""
import os, math, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from models.qap_policy import QAPPolicy
from environment.cvrp_env import CVRPEnv
from utils.data_generator import load_dataset

GRAPH_SIZE = 10
CAPACITY   = 30
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs", "n10")
VAL_PATH   = os.path.join(SCRIPT_DIR, "datasets", "val_n10.pkl")
COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#9467bd","#8c564b",
          "#e377c2","#7f7f7f","#bcbd22","#17becf"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

best_policy = QAPPolicy()
best_policy.load_state_dict(
    torch.load(os.path.join(OUTPUT_DIR, "best_model.pt"), map_location=device)
)
best_policy.to(device).eval()

val_coords, val_demands, val_cap = load_dataset(VAL_PATH, device=str(device))
n_eval = min(256, val_coords.size(0))
bc = val_coords[:n_eval].to(device)
bd = val_demands[:n_eval].to(device)

route_env = CVRPEnv(num_loc=GRAPH_SIZE, device=str(device))
state = route_env.reset({
    "coords": bc, "demands": bd,
    "capacity": torch.full((n_eval,), float(val_cap), device=device),
})
with torch.no_grad():
    actions, _, _ = best_policy(state, route_env, deterministic=True)

T_act = actions.shape[1]
idx = actions.unsqueeze(-1).expand(n_eval, T_act, 2)
route_pts = bc.gather(1, idx)
depot = bc[:, 0:1, :]
full = torch.cat([depot, route_pts, depot], dim=1)
dists = (full[:, 1:] - full[:, :-1]).norm(p=2, dim=-1).sum(-1)
best_i = dists.argmin().item()

coords_np    = bc[best_i].cpu().numpy()
actions_list = actions[best_i].cpu().tolist()
demands_np   = bd[best_i].cpu().numpy()

# Parse routes
routes, cur = [], []
for node in actions_list:
    if node == 0:
        if cur:
            routes.append(cur)
        cur = []
    else:
        cur.append(node)
if cur:
    routes.append(cur)

total_demand = int(sum(demands_np[1:GRAPH_SIZE + 1]))
k_theory = math.ceil(total_demand / CAPACITY)
k_actual = len(routes)

fig, ax = plt.subplots(figsize=(8, 8))
fig.patch.set_facecolor("white")
ax.plot(coords_np[0, 0], coords_np[0, 1], "r*", markersize=20, label="Depot", zorder=5)
for r_idx, route in enumerate(routes):
    color = COLORS[r_idx % len(COLORS)]
    xs = [coords_np[n, 0] for n in route]
    ys = [coords_np[n, 1] for n in route]
    ax.scatter(xs, ys, color=color, s=100, zorder=4,
               label=f"Cluster {r_idx+1}  ({len(route)} nodes)")
    for n in route:
        ax.annotate(str(n), (coords_np[n, 0], coords_np[n, 1]),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)

ax.set_title(
    f"Cluster Map - CVRP-{GRAPH_SIZE}\n"
    f"K = {k_actual} routes  |  theory ceil({total_demand}/{CAPACITY}) = {k_theory}",
    fontsize=13,
)
ax.legend(loc="upper right", fontsize=9)
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("y", fontsize=12)
ax.set_aspect("equal")

save_path = os.path.join(OUTPUT_DIR, "cluster_map.png")
fig.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {save_path}")
