import sys, torch
sys.path.insert(0, '.')
from utils.data_generator import load_dataset
from models.qap_policy import QAPPolicy
from environment.cvrp_env import CVRPEnv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
policy = QAPPolicy()
policy.load_state_dict(torch.load('outputs/n10/best_model.pt', map_location=device))
policy.to(device).eval()

val_coords, val_demands, val_cap = load_dataset('datasets/val_n10.pkl', device=str(device))
bc = val_coords[:256].to(device)
bd = val_demands[:256].to(device)

route_env = CVRPEnv(num_loc=10, device=str(device))
state = route_env.reset({'coords': bc, 'demands': bd,
    'capacity': torch.full((256,), float(val_cap), device=device)})
with torch.no_grad():
    actions, _, _ = policy(state, route_env, deterministic=True)

T = actions.shape[1]
idx = actions.unsqueeze(-1).expand(256, T, 2)
full = torch.cat([bc[:,0:1,:], bc.gather(1,idx), bc[:,0:1,:]], dim=1)
dists = (full[:,1:] - full[:,:-1]).norm(dim=-1).sum(-1)
best_i = dists.argmin().item()

demands = bd[best_i].cpu().tolist()
actions_list = actions[best_i].cpu().tolist()

routes, cur = [], []
for node in actions_list:
    if node == 0:
        if cur: routes.append(cur)
        cur = []
    else:
        cur.append(node)
if cur: routes.append(cur)

print(f"\nCapacity: {int(val_cap)}")
print(f"Demands (node 0=depot): {[int(d) for d in demands]}\n")
for r_idx, route in enumerate(routes):
    d_list = [int(demands[n]) for n in route]
    total  = sum(d_list)
    print(f"Route {r_idx+1}: nodes={route}  demands={d_list}  total={total}/{int(val_cap)}")
