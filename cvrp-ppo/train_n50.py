#!/usr/bin/env python
"""
train_n50.py — Train QAP-DRL on CVRP-50.

Fully self-contained.  Just run:

    python train_n50.py

Run train_n20.py first to confirm the fix is working.

Changelog:
  v4 — ENTROPY_COEF 0.01 → 0.02.
  v5 — 6-panel chart.
  v6 — eta_min 1e-6 → 1e-5.
  v7 — 8-panel chart (4×2).
  v8 — Phase 1 improvements (mirror train_n20.py):
       ENTROPY_COEF  0.02  → 0.05
       EPOCH_SIZE    51200 → 128000 (thesis spec)
       KNN_K         5     → 10
       Evaluation    greedy → augmented ×8
"""

import os, sys, time, json, shutil
import torch
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass

from models.qap_policy import QAPPolicy
from environment.cvrp_env import CVRPEnv
from training.ppo_agent import PPOTrainer
from training.evaluate import evaluate, evaluate_augmented
from utils.seed import set_seed
from utils.data_generator import generate_instances, load_dataset
from utils.ortools_refs import ensure_ortools_ref

# ═══════════════════════════════════════════════════════════════════════════
# Settings  — Phase 1 values marked with (P1)
# ═══════════════════════════════════════════════════════════════════════════
GRAPH_SIZE           = 50
CAPACITY             = 40
BATCH_SIZE           = 256
N_EPOCHS             = 200
EPOCH_SIZE           = 128_000          # (P1) was 51_200
LR                   = 1e-4
ENTROPY_COEF         = 0.05             # (P1) was 0.02
VALUE_COEF           = 0.5
KNN_K                = 10              # (P1) was 5
SEED                 = 1234
BATCHES_PER_EPOCH    = EPOCH_SIZE // BATCH_SIZE          # 500
TOTAL_OPT_STEPS      = N_EPOCHS * BATCHES_PER_EPOCH * 3 * 8   # 2,400,000
AUG_SAMPLES          = 8               # (P1) inference augmentation
OUTPUT_DIR           = os.path.join(SCRIPT_DIR, "outputs", "n50")
EPOCH_DIR            = os.path.join(OUTPUT_DIR, "epochs")
ARCHIVE_DIR          = os.path.join(OUTPUT_DIR, "Archive")
VAL_PATH             = os.path.join(SCRIPT_DIR, "datasets", "val_n50.pkl")
VAL_EVAL_SIZE        = 10_000         # key paper standard: 10K instances for reliable val_tour
ORTOOLS_TIME_LIMIT   = 2.0


# ═══════════════════════════════════════════════════════════════════════════
# Helpers  (identical structure to train_n20.py)
# ═══════════════════════════════════════════════════════════════════════════
class CVRPGenerator:
    def __init__(self, graph_size, capacity):
        self.num_loc = graph_size; self.capacity = capacity

    def generate(self, batch_size, device="cpu"):
        coords, demands, cap = generate_instances(batch_size, self.num_loc, self.capacity, device)
        return {"coords": coords, "demands": demands, "capacity": cap}


def check_feasibility(demands, capacity, actions):
    B, T = actions.shape
    used = torch.zeros(B, device=actions.device)
    violated = torch.zeros(B, dtype=torch.bool, device=actions.device)
    for t in range(T):
        at_depot = (actions[:, t] == 0)
        used = torch.where(at_depot, torch.zeros_like(used), used)
        d = demands.gather(1, actions[:, t:t+1]).squeeze(1)
        used = used + d
        violated = violated | (used > capacity + 1e-6)
    return (~violated).float().mean().item()


def _avg(losses_list, key):
    return sum(l[key] for l in losses_list) / len(losses_list)


def _archive_previous_run(output_dir, epoch_dir, archive_dir, prefix):
    result_files = ["train_log.jsonl","training_curves.png","best_model.pt","best_route.png","cluster_map.png"]
    existing_files = [f for f in result_files if os.path.exists(os.path.join(output_dir, f))]
    has_epoch_pts = os.path.isdir(epoch_dir) and any(f.endswith(".pt") for f in os.listdir(epoch_dir))
    if not existing_files and not has_epoch_pts: return
    today = datetime.now().strftime("%d_%m_%Y")
    base = os.path.join(archive_dir, f"{prefix}_{today}"); dest = base; counter = 2
    while os.path.exists(dest): dest = f"{base}_{counter}"; counter += 1
    os.makedirs(dest, exist_ok=True); print(f"\n  Archiving previous run → {dest}")
    for fname in existing_files:
        shutil.move(os.path.join(output_dir, fname), os.path.join(dest, fname)); print(f"    {fname}")
    if has_epoch_pts:
        dest_epochs = os.path.join(dest, "epochs"); shutil.move(epoch_dir, dest_epochs)
        print(f"    epochs/  ({len(os.listdir(dest_epochs))} files)")
    print()


def _load_chart_history(log_path):
    epochs_hist=[]; tour_hist=[]; reward_hist=[]
    ploss_hist=[]; aploss_hist=[]; lr_hist=[]
    vloss_hist=[]; ent_hist=[]; eloss_hist=[]
    gnorm_hist=[]; adv_hist=[]; clip_hist=[]; lam_hist=[]
    empty = (epochs_hist,tour_hist,reward_hist,ploss_hist,aploss_hist,
             lr_hist,vloss_hist,ent_hist,eloss_hist,gnorm_hist,adv_hist,clip_hist,lam_hist)
    if not os.path.exists(log_path): return empty
    try:
        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                row = json.loads(line)
                ep = row.get("step"); vt = row.get("val_tour")
                if ep is None or vt is None: continue
                pl = row.get("policy_loss") or 0.0
                ent = row.get("entropy") or 0.0
                tt = row.get("train_tour") or 0.0
                epochs_hist.append(ep); tour_hist.append(vt); reward_hist.append(-tt)
                ploss_hist.append(pl); aploss_hist.append(abs(pl))
                lr_hist.append(row.get("lr") or 0.0)
                vloss_hist.append(row.get("value_loss") or 0.0)
                ent_hist.append(ent); eloss_hist.append(ent * ENTROPY_COEF)
                gnorm_hist.append(row.get("grad_norm") or 0.0)
                adv_hist.append(row.get("adv_std") or 0.0)
                clip_hist.append((row.get("clip_fraction") or 0.0) * 100.0)
                lam_hist.append(row.get("lambda_val") or 0.0)
    except Exception as e:
        print(f"  [chart history] Could not reload: {e}")
    if epochs_hist:
        print(f"  Chart history  : reloaded {len(epochs_hist)} epochs from train_log.jsonl")
    return (epochs_hist,tour_hist,reward_hist,ploss_hist,aploss_hist,
            lr_hist,vloss_hist,ent_hist,eloss_hist,gnorm_hist,adv_hist,clip_hist,lam_hist)


def _draw_charts(axes, epochs_hist, tour_hist, reward_hist,
                 ploss_hist, aploss_hist, lr_hist,
                 vloss_hist, ent_hist, eloss_hist,
                 gnorm_hist, adv_hist, clip_hist, lam_hist, ORTOOLS_REF):
    (a00,a01),(a10,a11),(a20,a21),(a30,a31) = axes

    a00.cla()
    a00.plot(epochs_hist, tour_hist, "#378ADD", lw=1.5, label="val tour (aug×8)")
    a00.axhline(y=ORTOOLS_REF, color="darkorange", ls="--", lw=1.2, label=f"OR-Tools ({ORTOOLS_REF:.2f})")
    a00.fill_between(epochs_hist, tour_hist, ORTOOLS_REF, alpha=0.12, color="orange")
    a00.set_title("Tour length (augmented ×8)"); a00.set_xlabel("Epoch"); a00.set_ylabel("Avg distance"); a00.legend(fontsize=8)

    a01.cla()
    a01.plot(epochs_hist, reward_hist, "#1D9E75", lw=1.5)
    a01.set_title("Total reward  (= −train_tour, ↑ = better)"); a01.set_xlabel("Epoch"); a01.set_ylabel("−distance")

    a10.cla()
    a10.plot(epochs_hist, ploss_hist, "crimson", lw=1.5)
    a10.axhline(y=0, color="gray", ls=":", lw=1.0)
    a10.set_title("Actor loss — signed  (↓ more negative = better)")
    a10.set_xlabel("Epoch"); a10.set_ylabel("loss (negative)")

    a11.cla()
    a11.plot(epochs_hist, aploss_hist, color="orange", lw=1.5, label="|actor loss|")
    a11.set_ylabel("|loss|")
    ax_lr = a11.twinx()
    if lr_hist and min(lr_hist) > 0:
        ax_lr.semilogy(epochs_hist, lr_hist, color="#17becf", lw=1.2, ls="--", label="LR")
        ax_lr.axhline(y=1e-5, color="#17becf", ls=":", lw=0.8, alpha=0.6)
    ax_lr.set_ylabel("LR", color="#17becf"); ax_lr.tick_params(axis="y", labelcolor="#17becf")
    a11.set_title("|Actor loss| + LR (floor 1e-5)"); a11.set_xlabel("Epoch")
    lines1,labs1 = a11.get_legend_handles_labels(); lines2,labs2 = ax_lr.get_legend_handles_labels()
    a11.legend(lines1+lines2, labs1+labs2, fontsize=8)

    a20.cla()
    a20.plot(epochs_hist, vloss_hist, "#9467bd", lw=1.5)
    a20.axhline(y=1.0, color="gray", ls=":", lw=1.0, label="~1.0 early")
    a20.set_title("Critic loss  (↓ = better)"); a20.set_xlabel("Epoch"); a20.legend(fontsize=8)

    a21.cla()
    a21.plot(epochs_hist, ent_hist, "#639922", lw=1.5, label="entropy H[π]")
    a21.axhline(y=0.5, color="#639922", ls=":", lw=0.8, alpha=0.5)
    a21.set_ylabel("entropy", color="#639922"); a21.tick_params(axis="y", labelcolor="#639922")
    a21.set_ylim(0.0, max(1.6, max(ent_hist) * 1.1) if ent_hist else 1.6)
    ax_el = a21.twinx()
    ax_el.plot(epochs_hist, eloss_hist, "#BA7517", lw=1.2, ls="--", label=f"entropy loss (×{ENTROPY_COEF})")
    ax_el.set_ylabel(f"−H·{ENTROPY_COEF}", color="#BA7517"); ax_el.tick_params(axis="y", labelcolor="#BA7517")
    if eloss_hist:
        ax_el.set_ylim(0.0, max(eloss_hist) * 1.15)
        ax_el.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, prune='both'))
    a21.set_title("Entropy + entropy loss (dual axis)"); a21.set_xlabel("Epoch")
    lines1,labs1 = a21.get_legend_handles_labels(); lines2,labs2 = ax_el.get_legend_handles_labels()
    a21.legend(lines1+lines2, labs1+labs2, fontsize=8)

    a30.cla()
    a30.plot(epochs_hist, gnorm_hist, "#8c564b", lw=1.5, label="grad norm")
    a30.plot(epochs_hist, adv_hist,   "#e377c2", lw=1.5, ls="--", label="adv_std")
    a30.axhline(y=0.3, color="gray", ls=":", lw=1.0, label="adv_std min")
    a30.set_title("Grad norm & adv_std"); a30.set_xlabel("Epoch"); a30.legend(fontsize=8)

    a31.cla()
    a31.plot(epochs_hist, clip_hist, "#D85A30", lw=1.5, label="clip% (left)")
    a31.set_ylabel("clip fraction %", color="#D85A30"); a31.tick_params(axis="y", labelcolor="#D85A30")
    if clip_hist:
        a31.set_ylim(0.0, max(clip_hist) * 1.2 if max(clip_hist) > 0 else 0.1)
        a31.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, prune='both'))
    ax_lam = a31.twinx()
    ax_lam.plot(epochs_hist, lam_hist, "#534AB7", lw=1.2, ls="--", label="λ (right)")
    ax_lam.set_ylabel("λ value", color="#534AB7"); ax_lam.tick_params(axis="y", labelcolor="#534AB7")
    if lam_hist:
        lam_min = min(lam_hist); lam_max = max(lam_hist)
        lam_pad = max((lam_max - lam_min) * 0.15, 0.1)
        ax_lam.set_ylim(lam_min - lam_pad, lam_max + lam_pad)
        ax_lam.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=5, prune='both'))
    a31.set_title("Clip fraction + Lambda λ (dual axis)"); a31.set_xlabel("Epoch")
    lines1,labs1 = a31.get_legend_handles_labels(); lines2,labs2 = ax_lam.get_legend_handles_labels()
    a31.legend(lines1+lines2, labs1+labs2, fontsize=8)


def plot_route_map(coords_np, actions_list, graph_size, tour_length, title, save_path):
    COLORS=["#1f77b4","#ff7f0e","#2ca02c","#9467bd","#8c564b","#e377c2","#7f7f7f"]
    routes,cur=[],[]
    for node in actions_list:
        if node==0:
            if cur: routes.append(cur); cur=[]
        else: cur.append(node)
    if cur: routes.append(cur)
    fig,ax=plt.subplots(figsize=(8,8)); fig.patch.set_facecolor("white")
    ax.plot(coords_np[0,0],coords_np[0,1],"r*",markersize=20,label="Depot",zorder=5)
    for i in range(1,graph_size+1):
        ax.plot(coords_np[i,0],coords_np[i,1],"o",color="#4a90d9",markersize=5,zorder=4)
        ax.annotate(str(i),(coords_np[i,0],coords_np[i,1]),textcoords="offset points",xytext=(3,3),fontsize=7)
    for r_idx,route in enumerate(routes):
        color=COLORS[r_idx%len(COLORS)]; path=[0]+route+[0]
        for k in range(len(path)-1):
            x0,y0=coords_np[path[k]]; x1,y1=coords_np[path[k+1]]
            ax.annotate("",xy=(x1,y1),xytext=(x0,y0),arrowprops=dict(arrowstyle="->",color=color,lw=1.5))
        ax.plot([],[],color=color,lw=1.5,label=f"Route {r_idx+1}")
    ax.set_title(title,fontsize=14); ax.legend(loc="upper right",fontsize=9)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal")
    fig.savefig(save_path,dpi=150,bbox_inches="tight"); print(f"  Saved: {save_path}"); return fig


def plot_cluster_map(coords_np,actions_list,demands_np,capacity,graph_size,title,save_path):
    import math as _math
    COLORS=["#1f77b4","#ff7f0e","#2ca02c","#9467bd","#8c564b","#e377c2","#7f7f7f"]
    routes,cur=[],[]
    for node in actions_list:
        if node==0:
            if cur: routes.append(cur); cur=[]
        else: cur.append(node)
    if cur: routes.append(cur)
    total_demand=int(sum(demands_np[1:graph_size+1])); k_theory=_math.ceil(total_demand/capacity)
    fig,ax=plt.subplots(figsize=(8,8)); fig.patch.set_facecolor("white")
    ax.plot(coords_np[0,0],coords_np[0,1],"r*",markersize=20,label="Depot",zorder=5)
    for r_idx,route in enumerate(routes):
        color=COLORS[r_idx%len(COLORS)]
        xs=[coords_np[n,0] for n in route]; ys=[coords_np[n,1] for n in route]
        ax.scatter(xs,ys,color=color,s=100,zorder=4,label=f"Cluster {r_idx+1}  ({len(route)} nodes)")
        for n in route:
            ax.annotate(str(n),(coords_np[n,0],coords_np[n,1]),textcoords="offset points",xytext=(5,5),fontsize=8)
    ax.set_title(f"{title}\nK={len(routes)} routes | theory ceil({total_demand}/{capacity})={k_theory}",fontsize=13)
    ax.legend(loc="upper right",fontsize=9); ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect("equal"); fig.savefig(save_path,dpi=150,bbox_inches="tight")
    plt.close(fig); print(f"  Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(SEED)
    os.makedirs(OUTPUT_DIR,exist_ok=True); os.makedirs(EPOCH_DIR,exist_ok=True); os.makedirs(ARCHIVE_DIR,exist_ok=True)

    val_coords,val_demands,val_cap = load_dataset(VAL_PATH, device="cpu")
    ORTOOLS_REF,ORTOOLS_SOURCE = ensure_ortools_ref(
        n=GRAPH_SIZE,val_path=VAL_PATH,n_instances=VAL_EVAL_SIZE,
        coords_t=val_coords,demands_t=val_demands,capacity=val_cap,
        time_limit=ORTOOLS_TIME_LIMIT,
        output_dir=OUTPUT_DIR)

    existing = sorted([f for f in os.listdir(EPOCH_DIR) if f.startswith("epoch_") and f.endswith(".pt")])
    is_resume=bool(existing); start_epoch=1; best_tour=float("inf"); best_epoch=0; ckpt_data=None
    if not is_resume:
        _archive_previous_run(OUTPUT_DIR,EPOCH_DIR,ARCHIVE_DIR,f"n{GRAPH_SIZE}"); os.makedirs(EPOCH_DIR,exist_ok=True)
    else:
        ckpt_data=torch.load(os.path.join(EPOCH_DIR,existing[-1]),map_location=device)
        start_epoch=ckpt_data["iteration"]+1; best_tour=ckpt_data["metrics"].get("val_tour",float("inf")); best_epoch=ckpt_data["iteration"]

    # (P1) knn_k=10
    policy=QAPPolicy(knn_k=KNN_K); env=CVRPEnv(num_loc=GRAPH_SIZE,device=str(device)); generator=CVRPGenerator(GRAPH_SIZE,CAPACITY)
    n_params=sum(p.numel() for p in policy.parameters())
    trainer=PPOTrainer(policy=policy,env=env,generator=generator,lr=LR,entropy_coef=ENTROPY_COEF,value_coef=VALUE_COEF,
                       batch_size=BATCH_SIZE,total_steps=TOTAL_OPT_STEPS,device=str(device),log_dir=OUTPUT_DIR)
    if is_resume and ckpt_data is not None:
        trainer.policy.load_state_dict(ckpt_data["policy_state"]); trainer.optimizer.load_state_dict(ckpt_data["optimizer_state"])

    val_instances=(val_coords[:VAL_EVAL_SIZE].to(device),val_demands[:VAL_EVAL_SIZE].to(device),val_cap)

    print("="*72); print(f"  QAP-DRL Training — CVRP-{GRAPH_SIZE}  [v8 — Phase 1]"); print("="*72)
    print(f"  Graph size: {GRAPH_SIZE}  |  Capacity: {CAPACITY}  |  Params: {n_params}")
    print(f"  Epoch size: {EPOCH_SIZE:,}  ({BATCHES_PER_EPOCH} batches/epoch)  [P1: 128K]")
    print(f"  LR: {LR} (cosine → 1e-5)  |  ENTROPY_COEF: {ENTROPY_COEF}  [P1: 0.05]")
    print(f"  kNN k: {KNN_K}  [P1: k=10]  |  Eval aug: ×{AUG_SAMPLES}  [P1]")
    print(f"  Benchmark: {ORTOOLS_REF:.4f}  ({ORTOOLS_SOURCE})")
    if device.type=="cuda":
        print(f"  Device: {device}  ({torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
    print("="*72); print(f"  {'RESUMING from: '+existing[-1] if is_resume else 'Fresh start: epoch 1'}"); print("="*72)

    log_path=os.path.join(OUTPUT_DIR,"train_log.jsonl")
    fig_live,axes=plt.subplots(4,2,figsize=(14,16))
    fig_live.suptitle(f"QAP-DRL Training — CVRP-{GRAPH_SIZE} [Phase 1]",fontsize=14)

    if is_resume:
        (epochs_hist,tour_hist,reward_hist,ploss_hist,aploss_hist,
         lr_hist,vloss_hist,ent_hist,eloss_hist,gnorm_hist,adv_hist,clip_hist,lam_hist)=_load_chart_history(log_path)
    else:
        epochs_hist=[]; tour_hist=[]; reward_hist=[]; ploss_hist=[]; aploss_hist=[]; lr_hist=[]
        vloss_hist=[]; ent_hist=[]; eloss_hist=[]; gnorm_hist=[]; adv_hist=[]; clip_hist=[]; lam_hist=[]

    header=(f"{'Ep':>4} | {'val_tour':>8} | {'vs ORT':>7} | {'actor_L':>8} | {'critic_L':>8} | "
            f"{'entropy':>7} | {'grad':>6} | {'clip%':>5} | {'adv_std':>7} | {'λ':>6} | {'feas%':>5} | {'t(s)':>5}")
    tqdm.write(f"\n{header}"); tqdm.write("-"*len(header))

    epoch_bar=tqdm(range(start_epoch,N_EPOCHS+1),desc="Training",unit="epoch")
    for epoch in epoch_bar:
        t0=time.time()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        losses_list,feas_list=[],[]
        for batch_idx in range(BATCHES_PER_EPOCH):
            trainer.collect_rollout(); losses=trainer.update(); losses_list.append(losses)
            with torch.no_grad():
                feas=check_feasibility(trainer._last_instance["demands"],CAPACITY,trainer._last_actions)
            feas_list.append(feas)
            if batch_idx%50==0:
                epoch_bar.set_postfix(batch=f"{batch_idx+1}/{BATCHES_PER_EPOCH}",
                    actor=f"{losses['policy_loss']:.3e}",critic=f"{losses['value_loss']:.3f}",
                    ent=f"{losses['entropy']:.3f}",grad=f"{losses['grad_norm']:.3f}")

        avg_ploss=_avg(losses_list,"policy_loss"); avg_vloss=_avg(losses_list,"value_loss")
        avg_ent=_avg(losses_list,"entropy"); avg_gnorm=_avg(losses_list,"grad_norm")
        avg_clip=_avg(losses_list,"clip_fraction"); avg_ratio=_avg(losses_list,"ratio_mean")
        avg_adv_std=_avg(losses_list,"adv_std"); avg_imprv=_avg(losses_list,"improvement")
        avg_train_t=_avg(losses_list,"train_tour"); avg_greedy=_avg(losses_list,"greedy_tour")
        last_lambda=losses_list[-1]["lambda_val"]; last_lr=losses_list[-1]["lr"]
        avg_feas=sum(feas_list)/len(feas_list); elapsed=time.time()-t0
        vram=torch.cuda.memory_allocated()/1e6 if torch.cuda.is_available() else 0.0

        # (P1) augmented evaluation
        val_tour=evaluate_augmented(trainer.policy,val_instances,device,n_samples=AUG_SAMPLES)
        val_gap_ort=100.0*(val_tour-ORTOOLS_REF)/ORTOOLS_REF

        trainer.logger.log_scalars({"val_tour":val_tour,"val_gap_ortools":val_gap_ort,
            "best_tour":min(best_tour,val_tour),"ortools_ref":ORTOOLS_REF,
            "train_tour":avg_train_t,"greedy_tour":avg_greedy,"improvement":avg_imprv,
            "policy_loss":avg_ploss,"value_loss":avg_vloss,"entropy":avg_ent,
            "grad_norm":avg_gnorm,"clip_fraction":avg_clip,"ratio_mean":avg_ratio,
            "adv_std":avg_adv_std,"lambda_val":last_lambda,"lr":last_lr,
            "feasibility":avg_feas,"vram_mb":vram,"epoch_time_s":elapsed},step=epoch)

        torch.save({"iteration":epoch,"policy_state":trainer.policy.state_dict(),
            "optimizer_state":trainer.optimizer.state_dict(),
            "metrics":{"val_tour":val_tour,"val_gap_ortools":val_gap_ort}},
            os.path.join(EPOCH_DIR,f"epoch_{epoch:03d}.pt"))

        is_best=val_tour<best_tour
        if is_best:
            best_tour=val_tour; best_epoch=epoch
            torch.save(trainer.policy.state_dict(),os.path.join(OUTPUT_DIR,"best_model.pt"))

        epoch_bar.set_postfix(best=f"{best_tour:.4f}",cur=f"{val_tour:.4f}")
        tqdm.write(f"{epoch:4d} | {val_tour:8.4f} | {val_gap_ort:+6.1f}% | {avg_ploss:8.3e} | "
                   f"{avg_vloss:8.3f} | {avg_ent:7.4f} | {avg_gnorm:6.3f} | {avg_clip*100:4.1f}% | "
                   f"{avg_adv_std:7.4f} | {last_lambda:6.4f} | {avg_feas*100:4.1f}% | {elapsed:4.0f}s"
                   + (" *" if is_best else ""))

        epochs_hist.append(epoch); tour_hist.append(val_tour); reward_hist.append(-avg_train_t)
        ploss_hist.append(avg_ploss); aploss_hist.append(abs(avg_ploss)); lr_hist.append(last_lr)
        vloss_hist.append(avg_vloss); ent_hist.append(avg_ent); eloss_hist.append(avg_ent*ENTROPY_COEF)
        gnorm_hist.append(avg_gnorm); adv_hist.append(avg_adv_std)
        clip_hist.append(avg_clip*100.0); lam_hist.append(last_lambda)

        _draw_charts(axes,epochs_hist,tour_hist,reward_hist,ploss_hist,aploss_hist,lr_hist,
                     vloss_hist,ent_hist,eloss_hist,gnorm_hist,adv_hist,clip_hist,lam_hist,ORTOOLS_REF)
        fig_live.tight_layout(rect=[0,0,1,0.97])
        fig_live.savefig(os.path.join(OUTPUT_DIR,"training_curves.png"),dpi=150,bbox_inches="tight")

    trainer.logger.close()

    # (P1) knn_k=KNN_K for post-training visualisation
    best_policy=QAPPolicy(knn_k=KNN_K)
    best_policy.load_state_dict(torch.load(os.path.join(OUTPUT_DIR,"best_model.pt"),map_location=device))
    best_policy.to(device).eval()
    n_eval=min(256,val_coords.size(0)); bc=val_coords[:n_eval].to(device); bd=val_demands[:n_eval].to(device)
    route_env=CVRPEnv(num_loc=GRAPH_SIZE,device=str(device))
    state=route_env.reset({"coords":bc,"demands":bd,"capacity":torch.full((n_eval,),float(val_cap),device=device)})
    with torch.no_grad(): actions,_,_=best_policy(state,route_env,deterministic=True)
    T_act=actions.shape[1]
    full=torch.cat([bc[:,0:1,:],bc.gather(1,actions.unsqueeze(-1).expand(n_eval,T_act,2)),bc[:,0:1,:]],dim=1)
    dists=(full[:,1:]-full[:,:-1]).norm(p=2,dim=-1).sum(-1); best_i=dists.argmin().item()
    plot_route_map(bc[best_i].cpu().numpy(),actions[best_i].cpu().tolist(),GRAPH_SIZE,
                   dists[best_i].item(),f"Best Route — CVRP-{GRAPH_SIZE} | Tour: {dists[best_i].item():.2f}",
                   os.path.join(OUTPUT_DIR,"best_route.png"))
    plot_cluster_map(bc[best_i].cpu().numpy(),actions[best_i].cpu().tolist(),
                     bd[best_i].cpu().numpy(),CAPACITY,GRAPH_SIZE,
                     f"Cluster Map — CVRP-{GRAPH_SIZE}",os.path.join(OUTPUT_DIR,"cluster_map.png"))

    print(); print("="*72); print(f"  Training Complete — CVRP-{GRAPH_SIZE} [Phase 1]"); print("="*72)
    print(f"  Best epoch: {best_epoch}  |  Best tour: {best_tour:.4f}  (augmented ×{AUG_SAMPLES})")
    print(f"  OR-Tools ref: {ORTOOLS_REF:.4f}  |  Gap: {100*(best_tour-ORTOOLS_REF)/ORTOOLS_REF:+.2f}%")
    print("="*72); plt.close("all")


if __name__ == "__main__":
    main()
