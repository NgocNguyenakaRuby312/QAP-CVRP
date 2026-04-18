"""
Patch script: rewrites the chart section in train_n20.py and train_n50.py
to a 2-row × 3-column layout adding Value Loss, Grad Norm + adv_std, LR.
Run once: python _chart_patch.py
Then delete this file.
"""
import re, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── snippets that replace the old chart blocks ──────────────────────────────

OLD_HIST_LINES = """\
    if is_resume:
        epochs_hist, tour_hist, ploss_hist, ent_hist = _load_chart_history(log_path)
    else:
        epochs_hist, tour_hist, ploss_hist, ent_hist = [], [], [], []"""

NEW_HIST_LINES = """\
    if is_resume:
        epochs_hist, tour_hist, ploss_hist, ent_hist = _load_chart_history(log_path)
        vloss_hist, gnorm_hist, adv_hist, lr_hist = [], [], [], []
    else:
        epochs_hist, tour_hist, ploss_hist, ent_hist = [], [], [], []
        vloss_hist, gnorm_hist, adv_hist, lr_hist = [], [], [], []"""

OLD_FIG_INIT_3 = """\
    fig_live, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    fig_live.suptitle(f"QAP-DRL Training — CVRP-{GRAPH_SIZE}", fontsize=14)"""

NEW_FIG_INIT_6 = """\
    fig_live, axes = plt.subplots(2, 3, figsize=(15, 8))
    (ax1, ax2, ax3), (ax4, ax5, ax6) = axes
    fig_live.suptitle(f"QAP-DRL Training — CVRP-{GRAPH_SIZE}", fontsize=14)"""

OLD_APPEND = """\
        # Plot |policy_loss| so down = better (fix v4)
        epochs_hist.append(epoch); tour_hist.append(val_tour)
        ploss_hist.append(abs(avg_ploss)); ent_hist.append(avg_ent)"""

NEW_APPEND = """\
        # Accumulate history for all 6 panels
        epochs_hist.append(epoch); tour_hist.append(val_tour)
        ploss_hist.append(abs(avg_ploss)); ent_hist.append(avg_ent)
        vloss_hist.append(avg_vloss); gnorm_hist.append(avg_gnorm)
        adv_hist.append(avg_adv_std); lr_hist.append(last_lr)"""

OLD_PLOT_BLOCK = """\
        ax1.cla()
        ax1.plot(epochs_hist, tour_hist, "b-", lw=1.5, label="Model")
        ax1.axhline(y=ORTOOLS_REF, color="darkorange", ls="--", lw=1.2,
                    label=f"OR-Tools ({ORTOOLS_REF:.2f})")
        ax1.fill_between(epochs_hist, tour_hist, ORTOOLS_REF, alpha=0.15, color="orange")
        ax1.set_title("Tour Length"); ax1.set_ylabel("Avg Distance")
        ax1.set_xlabel("Epoch"); ax1.legend(fontsize=10)

        ax2.cla(); ax2.plot(epochs_hist, ploss_hist, color="orange", lw=1.5)
        ax2.set_title("|Policy Loss|  (↓ = better)")   # clear label
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("|loss|")

        ax3.cla(); ax3.plot(epochs_hist, ent_hist, color="green", lw=1.5)
        ax3.axhline(y=0.5, color="gray", ls=":", lw=1.0, label="min healthy (0.5)")
        ax3.set_title("Policy Entropy  (stay > 0.5)")
        ax3.set_xlabel("Epoch"); ax3.legend(fontsize=9)

        fig_live.tight_layout(rect=[0,0,1,0.93])
        fig_live.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"),
                         dpi=150, bbox_inches="tight")"""

NEW_PLOT_BLOCK = """\
        # Row 1: Tour length | |Policy loss| | Entropy
        ax1.cla()
        ax1.plot(epochs_hist, tour_hist, "b-", lw=1.5, label="Model")
        ax1.axhline(y=ORTOOLS_REF, color="darkorange", ls="--", lw=1.2,
                    label=f"OR-Tools ({ORTOOLS_REF:.2f})")
        ax1.fill_between(epochs_hist, tour_hist, ORTOOLS_REF, alpha=0.15, color="orange")
        ax1.set_title("Tour Length"); ax1.set_ylabel("Avg Distance")
        ax1.set_xlabel("Epoch"); ax1.legend(fontsize=9)

        ax2.cla(); ax2.plot(epochs_hist, ploss_hist, color="orange", lw=1.5)
        ax2.set_title("|Policy Loss|  (↓ = better)")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("|loss|")

        ax3.cla(); ax3.plot(epochs_hist, ent_hist, color="green", lw=1.5)
        ax3.axhline(y=0.5, color="gray", ls=":", lw=1.0, label="min healthy (0.5)")
        ax3.set_title("Entropy  (stay > 0.5)")
        ax3.set_xlabel("Epoch"); ax3.legend(fontsize=9)

        # Row 2: Value loss | Grad norm + adv_std | Learning rate
        ax4.cla(); ax4.plot(epochs_hist, vloss_hist, color="#9467bd", lw=1.5)
        ax4.axhline(y=1.0, color="gray", ls=":", lw=1.0, label="target ~1.0 early")
        ax4.set_title("Value Loss  (↓ after warm-up)")
        ax4.set_xlabel("Epoch"); ax4.legend(fontsize=9)

        ax5.cla()
        ax5.plot(epochs_hist, gnorm_hist, color="#8c564b", lw=1.5, label="grad norm")
        ax5.plot(epochs_hist, adv_hist,   color="#e377c2", lw=1.5, label="adv_std", ls="--")
        ax5.axhline(y=0.3, color="gray", ls=":", lw=1.0, label="adv_std min (0.3)")
        ax5.set_title("Grad Norm & Adv Std")
        ax5.set_xlabel("Epoch"); ax5.legend(fontsize=9)

        ax6.cla(); ax6.semilogy(epochs_hist, lr_hist, color="#17becf", lw=1.5)
        ax6.axhline(y=1e-5, color="gray", ls=":", lw=1.0, label="eta_min target (1e-5)")
        ax6.set_title("Learning Rate (log scale)")
        ax6.set_xlabel("Epoch"); ax6.legend(fontsize=9)

        fig_live.tight_layout(rect=[0,0,1,0.96])
        fig_live.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"),
                         dpi=150, bbox_inches="tight")"""


def patch_file(path):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()

    changes = [
        (OLD_HIST_LINES,   NEW_HIST_LINES,   "hist init"),
        (OLD_FIG_INIT_3,   NEW_FIG_INIT_6,   "fig init"),
        (OLD_APPEND,       NEW_APPEND,        "append"),
        (OLD_PLOT_BLOCK,   NEW_PLOT_BLOCK,    "plot block"),
    ]

    for old, new, label in changes:
        if old in src:
            src = src.replace(old, new, 1)
            print(f"  [{label}] patched")
        else:
            print(f"  [{label}] NOT FOUND — already patched or text changed")

    with open(path, "w", encoding="utf-8") as f:
        f.write(src)


for fname in ["train_n20.py", "train_n50.py"]:
    fpath = os.path.join(SCRIPT_DIR, fname)
    print(f"\nPatching {fname} ...")
    patch_file(fpath)

print("\nDone.")
