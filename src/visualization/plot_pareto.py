import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_pareto_front(objectives, labels, save_path, title="Pareto Front Comparison"):
    plt.figure(figsize=(8,6))
    colors = plt.cm.tab10(np.linspace(0,1,len(objectives)))
    for i,(objs,label) in enumerate(zip(objectives, labels)):
        if len(objs)==0: continue
        plt.scatter(objs[:,0], objs[:,1]/1e6, c=[colors[i]], label=label,
                    alpha=0.7, edgecolors="black", linewidth=0.5)
    plt.xlabel("Coverage"); plt.ylabel("Total Throughput R_sum (Mbps)")
    plt.title(title); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150); plt.close()

def plot_single_pareto(objs, label, save_path):
    plt.figure(figsize=(7,5))
    plt.scatter(objs[:,0], objs[:,1]/1e6, c="blue", alpha=0.7,
                edgecolors="black", linewidth=0.5)
    plt.xlabel("Coverage"); plt.ylabel("Total Throughput R_sum (Mbps)")
    plt.title(f"Pareto Front - {label}")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150); plt.close()
