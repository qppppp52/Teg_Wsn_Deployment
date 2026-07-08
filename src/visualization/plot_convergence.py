"""Convergence plot utilities"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def plot_convergence(history, ylabel, save_path, title=None, skip_nan=True, hline=None, hline_label=None, fmt="%.2f"):
    plt.figure(figsize=(7, 4))
    arr = np.array(history, dtype=float)
    gens = np.arange(len(arr))
    if skip_nan:
        valid = ~np.isnan(arr)
        if np.any(valid):
            plt.plot(gens[valid], arr[valid], "b-", linewidth=1.5)
    else:
        plt.plot(gens, arr, "b-", linewidth=1.5)
    if hline is not None:
        plt.axhline(y=hline, color="r", linestyle="--", alpha=0.6, linewidth=1.0, label=hline_label)
        if hline_label:
            plt.legend(fontsize=9)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter(fmt))
    plt.xlabel("Generation", fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.title(title or "Convergence Curve", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_dual_convergence(history_a, label_a, history_b, label_b, ylabel, save_path, title=None, fmt="%.2f"):
    plt.figure(figsize=(7, 4))
    gens = np.arange(max(len(history_a), len(history_b)))
    arr_a = np.array(history_a, dtype=float)
    arr_b = np.array(history_b, dtype=float)
    valid_a = ~np.isnan(arr_a)
    valid_b = ~np.isnan(arr_b)
    if np.any(valid_a):
        plt.plot(gens[valid_a], arr_a[valid_a], "b-", linewidth=1.5, label=label_a)
    if np.any(valid_b):
        plt.plot(gens[valid_b], arr_b[valid_b], "r--", linewidth=1.5, label=label_b)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter(fmt))
    plt.xlabel("Generation", fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.title(title or "Convergence", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
