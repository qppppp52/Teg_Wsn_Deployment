import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_deployment(solution, ctx, save_path):
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection="3d")
    coords = ctx.candidate_coords
    sids = np.where(solution.x==1)[0]
    aids = np.where(solution.y==1)[0]
    ax.scatter(coords[sids,0], coords[sids,1], coords[sids,2],
               c="blue", marker="o", s=50, label="Sensors", alpha=0.8)
    ax.scatter(coords[aids,0], coords[aids,1], coords[aids,2],
               c="red", marker="^", s=80, label="APs", alpha=0.9)
    ax.scatter(ctx.target_coords[:,0], ctx.target_coords[:,1], ctx.target_coords[:,2],
               c="green", marker=".", s=10, label="Targets", alpha=0.5)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.legend(); ax.set_title("Deployment Visualization")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150); plt.close()
