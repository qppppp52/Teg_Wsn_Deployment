#!/usr/bin/env python
"""
Pareto 解部署状态可视化 — 内表面温度热力图 + 传感器/AP/散热片叠加

用法:
  python tools/plot_selected_pareto_solution.py --mode knee
  python tools/plot_selected_pareto_solution.py --mode highest_coverage
"""
import os, sys, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch as LegendPatch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')


def load_data(npz_path, ctx_path):
    p = np.load(npz_path, allow_pickle=True)
    c = np.load(ctx_path, allow_pickle=True)
    return p, c


def select_pareto_solution(pareto_data, mode, index=0):
    objs = pareto_data["objectives"]
    cov, rsum = objs[:, 0], objs[:, 1]
    N = len(cov)

    _, uidx = np.unique(np.round(objs, 6), axis=0, return_index=True)
    u = objs[np.sort(uidx)]
    nd = np.ones(len(u), dtype=bool)
    for i in range(len(u)):
        for j in range(len(u)):
            if i != j and u[j, 0] >= u[i, 0] and u[j, 1] >= u[i, 1] and (
                    u[j, 0] > u[i, 0] or u[j, 1] > u[i, 1]):
                nd[i] = False; break
    pareto = u[nd]; pareto = pareto[np.argsort(pareto[:, 0])]
    pareto_indices = []
    for p_pt in pareto:
        for orig_idx in range(N):
            if abs(cov[orig_idx] - p_pt[0]) < 1e-6 and abs(rsum[orig_idx] - p_pt[1]) < 1e-6:
                pareto_indices.append(orig_idx); break

    if mode == "highest_coverage":   return np.argmax(cov), "max Coverage"
    if mode == "highest_rsum":       return np.argmax(rsum), "max Rsum"
    if mode == "knee" and len(pareto) >= 2:
        cn = (pareto[:,0] - pareto[:,0].min()) / (pareto[:,0].max() - pareto[:,0].min() + 1e-12)
        rn = (pareto[:,1] - pareto[:,1].min()) / (pareto[:,1].max() - pareto[:,1].min() + 1e-12)
        d = np.sqrt((1-cn)**2 + (1-rn)**2)
        return pareto_indices[np.argmin(d)], "knee"
    if mode == "index": return max(0, min(index, N-1)), f"index={index}"
    return 0, "first"


def reconstruct_solution(pareto_data, sol_idx, ctx_data):
    K = int(ctx_data["num_candidates"])
    return {
        "x": pareto_data["x"][sol_idx],
        "y": pareto_data["y"][sol_idx],
        "c": pareto_data["c"][sol_idx],
        "cv": float(pareto_data["cv"][sol_idx]),
        "coverage": float(pareto_data["objectives"][sol_idx, 0]),
        "throughput": float(pareto_data["objectives"][sol_idx, 1]),
        "sink_s_grids": list(np.where(pareto_data["z_sink_sensor"][sol_idx] > 0)[0]),
        "sink_a_grids": list(np.where(pareto_data["z_sink_ap"][sol_idx] > 0)[0]),
    }


def plot_3d_deployment(pareto_data, sol_idx, ctx_data, mode_label, save_path):
    Lx, Ly, Lz = 5.0, 5.0, 5.0; gs = 1.0
    K = int(ctx_data["num_candidates"])
    sol = reconstruct_solution(pareto_data, sol_idx, ctx_data)
    cand = ctx_data["candidate_points"]
    T_r = ctx_data["T_r"]
    targets = ctx_data["target_coords"]

    sensor_ids = np.where(sol["x"] == 1)[0];  ns = len(sensor_ids)
    ap_ids     = np.where(sol["y"] == 1)[0];  na = len(ap_ids)

    # ===== colormap: 蓝(冷) → 红(热) =====
    cmap = plt.cm.coolwarm
    t_min, t_max = float(T_r.min()), float(T_r.max())
    norm = Normalize(vmin=t_min, vmax=t_max)

    # ===== 图 =====
    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    # ===== 1) 温度面片（只画可见的3个内表面：Bottom, Left, Back） =====
    # elev=30, azim=-60 视角下可见底(z=0)、左(x=0)、后(y=Ly)三面
    visible_faces = {1, 3, 4}  # Bottom=1, Back(y=Ly)=3, Left(x=0)=4
    patches, pcolors = [], []
    for k in range(K):
        fid = int(cand[k, 3])
        if fid not in visible_faces:
            continue
        x0, y0, z0 = float(cand[k,0]), float(cand[k,1]), float(cand[k,2])
        h = gs / 2.2
        if fid in (0, 1):   v = [[x0-h,y0-h,z0],[x0+h,y0-h,z0],[x0+h,y0+h,z0],[x0-h,y0+h,z0]]
        elif fid in (2, 3): v = [[x0-h,y0,z0-h],[x0+h,y0,z0-h],[x0+h,y0,z0+h],[x0-h,y0,z0+h]]
        else:               v = [[x0,y0-h,z0-h],[x0,y0+h,z0-h],[x0,y0+h,z0+h],[x0,y0-h,z0+h]]
        patches.append(v)
        pcolors.append(cmap(norm(T_r[k])))
    pc = Poly3DCollection(patches, alpha=0.72, linewidth=0.15, edgecolor="#E0E0E0")
    pc.set_facecolor(pcolors)
    ax.add_collection3d(pc)

    # colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.06)
    cbar.set_label("Wall Temperature (K)", fontsize=11)

    # ===== 2) 目标点 =====
    ax.scatter(targets[:,0], targets[:,1], targets[:,2],
               c="black", marker=".", s=6, alpha=0.6, label="Targets", zorder=3)

    # ===== 3) 散热片高亮 =====
    def _verts(g, hh):
        x0,y0,z0=float(cand[g,0]),float(cand[g,1]),float(cand[g,2]); fid=int(cand[g,3])
        if fid in (0,1):   return [[x0-hh,y0-hh,z0],[x0+hh,y0-hh,z0],[x0+hh,y0+hh,z0],[x0-hh,y0+hh,z0]]
        elif fid in (2,3): return [[x0-hh,y0,z0-hh],[x0+hh,y0,z0-hh],[x0+hh,y0,z0+hh],[x0-hh,y0,z0+hh]]
        else:              return [[x0,y0-hh,z0-hh],[x0,y0+hh,z0-hh],[x0,y0+hh,z0+hh],[x0,y0-hh,z0+hh]]
    h_sink = gs / 2.6
    for g in sol["sink_s_grids"]:
        ax.add_collection3d(Poly3DCollection([_verts(g,h_sink)], alpha=0.85,
                            facecolor="#E65100", edgecolor="#BF360C", linewidth=1.0))
    for g in sol["sink_a_grids"]:
        ax.add_collection3d(Poly3DCollection([_verts(g,h_sink)], alpha=0.75,
                            facecolor="#1565C0", edgecolor="#0D47A1", linewidth=1.0))

    # ===== 4) 传感器(红) + AP(深蓝) =====
    ax.scatter(cand[sensor_ids,0], cand[sensor_ids,1], cand[sensor_ids,2],
               c="#C62828", marker="o", s=140, edgecolors="#8E0000",
               linewidth=1.5, zorder=10)
    ax.scatter(cand[ap_ids,0], cand[ap_ids,1], cand[ap_ids,2],
               c="#0D47A1", marker="^", s=220, edgecolors="#01579B",
               linewidth=1.5, zorder=10)

    # ===== 4b) 热源标记 =====
    hs_path = os.path.join(ROOT, "results/pareto/cr_mode_heat_sources.npz")
    if os.path.exists(hs_path):
        hs = np.load(hs_path)
        hs_coords = hs["heat_source_coords"]
        hs_T = hs["heat_source_T"]
        ax.scatter(hs_coords[:,0], hs_coords[:,1], hs_coords[:,2],
                   c="#FFEA00", marker="*", s=1200, edgecolors="#000000",
                   linewidth=2.0, zorder=20)
        for i, (x,y,z) in enumerate(hs_coords):
            ax.text(x + 0.12, y + 0.12, z + 0.12, f"Heat source {hs_T[i]:.0f}K", fontsize=9, color="#000000", zorder=21, bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="#FFB300"))

    # ===== 5) 连接线 =====
    link_count = 0
    for si in sensor_ids:
        for aj in ap_ids:
            if sol["c"][si, aj] == 1:
                ax.plot([cand[si,0],cand[aj,0]],[cand[si,1],cand[aj,1]],[cand[si,2],cand[aj,2]],
                        ":", color="#BDBDBD", linewidth=0.6, alpha=0.4, zorder=1)
                link_count += 1

    # ===== 6) 轻量边框（只画12条棱） =====
    edges = [
        (0,0,0, Lx,0,0), (0,Ly,0, Lx,Ly,0), (0,0,Lz, Lx,0,Lz), (0,Ly,Lz, Lx,Ly,Lz),
        (0,0,0, 0,Ly,0), (Lx,0,0, Lx,Ly,0), (0,0,Lz, 0,Ly,Lz), (Lx,0,Lz, Lx,Ly,Lz),
        (0,0,0, 0,0,Lz), (Lx,0,0, Lx,0,Lz), (0,Ly,0, 0,Ly,Lz), (Lx,Ly,0, Lx,Ly,Lz),
    ]
    for x1,y1,z1,x2,y2,z2 in edges:
        ax.plot([x1,x2],[y1,y2],[z1,z2], "k-", linewidth=0.5, alpha=0.25)

    ax.set_xlim(0,Lx); ax.set_ylim(0,Ly); ax.set_zlim(0,Lz)
    ax.set_xlabel("X (m)", fontsize=11, labelpad=8)
    ax.set_ylabel("Y (m)", fontsize=11, labelpad=8)
    ax.set_zlabel("Z (m)", fontsize=11, labelpad=8)
    # 等比例
    ax.set_box_aspect([1.0, Ly/Lx if Lx else 1.0, Lz/Lx if Lx else 1.0])

    # ===== 图例（无重叠，单列右侧）=====
    custom_handles = [
        ax.scatter([],[], c="black", marker=".", s=20, alpha=0.6, label="Target"),
        ax.scatter([],[], c="#C62828", marker="o", s=60, label=f"Sensor ({ns})"),
        ax.scatter([],[], c="#0D47A1", marker="^", s=60, label=f"AP ({na})"),
        ax.scatter([],[], c="#FFEA00", marker="*", s=160, edgecolors="#000000", label="Heat source"),
        LegendPatch(facecolor="#E65100", edgecolor="#BF360C", label="Sensor heatsink"),
        LegendPatch(facecolor="#1565C0", edgecolor="#0D47A1", label="AP heatsink"),
    ]
    ax.legend(handles=custom_handles, loc="upper right", fontsize=9, ncol=1,
              bbox_to_anchor=(1.12, 0.96))

    # ===== 简洁信息框 =====
    info = (
        f"{mode_label}\n"
        f"Coverage = {sol['coverage']:.4f}    Rsum = {sol['throughput']/1e6:.1f} Mbps\n"
        f"Sensors = {ns}    APs = {na}"
    )
    ax.text2D(0.02, 0.985, info, transform=ax.transAxes, fontsize=10,
              verticalalignment="top", fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="#BDBDBD"))

    ax.set_title("CR-MODE Deployment State", fontsize=14, fontweight="bold", pad=12)
    ax.view_init(elev=30, azim=-60)

    plt.subplots_adjust(right=0.82)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Pareto 解部署状态 3D 可视化")
    parser.add_argument("--npz", default="results/pareto/cr_mode.npz")
    parser.add_argument("--ctx", default="results/pareto/cr_mode_context.npz")
    parser.add_argument("--mode", default="knee",
                        choices=["highest_coverage","highest_rsum","knee","index"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    npz_path = os.path.join(ROOT, args.npz) if not os.path.isabs(args.npz) else args.npz
    ctx_path = os.path.join(ROOT, args.ctx) if not os.path.isabs(args.ctx) else args.ctx

    if not os.path.exists(npz_path) or not os.path.exists(ctx_path):
        print("ERROR: run 'python main.py' first.")
        sys.exit(1)

    pareto_data, ctx_data = load_data(npz_path, ctx_path)
    sol_idx, mode_label = select_pareto_solution(pareto_data, args.mode, args.index)
    cov_val = float(pareto_data["objectives"][sol_idx, 0])
    rsum_val = float(pareto_data["objectives"][sol_idx, 1])
    print(f"Selected: {mode_label} — Coverage={cov_val:.4f}, Rsum={rsum_val/1e6:.1f} Mbps")

    out_name = args.out or (f"results/figures/selected_solution_"
        f"{mode_label.replace('=','_').replace(' ','_')}_3d.png")
    plot_3d_deployment(pareto_data, sol_idx, ctx_data, mode_label,
                       os.path.join(ROOT, out_name))


if __name__ == "__main__":
    main()



