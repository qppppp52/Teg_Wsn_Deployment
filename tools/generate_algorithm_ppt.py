from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import win32com.client


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "experiments" / "small_center_heat_compare" / "20260713_113757"
ASSET_DIR = ROOT / ".test_outputs" / "algorithm_ppt_assets"
ASSET_DIR.mkdir(parents=True, exist_ok=True)

SLIDE_W = 960
SLIDE_H = 540

COLORS = {
    "ink": "18323F",
    "muted": "5F6B72",
    "paper": "F6F8F7",
    "white": "FFFFFF",
    "teal": "087E8B",
    "green": "2A9D6F",
    "yellow": "F2C14E",
    "red": "D1495B",
    "blue": "357DED",
    "navy": "203A5F",
    "line": "D7DEDC",
    "soft_teal": "E4F3F2",
    "soft_yellow": "FFF5D6",
    "soft_red": "FCE8EB",
    "soft_blue": "E8F0FD",
}


def rgb(value: str) -> int:
    value = value.lstrip("#")
    r, g, b = (int(value[i : i + 2], 16) for i in (0, 2, 4))
    return r + (g << 8) + (b << 16)


def _font(text_range, size=18, color="ink", bold=False, name="Microsoft YaHei"):
    text_range.Font.Name = name
    text_range.Font.NameFarEast = name
    text_range.Font.Size = size
    text_range.Font.Bold = -1 if bold else 0
    text_range.Font.Color.RGB = rgb(COLORS[color])


def add_text(slide, text, x, y, w, h, size=18, color="ink", bold=False,
             align=1, valign=1, margin=8, fill=None, line=None, radius=False):
    shape_type = 5 if radius else 1
    shape = slide.Shapes.AddShape(shape_type, x, y, w, h)
    shape.Fill.Visible = -1 if fill else 0
    if fill:
        shape.Fill.ForeColor.RGB = rgb(COLORS[fill])
        shape.Fill.Transparency = 0
    shape.Line.Visible = -1 if line else 0
    if line:
        shape.Line.ForeColor.RGB = rgb(COLORS[line])
        shape.Line.Weight = 1
    tf = shape.TextFrame
    tf.MarginLeft = margin
    tf.MarginRight = margin
    tf.MarginTop = margin
    tf.MarginBottom = margin
    tf.WordWrap = -1
    tf.VerticalAnchor = 3 if valign in (2, 3) else 1
    tf.TextRange.Text = text
    tf.TextRange.ParagraphFormat.Alignment = align
    _font(tf.TextRange, size, color, bold)
    return shape


def add_line(slide, x1, y1, x2, y2, color="line", width=1.5, arrow=False):
    line = slide.Shapes.AddLine(x1, y1, x2, y2)
    line.Line.ForeColor.RGB = rgb(COLORS[color])
    line.Line.Weight = width
    if arrow:
        line.Line.EndArrowheadStyle = 3
    return line


def add_title(slide, title, subtitle=None, section=None):
    add_text(slide, title, 42, 24, 830, 46, size=26, bold=True, margin=0)
    add_line(slide, 42, 76, 918, 76, color="line", width=1)
    if subtitle:
        add_text(slide, subtitle, 44, 83, 820, 25, size=12, color="muted", margin=0)
    if section:
        add_text(slide, section, 842, 26, 76, 26, size=11, color="white",
                 bold=True, align=2, valign=2, margin=3, fill="teal", radius=True)


def add_footer(slide, page, note=None):
    if note:
        add_text(slide, "讲解提示｜" + note, 42, 502, 800, 20, size=10,
                 color="muted", margin=0)
    add_text(slide, f"{page:02d}", 875, 500, 43, 22, size=11,
             color="muted", bold=True, align=3, margin=0)


def add_bullets(slide, items, x, y, w, h, size=17, color="ink",
                bullet_color="teal", gap=7):
    if not items:
        return
    row_h = h / len(items)
    for idx, item in enumerate(items):
        yy = y + idx * row_h
        add_text(slide, "•", x, yy + 1, 20, row_h - gap, size=size + 2,
                 color=bullet_color, bold=True, margin=0)
        add_text(slide, item, x + 24, yy, w - 24, row_h - gap, size=size,
                 color=color, margin=0)


def add_metric(slide, value, label, x, y, w, accent="teal", detail=None):
    add_text(slide, value, x, y, w, 48, size=22, color=accent, bold=True,
             align=2, valign=2, margin=2, fill="white", line="line", radius=True)
    add_text(slide, label, x, y + 51, w, 24, size=12, color="muted",
             bold=True, align=2, margin=0)
    if detail:
        add_text(slide, detail, x, y + 75, w, 27, size=10, color="muted",
                 align=2, margin=0)


def add_table(slide, headers, rows, x, y, w, h, widths=None, font_size=12):
    table_shape = slide.Shapes.AddTable(len(rows) + 1, len(headers), x, y, w, h)
    table = table_shape.Table
    if widths:
        for i, width in enumerate(widths):
            table.Columns(i + 1).Width = width
    for c, header in enumerate(headers, start=1):
        cell = table.Cell(1, c).Shape
        cell.Fill.ForeColor.RGB = rgb(COLORS["navy"])
        cell.TextFrame.TextRange.Text = str(header)
        cell.TextFrame.TextRange.ParagraphFormat.Alignment = 2
        cell.TextFrame.VerticalAnchor = 3
        _font(cell.TextFrame.TextRange, font_size, "white", True)
    for r, row in enumerate(rows, start=2):
        for c, value in enumerate(row, start=1):
            cell = table.Cell(r, c).Shape
            cell.Fill.ForeColor.RGB = rgb(COLORS["white" if r % 2 == 0 else "paper"])
            cell.TextFrame.TextRange.Text = str(value)
            cell.TextFrame.TextRange.ParagraphFormat.Alignment = 1 if c == 1 else 2
            cell.TextFrame.VerticalAnchor = 3
            cell.TextFrame.MarginLeft = 5
            cell.TextFrame.MarginRight = 5
            _font(cell.TextFrame.TextRange, font_size, "ink", c == 1)
    return table_shape


def add_flow(slide, labels, x, y, total_w, box_h=62, colors=None, size=15):
    gap = 24
    box_w = (total_w - gap * (len(labels) - 1)) / len(labels)
    for idx, label in enumerate(labels):
        xx = x + idx * (box_w + gap)
        fill = colors[idx] if colors else ("soft_teal" if idx % 2 == 0 else "soft_blue")
        add_text(slide, label, xx, y, box_w, box_h, size=size, bold=True,
                 align=2, valign=3, fill=fill, line="line", radius=True)
        if idx < len(labels) - 1:
            add_line(slide, xx + box_w + 3, y + box_h / 2,
                     xx + box_w + gap - 3, y + box_h / 2,
                     color="teal", width=2, arrow=True)


def add_picture(slide, path, x, y, w, h):
    path = Path(path)
    if not path.exists():
        add_text(slide, f"图像不存在\n{path.name}", x, y, w, h, size=14,
                 color="red", align=2, valign=3, fill="soft_red", line="red")
        return None
    return slide.Shapes.AddPicture(str(path.resolve()), 0, -1, x, y, w, h)


def new_slide(pres, title=None, subtitle=None, section=None, note=None):
    slide = pres.Slides.Add(pres.Slides.Count + 1, 12)
    slide.FollowMasterBackground = 0
    slide.Background.Fill.ForeColor.RGB = rgb(COLORS["paper"])
    if title:
        add_title(slide, title, subtitle, section)
    add_footer(slide, pres.Slides.Count, note)
    return slide


def load_summary():
    rows = list(csv.DictReader((RESULT_ROOT / "summary_all_algorithms.csv").open(encoding="utf-8")))
    result = {}
    for alg in ("cr_mode", "dqn_cr_mode", "drl_init_cr_mode"):
        selected = [row for row in rows if row["algorithm"] == alg]
        result[alg] = {
            "hv": np.mean([float(row["final_HV"]) for row in selected]),
            "coverage": np.mean([float(row["best_coverage"]) for row in selected]),
            "rsum": np.mean([float(row["best_rsum_capacity"]) for row in selected]),
            "pareto": np.mean([float(row["pareto_count"]) for row in selected]),
            "runtime": np.mean([float(row["runtime_seconds"]) for row in selected]),
        }
    return result


def make_result_chart(summary):
    labels = ["CR-MODE", "DQN-CR-MODE", "DRL-Init-CR-MODE"]
    keys = ["cr_mode", "dqn_cr_mode", "drl_init_cr_mode"]
    colors = ["#5F6B72", "#087E8B", "#D1495B"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    values = [
        [summary[k]["hv"] for k in keys],
        [summary[k]["coverage"] for k in keys],
        [summary[k]["rsum"] / 1e6 for k in keys],
    ]
    titles = ["Mean final HV", "Mean best Coverage", "Mean best Rsum (Mbit/s)"]
    for ax, vals, title in zip(axes, values, titles):
        bars = ax.bar(labels, vals, color=colors, width=0.62)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="x", labelrotation=18, labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}",
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out = ASSET_DIR / "formal_results.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def make_dqn_action_chart():
    path = RESULT_ROOT / "dqn_action_reward_summary.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    names = ["balanced", "exploration_high_F", "exploitation_low_F", "energy_first",
             "link_first", "ap_load_first", "sink_first", "rsum_capacity_priority",
             "diversity_boost", "conservative_repair"]
    counts = {name: [] for name in names}
    seeds = []
    for row in rows:
        seeds.append("seed " + row["seed"])
        parsed = Counter()
        for item in row["action_usage_distribution"].split(";"):
            if not item:
                continue
            name, fraction = item.split(":", 1)
            parsed[name] = int(fraction.split("/")[0])
        for name in names:
            counts[name].append(parsed[name])
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    bottom = np.zeros(len(seeds))
    palette = plt.cm.tab10(np.linspace(0, 1, len(names)))
    for color, name in zip(palette, names):
        vals = np.asarray(counts[name])
        if np.any(vals):
            ax.bar(seeds, vals, bottom=bottom, label=name, color=color)
            bottom += vals
    ax.set_ylabel("Selected generations / 60")
    ax.set_ylim(0, 64)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = ASSET_DIR / "dqn_action_distribution.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def make_ablation_chart():
    path = ROOT / ".test_outputs" / "dqn_action_ablation" / "summary_means.csv"
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    labels = [row["policy"] for row in rows]
    hv = [float(row["mean_final_hv"]) for row in rows]
    rsum = [float(row["mean_best_rsum_capacity"]) / 1e6 for row in rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.3))
    colors = ["#087E8B", "#F2C14E", "#357DED", "#D1495B"]
    axes[0].bar(x, hv, color=colors)
    axes[0].set_title("Mean final HV")
    axes[1].bar(x, rsum, color=colors)
    axes[1].set_title("Mean best Rsum (Mbit/s)")
    for ax, vals in zip(axes, [hv, rsum]):
        ax.set_xticks(x, labels, rotation=18, fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        for i, value in enumerate(vals):
            ax.text(i, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    out = ASSET_DIR / "dqn_ablation.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def make_source_mix_chart():
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    vals = [18, 6, 6]
    labels = ["PPO / DRL 60%", "Heuristic 20%", "Random 20%"]
    ax.pie(vals, labels=labels, autopct="%1.0f%%", startangle=90,
           colors=["#D1495B", "#087E8B", "#F2C14E"],
           wedgeprops={"width": 0.42, "edgecolor": "white"}, textprops={"fontsize": 8})
    ax.text(0, 0, "Gen0\nNP=30", ha="center", va="center", weight="bold", fontsize=12)
    out = ASSET_DIR / "source_mix.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return out


def make_reward_scale_chart():
    labels = ["Gen 1–2", "Gen 3–10", "Gen 11–30", "Gen 31–60"]
    seed42 = [0.3223, 0.00322, 0.00118, 0.00118]
    seed43 = [0.3082, 0.00454, 0.00077, 0.00049]
    seed44 = [0.1797, 0.00223, 0.00152, 0.00117]
    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    x = np.arange(len(labels))
    for vals, name, color in zip([seed42, seed43, seed44], ["seed 42", "seed 43", "seed 44"],
                                 ["#087E8B", "#357DED", "#D1495B"]):
        ax.plot(x, vals, marker="o", linewidth=2, label=name, color=color)
    ax.set_yscale("log")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean absolute reward (log scale)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = ASSET_DIR / "reward_scale.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def build_presentation(output_path: Path):
    summary = load_summary()
    result_chart = make_result_chart(summary)
    action_chart = make_dqn_action_chart()
    ablation_chart = make_ablation_chart()
    source_mix = make_source_mix_chart()
    reward_chart = make_reward_scale_chart()

    app = win32com.client.DispatchEx("PowerPoint.Application")
    app.Visible = True
    pres = app.Presentations.Add()
    pres.PageSetup.SlideWidth = SLIDE_W
    pres.PageSetup.SlideHeight = SLIDE_H

    # 1. Cover
    slide = new_slide(pres, note="先统一名称：项目中的 DRL 控制版本实际是 DQN-CR-MODE。")
    add_text(slide, "DQN-CR-MODE 与\nDRL-Init-CR-MODE", 58, 82, 560, 145,
             size=34, color="ink", bold=True, margin=0)
    add_text(slide, "面向 TEG 供能无线传感器网络联合部署的深度强化学习增强多目标优化方案",
             60, 242, 600, 62, size=19, color="muted", margin=0)
    add_text(slide, "导师汇报版｜算法流程 · 状态 · 动作 · 奖励 · 训练边界 · 公平对比 · 实验证据",
             60, 323, 635, 35, size=13, color="teal", bold=True, margin=0)
    add_text(slide, "研究主线", 704, 88, 175, 30, size=13, color="white", bold=True,
             align=2, valign=2, fill="navy", radius=True)
    add_text(slide, "空气传热温度场\n→ TEG 能量采集\n→ 传感器/AP 联合部署\n→ Coverage 与 Rsum_capacity\n→ 多约束 Pareto 搜索",
             704, 128, 175, 230, size=17, color="ink", bold=True,
             align=2, valign=3, fill="white", line="line", radius=True)
    add_text(slide, "基于当前项目代码与 2026-07-13 正式实验结果", 60, 438, 600, 28,
             size=12, color="muted", margin=0)

    # 2. Agenda
    slide = new_slide(pres, "汇报结构", section="总览", note="先讲共同问题，再讲两个 DRL 介入位置。")
    sections = [
        ("01", "研究问题与共同内核", "建模目标、约束、CR-MODE"),
        ("02", "DQN-CR-MODE", "每代自适应选择搜索/修复动作"),
        ("03", "DRL-Init-CR-MODE", "PPO 只负责构造高质量 Gen0"),
        ("04", "对比、公平性与实验", "预算、边界、指标、结果"),
        ("05", "问题诊断与改进路线", "证据链、局限、后续工作"),
    ]
    for i, (num, title, desc) in enumerate(sections):
        y = 105 + i * 72
        add_text(slide, num, 62, y, 50, 48, size=20, color="white", bold=True,
                 align=2, valign=3, fill="teal", radius=True)
        add_text(slide, title, 132, y, 280, 25, size=18, bold=True, margin=0)
        add_text(slide, desc, 132, y + 29, 520, 22, size=13, color="muted", margin=0)
        add_line(slide, 132, y + 58, 870, y + 58)

    # 3. Naming
    slide = new_slide(pres, "名称与研究边界先统一", section="总览",
                      note="避免把 DQN 控制和 PPO 初始化都笼统叫成 DRL-CR-MODE。")
    add_text(slide, "DQN-CR-MODE", 62, 120, 380, 58, size=25, color="white", bold=True,
             align=2, valign=3, fill="teal", radius=True)
    add_text(slide, "DQN 在每一代读取种群状态，选择一组 F、CR、变异策略、修复顺序和功率策略。\n\nDRL 持续介入 Gen 1–60。",
             62, 190, 380, 190, size=17, fill="white", line="line", radius=True)
    add_text(slide, "DRL-Init-CR-MODE", 518, 120, 380, 58, size=25, color="white", bold=True,
             align=2, valign=3, fill="red", radius=True)
    add_text(slide, "PPO 顺序选择候选点及角色，只生成初始种群 Gen0。\n\n进入 Gen 1–60 后完全回到与基线一致的 CR-MODE 内核。",
             518, 190, 380, 190, size=17, fill="white", line="line", radius=True)
    add_text(slide, "核心区别不是“用了哪种神经网络”，而是 DRL 在优化流程中的介入位置不同。",
             128, 413, 704, 48, size=18, color="navy", bold=True,
             align=2, valign=3, fill="soft_yellow", line="yellow", radius=True)

    # 4. Problem
    slide = new_slide(pres, "研究问题：热环境约束下的传感器与 AP 联合部署", section="共同内核",
                      note="强调这是多目标、离散-连续混合、强约束问题。")
    add_flow(slide, ["内部热源\n空气传热", "内表面温度\n与 P_grid", "传感器/AP\n部署与关联", "Coverage 与\nRsum_capacity"],
             60, 120, 840, box_h=72, colors=["soft_red", "soft_yellow", "soft_teal", "soft_blue"])
    add_bullets(slide, [
        "决策包含：候选位置选择、传感器/AP 角色、连接关系、发射功率与散热资源。",
        "两个正式目标：最大化覆盖率 Coverage；最大化有效传感器–AP 链路理论 Shannon 容量和 Rsum_capacity。",
        "约束相互耦合：部署数量、连通性、AP 容量、能量自持、散热片冲突、AP 服务与 sink 风险。",
        "搜索空间高度非凸、离散且存在大量不可行解，因此需要修复算子与约束支配。",
    ], 78, 235, 805, 210, size=15)

    # 5. Physical scene
    slide = new_slide(pres, "小场景物理输入：中心热源与内表面温度场", section="共同内核",
                      note="温度场是算法共同输入，不属于任何一个算法的独有结果。")
    temp_img = RESULT_ROOT / "scene_preprocess" / "shared" / "figures" / "temperature_faces.png"
    add_picture(slide, temp_img, 55, 112, 540, 340)
    add_metric(slide, "96", "内表面候选点", 640, 120, 110, "teal", "每面 4×4")
    add_metric(slide, "40", "覆盖目标点", 780, 120, 110, "blue")
    add_metric(slide, "24.68–38.37°C", "表面温度范围", 640, 235, 250, "red")
    add_metric(slide, "33.68°C", "顶面平均", 640, 350, 110, "red")
    add_metric(slide, "25.71°C", "底面平均", 780, 350, 110, "blue")

    # 6. Objectives and constraints
    slide = new_slide(pres, "目标函数与约束：所有算法共享同一评价器", section="共同内核",
                      note="这是公平对比最重要的边界：目标和约束绝不随算法改变。")
    add_text(slide, "目标 1", 55, 112, 115, 32, size=14, color="white", bold=True,
             align=2, valign=2, fill="teal", radius=True)
    add_text(slide, "max Coverage = 被至少一个已部署传感器覆盖的目标点比例",
             55, 150, 400, 70, size=18, bold=True, align=2, valign=3,
             fill="white", line="line", radius=True)
    add_text(slide, "目标 2", 505, 112, 115, 32, size=14, color="white", bold=True,
             align=2, valign=2, fill="red", radius=True)
    add_text(slide, "max Rsum_capacity = Σ B·log₂(1 + SNRᵢⱼ)\n仅统计有效传感器–AP 连接",
             505, 150, 400, 70, size=18, bold=True, align=2, valign=3,
             fill="white", line="line", radius=True)
    constraints = [
        ("部署", "传感器/AP 数量与候选合法性"), ("链路", "连接、距离、SNR 与可达性"),
        ("容量", "AP 服务数量/业务承载限制"), ("能量", "TEG 采能覆盖电路与发射功耗"),
        ("散热", "邻域散热片数量及冲突"), ("服务", "每个传感器必须获得 AP 服务"),
    ]
    for i, (name, desc) in enumerate(constraints):
        x = 55 + (i % 3) * 285
        y = 265 + (i // 3) * 92
        add_text(slide, name, x, y, 70, 52, size=15, color="white", bold=True,
                 align=2, valign=3, fill="navy", radius=True)
        add_text(slide, desc, x + 78, y, 195, 52, size=13, valign=3,
                 fill="white", line="line", radius=True)

    # 7. Encoding
    slide = new_slide(pres, "个体编码、解码与统一评价", section="共同内核",
                      note="DRL 操作的最终对象仍会转换为同一种 Individual/Solution。")
    add_flow(slide, ["连续优先级向量\nρs, ρa", "解码部署\nx, y, c, ptx", "约束评价\nCV 分量", "目标评价\nCoverage, Rsum"],
             55, 118, 850, box_h=70)
    add_bullets(slide, [
        "优先级向量便于差分变异与交叉；解码后得到离散部署、连接、功率和散热决策。",
        "每个候选解先评价约束；若不可行，则进入最多 5 次的修复循环。",
        "评价输出同时包含：feasible、总 CV、分项压力、修复次数、Coverage 与 Rsum_capacity。",
        "DQN 和 PPO 读取的状态/奖励均来自这套统一物理模型与评价器。",
    ], 75, 235, 810, 200, size=16)

    # 8. Repair
    slide = new_slide(pres, "约束修复：为什么是算法设计的关键一层", section="共同内核",
                      note="修复不是修改目标，而是把不可行解尽量映射回可行域。")
    add_text(slide, "不可行解", 58, 145, 140, 58, size=19, color="white", bold=True,
             align=2, valign=3, fill="red", radius=True)
    add_line(slide, 202, 174, 272, 174, color="red", width=2, arrow=True)
    add_text(slide, "按策略顺序修复\ndeploy / link / capacity / energy / service / sink",
             275, 125, 390, 98, size=17, bold=True, align=2, valign=3,
             fill="soft_yellow", line="yellow", radius=True)
    add_line(slide, 669, 174, 738, 174, color="teal", width=2, arrow=True)
    add_text(slide, "可行或 CV 更低", 742, 145, 160, 58, size=18, color="white", bold=True,
             align=2, valign=3, fill="green", radius=True)
    add_bullets(slide, [
        "保留修复过程中 CV 最低的状态；可行时立即停止。",
        "重复状态或连续无改善时提前停止，避免无意义循环。",
        "DQN 动作会改变修复顺序与功率策略；DRL-Init 只产生部署优先级，之后仍使用公共修复器。",
        "修复次数既是计算成本指标，也进入强化学习状态或奖励。",
    ], 90, 280, 780, 165, size=16)

    # 9. Shared CR-MODE
    slide = new_slide(pres, "CR-MODE 共同搜索内核", section="共同内核",
                      note="两个增强算法都不能绕开这条主循环。")
    add_flow(slide, ["当前种群\nNP=30", "差分变异\nF + strategy", "交叉\nCR", "评价与修复\n30 trials", "约束 Pareto\n环境选择"],
             38, 118, 884, box_h=74, size=14)
    add_bullets(slide, [
        "每代每个父代生成 1 个 trial，因此在线评价数固定为 NP×Tmax，加上 Gen0。",
        "可行解优先于不可行解；不可行解之间比较 CV；可行解之间比较 Coverage 与 Rsum_capacity。",
        "同一非支配层采用拥挤距离，保持 Pareto 前沿分布。",
        "外部 Pareto Archive 保存跨代非支配可行解，并去除目标值完全重复的解。",
    ], 72, 242, 820, 190, size=16)
    add_text(slide, "正式小场景：30 + 30×60 = 1830 次在线评价 / seed",
             230, 444, 500, 38, size=18, color="navy", bold=True,
             align=2, valign=3, fill="soft_blue", line="blue", radius=True)

    # 10. Why DRL
    slide = new_slide(pres, "为什么引入强化学习：两个不同的切入点", section="共同内核",
                      note="一个优化搜索策略，一个优化初始种群。")
    add_text(slide, "痛点 A：搜索阶段静态参数", 62, 116, 390, 42, size=19, color="white",
             bold=True, align=2, valign=3, fill="teal", radius=True)
    add_bullets(slide, [
        "不同代的可行率、约束压力和停滞程度不同。",
        "固定 F、CR、变异策略和修复顺序难以兼顾探索与开发。",
        "→ 用 DQN 每代选择控制动作。",
    ], 72, 180, 360, 180, size=16)
    add_text(slide, "痛点 B：Gen0 质量与可行性", 508, 116, 390, 42, size=19, color="white",
             bold=True, align=2, valign=3, fill="red", radius=True)
    add_bullets(slide, [
        "随机/混合初始化常产生大量不可行个体。",
        "早期代数被迫用于修复，而不是推进 Pareto 前沿。",
        "→ 用 PPO 学习顺序部署，生成部分高质量 Gen0。",
    ], 518, 180, 360, 180, size=16, bullet_color="red")
    add_text(slide, "DQN：在线控制搜索动作", 105, 400, 300, 42, size=17, color="teal", bold=True,
             align=2, valign=3, fill="soft_teal", radius=True)
    add_text(slide, "PPO：离线/前置生成初始种群", 555, 400, 300, 42, size=17, color="red", bold=True,
             align=2, valign=3, fill="soft_red", radius=True)

    # 11 DQN positioning
    slide = new_slide(pres, "DQN-CR-MODE：每代选择一个搜索控制动作", section="DQN",
                      note="DQN 不直接输出部署解，而是控制 CR-MODE 如何生成下一代。")
    add_flow(slide, ["种群状态 sₜ", "DQN Q(s,a)", "选择动作 aₜ", "执行 1 代\nCR-MODE", "得到 sₜ₊₁,rₜ"],
             40, 128, 880, box_h=76, colors=["soft_blue", "soft_teal", "soft_yellow", "soft_teal", "soft_blue"])
    add_bullets(slide, [
        "控制频率：generation-level，每一代只选一次动作，作用于该代全部 30 个 trial。",
        "训练阶段：ε-greedy 探索、经验回放、Double-DQN 更新。",
        "正式评估：加载最佳 checkpoint，epsilon=0，确定性 argmax，不收集经验、不更新梯度。",
        "研究问题：状态自适应的算子选择是否优于固定 CR-MODE 策略。",
    ], 78, 260, 805, 190, size=16)

    # 12 DQN state
    slide = new_slide(pres, "DQN 状态空间：19 维种群级状态", section="DQN",
                      note="状态既要描述可行性，也要描述目标进展和搜索阶段。")
    groups = [
        ("可行性 3", "CV_mean_norm\nCV_min_norm\nFR"),
        ("目标与进展 6", "HV_norm, ΔHV\nBest/Mean Coverage\nBest/Mean Rsum"),
        ("多样性 1", "Objective-space\nDiversity"),
        ("约束压力 6", "deploy, link, capacity\nenergy, sink, service"),
        ("修复与阶段 3", "repair success\nmean repair iter\ngeneration progress"),
    ]
    for i, (title, body) in enumerate(groups):
        x = 45 + i * 181
        add_text(slide, title, x, 122, 165, 36, size=14, color="white", bold=True,
                 align=2, valign=3, fill=["navy", "teal", "yellow", "red", "blue"][i], radius=True)
        add_text(slide, body, x, 166, 165, 185, size=15, bold=True, align=2, valign=3,
                 fill="white", line="line", radius=True)
    add_text(slide, "设计理由：仅使用当前种群和档案的可观测统计量，避免把完整个体向量直接输入网络导致维度随场景变化。",
             92, 390, 776, 58, size=17, color="navy", bold=True,
             align=2, valign=3, fill="soft_blue", line="blue", radius=True)

    # 13 DQN actions 0-4
    slide = new_slide(pres, "DQN 动作空间（1/2）：搜索强度与基础约束策略", section="DQN",
                      note="每个动作是一个完整的算子包，不只是修改 F。")
    rows = [
        ["0 balanced", "0.50", "0.80", "rand/1", "标准修复", "balanced"],
        ["1 exploration_high_F", "0.80", "0.90", "rand/1", "标准修复", "balanced"],
        ["2 exploitation_low_F", "0.35", "0.60", "current-to-best/1", "标准修复", "balanced"],
        ["3 energy_first", "0.45", "0.70", "rand/1", "energy→sink→link…", "conservative"],
        ["4 link_first", "0.55", "0.80", "rand/1", "link→capacity→energy…", "balanced"],
    ]
    add_table(slide, ["动作", "F", "CR", "变异", "修复顺序", "功率策略"], rows,
              42, 118, 876, 300, widths=[190, 55, 55, 145, 270, 150], font_size=11)
    add_text(slide, "探索/开发不是抽象标签：它们通过 F、CR 和变异策略直接改变 trial 在决策空间中的步长与方向。",
             100, 430, 760, 54, size=14, color="navy", bold=True,
             align=2, valign=3, fill="soft_yellow", line="yellow", radius=True)

    # 14 DQN actions 5-9
    slide = new_slide(pres, "DQN 动作空间（2/2）：容量、散热、多样性与保守修复", section="DQN")
    rows = [
        ["5 ap_load_first", "0.50", "0.75", "rand/1", "capacity→link→energy…", "balanced"],
        ["6 sink_first", "0.45", "0.65", "rand/1", "sink→energy→link…", "sink_limited"],
        ["7 rsum_capacity_priority", "0.60", "0.85", "best/1", "link→capacity→energy…", "Rsum priority"],
        ["8 diversity_boost", "0.90", "0.95", "rand/2", "标准修复", "balanced"],
        ["9 conservative_repair", "0.30", "0.50", "current-to-best/1", "energy→capacity→link…", "conservative"],
    ]
    add_table(slide, ["动作", "F", "CR", "变异", "修复顺序", "功率策略"], rows,
              42, 118, 876, 300, widths=[190, 55, 55, 145, 270, 150], font_size=11)
    add_text(slide, "动作设计原则：把“搜索算子”和“约束修复知识”绑定，使 DQN 的选择能真正影响可行域进入速度和 Pareto 前沿推进。",
             100, 430, 760, 54, size=14, color="navy", bold=True,
             align=2, valign=3, fill="soft_teal", line="teal", radius=True)

    # 15 action rationale
    slide = new_slide(pres, "动作何时应该被选择：设计意图", section="DQN",
                      note="这是理论预期，不等于网络一定已经学到。")
    rows = [
        ["低可行率/高 CV", "energy_first / conservative / link_first", "优先消除主导约束"],
        ["已经可行但 HV 停滞", "exploration_high_F / diversity_boost", "扩大搜索半径，跳出局部区域"],
        ["前沿稳定、需要精修", "exploitation_low_F", "向当前优秀解附近收敛"],
        ["容量或链路压力高", "ap_load_first / link_first", "先修复服务与连接瓶颈"],
        ["希望提高吞吐极值", "rsum_capacity_priority", "best/1 + 容量优先功率策略"],
    ]
    add_table(slide, ["状态特征", "候选动作", "为什么"], rows, 55, 120, 850, 300,
              widths=[230, 330, 290], font_size=12)
    add_text(slide, "DQN 的价值在于按状态切换，而不是证明某个单一动作永远最好。",
             200, 440, 560, 42, size=18, color="red", bold=True,
             align=2, valign=3, fill="soft_red", line="red", radius=True)

    # 16 mask
    slide = new_slide(pres, "Action Mask：把明显不合适的动作排除在决策外", section="DQN")
    add_flow(slide, ["读取 FR/CV", "识别主导压力", "构造合法动作集合", "DQN 在集合内 argmax"],
             60, 120, 840, box_h=68)
    rows = [
        ["FR < 0.2", "balanced / energy / link / AP-load / sink / conservative"],
        ["energy 压力 > 0.4", "balanced / energy_first / conservative / sink_first"],
        ["link 压力 > 0.4", "balanced / link_first / AP-load / Rsum priority"],
        ["已可行且 CV 很低", "额外开放 Rsum / exploration / diversity / exploitation"],
        ["HV 连续停滞", "额外开放 diversity_boost / exploration_high_F"],
    ]
    add_table(slide, ["条件", "动作白名单/扩展"], rows, 80, 230, 800, 215,
              widths=[260, 540], font_size=12)
    add_text(slide, "安全兜底：任何规则都不能产生空 mask；优先保留 balanced。",
             245, 458, 470, 30, size=14, color="muted", bold=True, align=2, margin=0)

    # 17 reward
    slide = new_slide(pres, "DQN 奖励函数：可行性、目标进步、多样性与成本", section="DQN",
                      note="奖励是“从第 t 代到第 t+1 代”的增量评价。")
    formula = "rₜ = 0.25·R_CV + 0.20·R_pressure + 0.20·R_FR + 0.25·ΔHV\n      + 0.05·R_obj + 0.05·ΔDiversity − 0.10·R_cost"
    add_text(slide, formula, 88, 112, 784, 96, size=20, color="navy", bold=True,
             align=2, valign=3, fill="white", line="blue", radius=True)
    rows = [
        ["R_CV", "相对 CV 下降", "奖励进入可行域"], ["R_pressure", "分项压力下降", "识别约束瓶颈"],
        ["R_FR", "可行比例增量", "促进整体种群可行"], ["ΔHV", "超体积增量", "直接推进 Pareto 前沿"],
        ["R_obj", "Coverage/Rsum 最优值增量", "照顾两个目标极值"], ["ΔDiversity", "目标空间多样性增量", "避免前沿塌缩"],
        ["R_cost", "平均修复次数", "减少高成本动作"],
    ]
    add_table(slide, ["分量", "定义", "设计目的"], rows, 90, 228, 780, 245,
              widths=[120, 270, 390], font_size=11)

    # 18 reward finding
    slide = new_slide(pres, "DQN 奖励的当前实测问题：后期信号过弱", section="DQN",
                      note="这是当前实现的真实诊断，不应在汇报中回避。")
    add_picture(slide, reward_chart, 55, 118, 540, 300)
    add_bullets(slide, [
        "前 1–2 代 reward 约 0.18–0.32；Gen 11–60 多数只有 10⁻³。",
        "进入可行区后，CV、FR、pressure 三项长期为 0。",
        "Coverage 与 Rsum 的实际单项权重仅 0.025。",
        "结果：Q 网络更像学习“哪种动作的长期负回报较小”。",
    ], 625, 128, 290, 255, size=15, bullet_color="red")
    add_text(slide, "改进方向：配置化权重、阶段平衡、稳定的 reward scaling；但不能改变正式目标定义。",
             100, 438, 760, 43, size=15, color="red", bold=True,
             align=2, valign=3, fill="soft_red", line="red", radius=True)

    # 19 Double DQN
    slide = new_slide(pres, "Double-DQN 网络与训练机制", section="DQN")
    add_flow(slide, ["19 维状态", "MLP\n128→128", "10 个 Q 值", "masked ε-greedy", "Replay Buffer"],
             45, 120, 870, box_h=70)
    add_bullets(slide, [
        "在线网络选择下一动作；目标网络评价该动作，减轻普通 DQN 的 Q 值过估计。",
        "经验回放容量 20,000；至少 64 条经验后开始训练；batch size=32。",
        "γ=0.95，学习率 1e-3；每交互一步更新 1 次；目标网络每 100 个梯度步同步。",
        "状态与奖励使用运行均值/方差归一化；梯度范数裁剪为 5。",
        "ε 从 1.0 在 6000 步内衰减到 0.05，正式评估强制为 0。",
    ], 82, 245, 795, 205, size=15)

    # 20 train/eval
    slide = new_slide(pres, "DQN 训练与正式评估必须严格分离", section="DQN",
                      note="否则 DQN 在线学习时间和随机探索会污染算法公平性。")
    add_text(slide, "训练阶段 train_dqn.py", 60, 115, 390, 44, size=20, color="white",
             bold=True, align=2, valign=3, fill="teal", radius=True)
    add_bullets(slide, [
        "120 个独立训练 seed：1000–1119。",
        "ε-greedy，收集 transition，更新 Q 网络。",
        "每 5 episode 在验证 seed 142/143/144 上评估。",
        "保存 best 与 latest checkpoint。",
    ], 75, 185, 350, 210, size=14)
    add_text(slide, "正式阶段 main.py", 510, 115, 390, 44, size=20, color="white",
             bold=True, align=2, valign=3, fill="red", radius=True)
    add_bullets(slide, [
        "加载 best checkpoint；不允许随机网络回退。",
        "epsilon=0，确定性动作选择。",
        "不调用 observe() / train_step()。",
        "60 代在线优化时间不包含离线训练时间。",
    ], 525, 185, 350, 210, size=14, bullet_color="red")
    add_text(slide, "正式命令：先 train_dqn.py，再 main.py；checkpoint 兼容且配置未变时可复用。",
             128, 430, 704, 48, size=16, color="navy", bold=True,
             align=2, valign=3, fill="soft_yellow", line="yellow", radius=True)

    # 21 checkpoint
    slide = new_slide(pres, "DQN checkpoint 选择机制及局限", section="DQN")
    add_metric(slide, "Episode 95", "当前 best checkpoint", 70, 118, 190, "teal")
    add_metric(slide, "0.4838", "验证 HV 中位数", 285, 118, 190, "blue")
    add_metric(slide, "5637", "gradient steps", 500, 118, 190, "red")
    add_metric(slide, "5700", "interaction steps", 715, 118, 175, "yellow")
    add_bullets(slide, [
        "优点：使用固定验证集和最终 HV 中位数，避免直接采用最后一个模型。",
        "局限：只有 3 个验证 seed，反复用于 24 次 checkpoint 选择，容易对验证 seed 过拟合。",
        "只按 HV 选择，不检查 Coverage、Rsum、Pareto 数量、动作塌缩或相对 CR-MODE 的增益。",
        "checkpoint 哈希只覆盖状态名、动作名和网络维度，未覆盖 reward、动作参数和场景语义。",
    ], 88, 270, 790, 190, size=15)

    # 22 action distribution
    slide = new_slide(pres, "当前 DQN 行为证据：energy_first 高度占优", section="DQN",
                      note="action mask 每代允许 10 个动作，因此不是 mask 强制。")
    add_picture(slide, action_chart, 55, 108, 570, 320)
    add_metric(slide, "169 / 180", "energy_first 胜出", 670, 128, 210, "red")
    add_metric(slide, "0.0258", "平均 top-2 Q 差", 670, 245, 210, "teal")
    add_metric(slide, "10 / 10", "平均合法动作数", 670, 362, 210, "blue")
    add_text(slide, "结论：策略确实形成 Q 值偏置；是否是合理最优偏好，必须通过固定动作消融判断。",
             130, 452, 700, 35, size=15, color="navy", bold=True, align=2, margin=0)

    # 23 ablation
    slide = new_slide(pres, "DQN 动作消融：当前策略没有稳定优于简单控制", section="DQN",
                      note="同 Gen0、同 seed、同 1830 次评价；仅替换每代动作选择器。")
    if ablation_chart:
        add_picture(slide, ablation_chart, 55, 118, 565, 300)
    rows = [
        ["Frozen DQN", "0.4572", "155.477M", "6.0"],
        ["Always energy_first", "0.4675", "159.958M", "7.67"],
        ["Always balanced", "0.4558", "158.505M", "6.33"],
        ["Random legal", "0.4847", "158.457M", "7.0"],
    ]
    add_table(slide, ["策略", "HV", "Rsum", "Pareto"], rows, 650, 135, 260, 230,
              widths=[105, 50, 65, 40], font_size=9)
    add_text(slide, "3 个 seed 尚不足以作最终统计结论，但足以说明当前 checkpoint 的动作排序需要重新审视。",
             115, 440, 730, 45, size=15, color="red", bold=True,
             align=2, valign=3, fill="soft_red", line="red", radius=True)

    # 24 DRL init positioning
    slide = new_slide(pres, "DRL-Init-CR-MODE：PPO 只优化 Gen0", section="DRL-Init",
                      note="研究变量是初始化质量，不是搜索内核。")
    add_flow(slide, ["PPO 训练/加载", "顺序构造部署", "生成混合 Gen0", "统一评价与修复", "标准 CR-MODE\nGen 1–60"],
             40, 125, 880, box_h=78, colors=["soft_red", "soft_red", "soft_yellow", "soft_teal", "soft_blue"])
    add_bullets(slide, [
        "PPO 学习在候选表面上按顺序选择节点及角色。",
        "最终个体仍转换为与 CR-MODE 相同的连续优先级编码。",
        "PPO 不在 Gen 1–60 中选择 F、CR 或变异策略。",
        "因此性能差异可以解释为“更好的初始种群是否帮助后续 Pareto 搜索”。",
    ], 82, 265, 795, 185, size=16, bullet_color="red")

    # 25 Complete pipeline
    slide = new_slide(pres, "DRL-Init 完整流程", section="DRL-Init")
    add_flow(slide, ["构建特征与 mask", "PPO 训练\n1000 episodes", "策略采样候选", "重试+多样性过滤", "18/6/6 混合 Gen0"],
             35, 115, 890, box_h=72, colors=["soft_blue", "soft_red", "soft_red", "soft_yellow", "soft_teal"])
    add_flow(slide, ["Gen0 再评价", "有效性检查", "共同 CR-MODE\n60 generations", "输出证据链"],
             115, 255, 730, box_h=72, colors=["soft_teal", "soft_yellow", "soft_blue", "soft_teal"])
    add_bullets(slide, [
        "候选不足时先增加 DRL 采样尝试，再按规则启用 heuristic/random fallback。",
        "正式运行若 DRL 接受比例不足或 PPO 没有真实更新，DRL_INIT_VALID=false 并按配置失败。",
    ], 115, 382, 730, 90, size=15)

    # 26 Env
    slide = new_slide(pres, "PPO 环境：把部署转化为顺序决策过程", section="DRL-Init")
    add_text(slide, "Episode 开始", 55, 135, 150, 52, size=18, color="white", bold=True,
             align=2, valign=3, fill="red", radius=True)
    add_line(slide, 210, 161, 280, 161, color="red", width=2, arrow=True)
    add_text(slide, "从 96 个候选点中\n选择 candidate + role", 285, 120, 250, 82,
             size=17, bold=True, align=2, valign=3, fill="soft_red", line="red", radius=True)
    add_line(slide, 540, 161, 610, 161, color="teal", width=2, arrow=True)
    add_text(slide, "更新已选集合\n计算增量 reward", 615, 120, 220, 82,
             size=17, bold=True, align=2, valign=3, fill="soft_teal", line="teal", radius=True)
    add_line(slide, 725, 207, 725, 264, color="teal", width=2, arrow=True)
    add_text(slide, "Stop / 达上限 / 无合法动作 / 30 步", 550, 270, 350, 52,
             size=16, bold=True, align=2, valign=3, fill="soft_yellow", line="yellow", radius=True)
    add_line(slide, 545, 296, 470, 296, color="teal", width=2, arrow=True)
    add_text(slide, "完整评价 + 修复\n终止奖励", 275, 258, 190, 76,
             size=17, bold=True, align=2, valign=3, fill="soft_blue", line="blue", radius=True)
    add_bullets(slide, [
        "最大选择：10 个传感器、2 个 AP；最少要求：3 个传感器、1 个 AP。",
        "PPO 不需要一次性输出巨大的组合向量，而是通过序列逐步构造部署。",
    ], 100, 386, 760, 84, size=15)

    # 27 candidate features
    slide = new_slide(pres, "PPO 候选点特征：每个候选 23 维", section="DRL-Init",
                      note="特征同时表达物理潜力、通信潜力、角色合法性和当前选择状态。")
    groups = [
        ("空间与表面 9", "xyz 归一化\n6 个 face one-hot"),
        ("热与能量 4", "Twall, Pgrid\nenergy margin\n距热源"),
        ("覆盖与通信 4", "coverage score\nlink quality / SNR\n距中心"),
        ("角色合法性 2", "is sensor candidate\nis AP candidate"),
        ("动态状态 4", "已选 sensor/AP\nskipped\nsink conflict risk"),
    ]
    for i, (title, body) in enumerate(groups):
        x = 45 + i * 181
        add_text(slide, title, x, 125, 165, 38, size=14, color="white", bold=True,
                 align=2, valign=3, fill=["navy", "red", "teal", "blue", "yellow"][i], radius=True)
        add_text(slide, body, x, 172, 165, 188, size=15, bold=True, align=2, valign=3,
                 fill="white", line="line", radius=True)
    add_text(slide, "矩阵形状：96 × 23；候选数量变化时，注意力编码器仍可处理变长集合。",
             155, 410, 650, 44, size=17, color="navy", bold=True,
             align=2, valign=3, fill="soft_blue", line="blue", radius=True)

    # 28 global features
    slide = new_slide(pres, "PPO 全局特征：15 维描述当前部分部署", section="DRL-Init")
    rows = [
        ["规模/预算 4", "已选 sensor/AP 比例；剩余 sensor/AP 槽位"],
        ["快速估计 5", "估计覆盖、平均 Pgrid、链路质量、能量压力、sink 压力"],
        ["进度 1", "当前步数 / 最大 30 步"],
        ["上次完整评价 5", "CV、feasible、Coverage、Rsum_norm、repair_iter_norm"],
    ]
    add_table(slide, ["分组", "特征"], rows, 95, 125, 770, 255,
              widths=[180, 590], font_size=14)
    add_text(slide, "候选特征回答“哪个点值得选”；全局特征回答“当前部署还缺什么”。",
             160, 415, 640, 50, size=18, color="red", bold=True,
             align=2, valign=3, fill="soft_red", line="red", radius=True)

    # 29 action
    slide = new_slide(pres, "PPO 动作空间：candidate × role 的联合离散动作", section="DRL-Init")
    add_text(slide, "动作 ID = candidate_id × 4 + role_id", 145, 112, 670, 56,
             size=24, color="navy", bold=True, align=2, valign=3,
             fill="white", line="blue", radius=True)
    roles = [
        ("sensor", "把候选点加入传感器集合", "teal"),
        ("AP", "把候选点加入 AP 集合", "blue"),
        ("skip", "明确跳过该候选点", "yellow"),
        ("stop", "结束当前部署 episode", "red"),
    ]
    for i, (role, desc, color) in enumerate(roles):
        x = 55 + i * 225
        add_text(slide, role, x, 220, 190, 48, size=20, color="white", bold=True,
                 align=2, valign=3, fill=color, radius=True)
        add_text(slide, desc, x, 278, 190, 100, size=15, bold=True,
                 align=2, valign=3, fill="white", line="line", radius=True)
    add_text(slide, "小场景原始联合动作数：96 × 4 = 384；action mask 会动态删除非法组合。",
             145, 420, 670, 42, size=17, color="navy", bold=True,
             align=2, valign=3, fill="soft_yellow", line="yellow", radius=True)

    # 30 mask
    slide = new_slide(pres, "PPO 联合 Action Mask：保证动作语义合法", section="DRL-Init")
    rows = [
        ["sensor", "候选必须具备 sensor 资格；未选、未 skip；sensor 数未达上限"],
        ["AP", "候选必须具备 AP 资格；未选、未 skip；AP 数未达上限"],
        ["skip", "候选尚未被处理，可标记跳过"],
        ["stop", "满足最少 3 sensor + 1 AP，或配置允许提前停止"],
    ]
    add_table(slide, ["角色", "合法条件"], rows, 75, 125, 810, 250,
              widths=[140, 670], font_size=13)
    add_bullets(slide, [
        "Mask 在网络 logits 上把非法动作置为 −1e9，Categorical 分布不会采到它们。",
        "仍保留 invalid/duplicate penalty 作为环境鲁棒性保护和日志证据。",
        "当没有任何合法动作时自动终止，避免死循环。",
    ], 105, 405, 750, 85, size=14, bullet_color="red")

    # 31 step reward
    slide = new_slide(pres, "PPO 步进奖励：鼓励每一次选择都有局部意义", section="DRL-Init")
    formula = "r_step = 0.10·ΔCoverage_est + 0.05·ΔEnergy_est + 0.05·ΔLink_est\n         − 0.05·ΔSinkRisk − 0.20·Invalid − 0.10·Duplicate"
    add_text(slide, formula, 80, 120, 800, 92, size=20, color="navy", bold=True,
             align=2, valign=3, fill="white", line="blue", radius=True)
    add_bullets(slide, [
        "覆盖增量：优先选择能覆盖更多未覆盖目标的传感器候选。",
        "能量增量：偏好表面温差和 P_grid 更有利的位置。",
        "链路增量：鼓励选择与潜在 AP 通信条件更好的节点。",
        "sink 风险：避免把过多节点集中到同一散热邻域。",
        "局部 reward 解决终止奖励过于稀疏的问题，但最终仍由完整评价纠偏。",
    ], 95, 255, 770, 205, size=15, bullet_color="red")

    # 32 terminal reward
    slide = new_slide(pres, "PPO 终止奖励：把部分部署映射回正式优化目标", section="DRL-Init")
    formula = "r_terminal = 0.25·Coverage + 0.25·Rsum_norm + 0.30·Feasible\n             − 0.35·CV_norm − 0.10·RepairCost + 0.08·Diversity\n             − 0.15·TooFewNodes"
    add_text(slide, formula, 75, 110, 810, 108, size=20, color="navy", bold=True,
             align=2, valign=3, fill="white", line="red", radius=True)
    add_bullets(slide, [
        "Rsum 统一使用 rsum_capacity，并按 [0, 2×10⁸] 归一化到 [0,1]。",
        "Feasible bonus 与 CV penalty 共同推动策略构造可修复、可行的部署。",
        "Repair cost 约束策略不要依赖大量后处理才能成立。",
        "Diversity bonus 鼓励生成不同部署，为 Gen0 提供多样的搜索起点。",
        "Too-few penalty 防止 PPO 通过过早 stop 获得表面上较低的约束风险。",
    ], 95, 255, 770, 205, size=15, bullet_color="red")

    # 33 network
    slide = new_slide(pres, "PPO Actor-Critic：候选注意力 + 角色策略头", section="DRL-Init")
    add_flow(slide, ["96×23\n候选特征", "Candidate MLP\n128", "Multi-head\nAttention", "融合全局 15 维", "Actor 4 roles\n+ Value head"],
             35, 120, 890, box_h=78, colors=["soft_blue", "soft_red", "soft_red", "soft_teal", "soft_yellow"])
    add_bullets(slide, [
        "候选 MLP 提取每个点的局部表示；全局状态作为 query 注意全部候选。",
        "Actor 为每个候选输出 4 个角色 logits，最终展平为联合动作分布。",
        "Value head 对池化后的整体状态估计 V(s)，用于 GAE 和 PPO 更新。",
        "hidden_dim=128，dropout=0.1；设备 auto，CUDA 可用时使用 GPU。",
    ], 80, 260, 800, 180, size=16, bullet_color="red")

    # 34 PPO objective
    slide = new_slide(pres, "PPO 更新：Clipped Objective + GAE", section="DRL-Init")
    add_text(slide, "L_policy = −E[min(rₜ(θ)Âₜ, clip(rₜ(θ),1−ε,1+ε)Âₜ)]",
             90, 115, 780, 58, size=20, color="navy", bold=True,
             align=2, valign=3, fill="white", line="blue", radius=True)
    add_text(slide, "L_total = L_policy + 0.5·L_value − 0.01·Entropy",
             150, 188, 660, 50, size=20, color="red", bold=True,
             align=2, valign=3, fill="soft_red", line="red", radius=True)
    rows = [
        ["γ", "0.98", "长期部署质量"], ["GAE λ", "0.95", "偏差-方差折中"],
        ["clip ε", "0.20", "限制单次策略更新幅度"], ["rollout", "512 steps", "跨 episode 累积轨迹"],
        ["epochs / minibatch", "8 / 128", "重复利用同一批 rollout"], ["grad clip", "0.5", "稳定训练"],
    ]
    add_table(slide, ["参数", "值", "含义"], rows, 145, 270, 670, 205,
              widths=[160, 150, 360], font_size=11)

    # 35 bootstrap
    slide = new_slide(pres, "Rollout 边界与 bootstrap：训练正确性的关键细节", section="DRL-Init")
    add_flow(slide, ["收集到 512 steps", "若 episode 未终止\n计算 V(s_last)", "GAE/Return", "PPO update", "清空 buffer"],
             45, 125, 870, box_h=76, colors=["soft_red", "soft_yellow", "soft_blue", "soft_red", "soft_teal"])
    add_bullets(slide, [
        "真正终止时 bootstrap=0；仅因 rollout 截断时必须使用 critic 的 V(s_last)。",
        "否则会把“截断”错误当成“环境终止”，系统性低估长轨迹价值。",
        "当前实现记录真实 PPO update_count、首次更新 episode、policy/value loss、entropy 和 approx KL。",
        "价值 bootstrap 使用 no_grad，避免额外建立计算图。",
    ], 80, 270, 800, 175, size=16, bullet_color="red")

    # 36 population mix
    slide = new_slide(pres, "混合 Gen0：为什么不是 100% PPO", section="DRL-Init")
    add_picture(slide, source_mix, 80, 125, 360, 280)
    add_bullets(slide, [
        "18 个 PPO：体现学习到的部署先验，是主要研究变量。",
        "6 个启发式：提供稳定的物理/覆盖/链路高分候选。",
        "6 个随机：保留未被策略和规则覆盖的探索方向。",
        "混合设计降低单一策略偏差，同时便于比较三种来源的可行率与目标质量。",
    ], 490, 135, 390, 225, size=16, bullet_color="red")
    add_text(slide, "正式有效性要求：实际接受的 DRL 个体比例不能低于 50%，否则本次不能宣称为有效 DRL 初始化。",
             110, 420, 740, 52, size=16, color="red", bold=True,
             align=2, valign=3, fill="soft_red", line="red", radius=True)

    # 37 retry diversity
    slide = new_slide(pres, "候选重试与多样性过滤", section="DRL-Init")
    add_flow(slide, ["PPO 采样部署", "完整评价+修复", "检查部署 Hamming", "检查优先级/目标距离", "接受或重试"],
             40, 115, 880, box_h=72, colors=["soft_red", "soft_teal", "soft_yellow", "soft_yellow", "soft_blue"])
    rows = [
        ["强部署距离", "Hamming ≥ 0.15", "优先判断真正部署是否不同"],
        ["弱部署距离", "Hamming ≥ 0.05", "允许优先级差异补充"],
        ["优先级 L2", "≥ 0.05", "避免连续编码近重复"],
        ["目标空间距离", "≥ 0.03", "仅作补充，不替代部署差异"],
        ["重试预算", "30 / candidate", "不足时记录原因并 fallback"],
    ]
    add_table(slide, ["规则", "阈值", "为什么"], rows, 80, 230, 800, 220,
              widths=[190, 180, 430], font_size=11)

    # 38 validity
    slide = new_slide(pres, "DRL_INIT_VALID：正式结果必须有证据链", section="DRL-Init")
    checks = [
        ("PPO 可用", "PyTorch 可用；策略成功加载或真实训练"),
        ("训练真实", "update_count>0；不是只创建网络或随机采样"),
        ("DRL 数量", "接受 DRL 个体数达到目标比例/最低阈值"),
        ("候选质量", "记录修复前后 FR/CV、Coverage、Rsum、来源差异"),
        ("checkpoint", "加载兼容性、新 checkpoint 保存状态分开记录"),
        ("失败策略", "invalid_run_policy=fail，不允许静默退化为随机/启发式"),
    ]
    for i, (title, desc) in enumerate(checks):
        x = 55 + (i % 2) * 445
        y = 115 + (i // 2) * 118
        add_text(slide, "✓", x, y, 46, 46, size=22, color="white", bold=True,
                 align=2, valign=3, fill="green", radius=True)
        add_text(slide, title, x + 58, y, 150, 24, size=16, bold=True, margin=0)
        add_text(slide, desc, x + 58, y + 29, 325, 55, size=13, color="muted", margin=0)

    # 39 cost accounting
    slide = new_slide(pres, "计算成本：在线公平与端到端成本要同时报告", section="DRL-Init")
    rows = [
        ["PPO terminal evaluations", "训练 episode 完整评价", "离线/前置"],
        ["Policy rollout terminal evals", "生成 DRL 候选时的终止评价", "离线/前置"],
        ["Candidate post evaluations", "候选转换后再评价", "离线/前置"],
        ["Heuristic/random evals", "混合种群其他来源评价", "离线/前置"],
        ["Gen0 re-evaluations", "正式初始种群统一评价", "在线"],
        ["CR-MODE trials", "60 代 trial 评价", "在线"],
    ]
    add_table(slide, ["计数项", "含义", "归属"], rows, 75, 120, 810, 285,
              widths=[260, 390, 160], font_size=11)
    add_text(slide, "公平主对比：在线 1830 次评价；完整成本报告：PPO + 候选生成 + 在线搜索。",
             140, 435, 680, 46, size=17, color="navy", bold=True,
             align=2, valign=3, fill="soft_yellow", line="yellow", radius=True)

    # 40 direct compare
    slide = new_slide(pres, "两个方案的本质差异", section="对比")
    rows = [
        ["DRL 介入位置", "每一代 Gen1–60", "仅 Gen0 之前"],
        ["学习算法", "Double-DQN", "PPO Actor-Critic"],
        ["状态粒度", "19 维种群统计", "96×23 候选 + 15 全局"],
        ["动作", "10 个搜索/修复动作包", "candidate × {sensor,AP,skip,stop}"],
        ["优化对象", "搜索策略选择", "初始部署构造"],
        ["正式评估网络", "冻结推理，每代调用", "生成 Gen0 后不再调用"],
        ["主要风险", "策略塌缩、reward 后期弱", "初始化过拟合、离线成本高"],
        ["解释重点", "自适应算子是否有效", "高质量 Gen0 是否加速/改善搜索"],
    ]
    add_table(slide, ["维度", "DQN-CR-MODE", "DRL-Init-CR-MODE"], rows,
              55, 112, 850, 355, widths=[180, 335, 335], font_size=11)

    # 41 fairness
    slide = new_slide(pres, "公平对比设计：哪些必须相同，哪些允许不同", section="对比")
    add_text(slide, "必须相同", 55, 112, 390, 42, size=20, color="white", bold=True,
             align=2, valign=3, fill="green", radius=True)
    add_bullets(slide, [
        "场景、温度场、TEG、信道、约束与目标定义",
        "NP=30、Tmax=60、每代 30 个 trial",
        "评价器、修复器、约束支配、Pareto Archive",
        "DQN 与 CR-MODE 使用同 seed 的原始 Gen0 快照",
    ], 72, 180, 350, 210, size=15, bullet_color="green")
    add_text(slide, "研究变量允许不同", 515, 112, 390, 42, size=20, color="white", bold=True,
             align=2, valign=3, fill="red", radius=True)
    add_bullets(slide, [
        "DQN：每代动作包由冻结 Q 网络选择",
        "DRL-Init：Gen0 由 PPO/heuristic/random 混合生成",
        "DRL-Init 的 PPO 前置训练成本单独报告",
        "不能要求 DRL-Init 与基线共用 Gen0，否则研究变量被消除",
    ], 532, 180, 350, 210, size=15, bullet_color="red")
    add_text(slide, "公平不等于所有东西都一样；公平是除研究变量外，其余边界一致且成本透明。",
             125, 430, 710, 48, size=17, color="navy", bold=True,
             align=2, valign=3, fill="soft_blue", line="blue", radius=True)

    # 42 parameters
    slide = new_slide(pres, "正式小场景实验参数", section="实验")
    rows = [
        ["空间", "3m × 3m × 3m；中心热源 70°C；96 candidates；40 targets"],
        ["部署", "最多 10 sensors + 2 APs"],
        ["CR-MODE", "NP=30；Tmax=60；Archive≤100；在线评价=1830/seed"],
        ["DQN", "2×128 MLP；120 episodes；γ=0.95；replay=20000；ε:1→0.05"],
        ["PPO", "1000 episodes；30 steps；rollout=512；8 epochs；γ=0.98；λ=0.95"],
        ["DRL Gen0", "60% PPO + 20% heuristic + 20% random"],
        ["正式 seed", "42、43、44"],
        ["目标", "maximize Coverage 与 Rsum_capacity；Rsum_ref_max=2×10⁸"],
    ]
    add_table(slide, ["项目", "设置"], rows, 70, 112, 820, 360,
              widths=[180, 640], font_size=12)

    # 43 results
    slide = new_slide(pres, "最新正式实验结果：三 seed 均值", section="实验",
                      note="DRL-Init 的端到端时间包含 PPO；在线预算仍与基线一致。")
    add_picture(slide, result_chart, 42, 108, 620, 300)
    rows = [
        ["CR-MODE", f"{summary['cr_mode']['hv']:.4f}", f"{summary['cr_mode']['coverage']:.4f}", f"{summary['cr_mode']['rsum']/1e6:.3f}M"],
        ["DQN", f"{summary['dqn_cr_mode']['hv']:.4f}", f"{summary['dqn_cr_mode']['coverage']:.4f}", f"{summary['dqn_cr_mode']['rsum']/1e6:.3f}M"],
        ["DRL-Init", f"{summary['drl_init_cr_mode']['hv']:.4f}", f"{summary['drl_init_cr_mode']['coverage']:.4f}", f"{summary['drl_init_cr_mode']['rsum']/1e6:.3f}M"],
    ]
    add_table(slide, ["算法", "HV", "Coverage", "Rsum"], rows, 680, 130, 235, 200,
              widths=[80, 48, 55, 52], font_size=9)
    add_text(slide, "DRL-Init 在当前 3 个 seed 上整体最好；DQN 与基线接近且稳定优势尚未建立。",
             120, 438, 720, 45, size=16, color="navy", bold=True,
             align=2, valign=3, fill="soft_yellow", line="yellow", radius=True)

    # 44 pareto
    slide = new_slide(pres, "Pareto 前沿与收敛证据", section="实验")
    pareto = RESULT_ROOT / "figures" / "pareto_compare_capacity.png"
    hv = RESULT_ROOT / "figures" / "hv_compare_all.png"
    add_picture(slide, pareto, 48, 112, 420, 330)
    add_picture(slide, hv, 492, 112, 420, 330)
    add_text(slide, "左：Coverage–Rsum_capacity 非支配前沿；右：HV 随代数的收敛表现。",
             160, 455, 640, 30, size=14, color="muted", bold=True, align=2, margin=0)

    # 45 interpretation
    slide = new_slide(pres, "如何解释实验，而不是只说“谁更好”", section="实验")
    cards = [
        ("CR-MODE", "提供稳定基线，证明公共修复和 Pareto 搜索本身可用。", "navy"),
        ("DQN", "当前 checkpoint 有明显 energy_first 偏置；平均 HV 未超过基线。", "teal"),
        ("DRL-Init", "Gen0 可行率更高，最终 HV/Rsum 更强；但前置训练成本约 5 分钟/seed。", "red"),
        ("研究结论", "目前最强证据支持“学习初始化有效”；“学习在线动作控制有效”尚需改进后复验。", "yellow"),
    ]
    for i, (title, body, color) in enumerate(cards):
        x = 55 + (i % 2) * 445
        y = 120 + (i // 2) * 165
        add_text(slide, title, x, y, 390, 40, size=18, color="white", bold=True,
                 align=2, valign=3, fill=color, radius=True)
        add_text(slide, body, x, y + 50, 390, 92, size=14, bold=True,
                 align=2, valign=3, fill="white", line="line", radius=True)

    # 46 limitations
    slide = new_slide(pres, "当前方案的主要局限与答辩口径", section="改进")
    rows = [
        ["DQN reward 后期弱", "已实测确认", "不能把动作偏置直接解释为学到最优策略"],
        ["DQN 训练动作日志被覆盖", "工程证据不足", "需要分离 train/validation 日志"],
        ["DQN 验证 seed 仅 3 个", "存在模型选择过拟合", "扩展验证集并保留独立测试集"],
        ["DRL-Init 是 per-seed PPO", "不是 unseen-seed 泛化", "只能称为场景内学习初始化"],
        ["温度场是 synthetic", "不是 CFD/ANSYS", "用于算法验证，不宣称高保真传热预测"],
        ["正式 seed 仅 3 个", "统计把握有限", "后续至少 10 个配对 seed + 显著性检验"],
    ]
    add_table(slide, ["问题", "当前判断", "汇报/改进方式"], rows, 50, 112, 860, 340,
              widths=[220, 210, 430], font_size=11)

    # 47 roadmap
    slide = new_slide(pres, "建议的下一轮改进路线", section="改进",
                      note="先补证据，再改 reward，最后重训，避免边改边猜。")
    roadmap = [
        ("1", "证据链", "train/validation 日志隔离；每动作 Q、reward、次数；参数哈希"),
        ("2", "Reward", "配置化权重；平衡前期可行性与后期 HV/目标增量；稳定 scaling"),
        ("3", "Checkpoint", "≥10 验证 seed；记录 best episode；多指标与基线增益诊断"),
        ("4", "重训与消融", "DQN vs fixed action vs random legal；同 Gen0、同预算、配对 seed"),
        ("5", "统计报告", "均值±标准差、置信区间、配对 Wilcoxon/Bootstrap、效应量"),
    ]
    for i, (num, title, desc) in enumerate(roadmap):
        y = 105 + i * 76
        add_text(slide, num, 55, y, 48, 48, size=20, color="white", bold=True,
                 align=2, valign=3, fill="red" if i < 2 else "teal", radius=True)
        add_text(slide, title, 125, y, 150, 24, size=17, bold=True, margin=0)
        add_text(slide, desc, 285, y, 610, 48, size=14, color="muted", valign=2, margin=0)
        add_line(slide, 125, y + 58, 895, y + 58)

    # 48 likely questions
    slide = new_slide(pres, "导师可能追问的问题与简要回答", section="答辩")
    qa = [
        ["为什么 DQN 不直接输出部署？", "组合空间太大；让 DQN 控制成熟优化器，保留约束处理和 Pareto 搜索能力。"],
        ["为什么 PPO 只做 Gen0？", "隔离研究变量，保证 Gen1–60 与基线共享搜索内核，便于归因。"],
        ["为什么不是 100% PPO？", "混合种群降低策略偏差并保留探索；同时可做来源质量分析。"],
        ["DQN 选 energy_first 是否说明它最好？", "不能。当前消融中 random/fixed 策略可超过 DQN，需要更多 seed 和重训。"],
        ["DRL-Init 算法是否公平？", "在线评价预算公平；PPO 前置成本单独报告，不隐去端到端成本。"],
        ["能否声称泛化？", "当前 per-seed PPO 不能；需要 shared-pretrain + unseen-seed 冻结测试。"],
    ]
    add_table(slide, ["问题", "回答要点"], qa, 45, 105, 870, 370,
              widths=[300, 570], font_size=11)

    # 49 conclusion
    slide = new_slide(pres, "一页总结：两个方案分别解决什么", section="总结",
                      note="最后回到研究问题，不停留在网络结构。")
    add_text(slide, "DQN-CR-MODE", 60, 112, 390, 48, size=23, color="white", bold=True,
             align=2, valign=3, fill="teal", radius=True)
    add_text(slide, "学习“这一代怎么搜”\n\n19 维种群状态 → 10 个动作包\n每代控制 F / CR / 变异 / 修复 / 功率\n\n当前：流程有效，但策略优势尚未建立",
             60, 175, 390, 245, size=18, bold=True, align=2, valign=3,
             fill="white", line="line", radius=True)
    add_text(slide, "DRL-Init-CR-MODE", 510, 112, 390, 48, size=23, color="white", bold=True,
             align=2, valign=3, fill="red", radius=True)
    add_text(slide, "学习“从哪里开始搜”\n\n96×23 候选特征 + 15 全局特征\nPPO 顺序构造部署 → 混合 Gen0\n\n当前：初始化质量与最终结果证据更强",
             510, 175, 390, 245, size=18, bold=True, align=2, valign=3,
             fill="white", line="line", radius=True)
    add_text(slide, "共同底座：同一物理场、同一目标约束、同一修复器、同一 CR-MODE 与 Pareto 选择。",
             115, 446, 730, 40, size=16, color="navy", bold=True,
             align=2, valign=3, fill="soft_yellow", line="yellow", radius=True)

    # 50 appendix symbols
    slide = new_slide(pres, "附录：关键符号与汇报时的准确口径", section="附录")
    rows = [
        ["F", "差分变异缩放因子", "控制搜索步长"], ["CR", "交叉概率", "控制父代/变异向量混合程度"],
        ["FR", "Feasible Ratio", "种群中可行解比例"], ["CV", "Constraint Violation", "所有约束违反程度的统一度量"],
        ["HV", "Hypervolume", "Pareto 前沿质量与覆盖范围"], ["Rsum_capacity", "有效链路 Shannon 容量之和", "唯一吞吐量目标口径"],
        ["GAE", "Generalized Advantage Estimation", "PPO 优势估计"], ["Gen0", "初始种群", "DRL-Init 的唯一搜索差异来源"],
    ]
    add_table(slide, ["符号", "定义", "在方案中的作用"], rows, 65, 110, 830, 365,
              widths=[150, 300, 380], font_size=11)

    # Final formatting sanity pass.
    for slide in pres.Slides:
        for shape in slide.Shapes:
            try:
                if shape.HasTextFrame and shape.TextFrame.HasText:
                    shape.TextFrame.TextRange.Font.Name = "Microsoft YaHei"
                    shape.TextFrame.TextRange.Font.NameFarEast = "Microsoft YaHei"
            except Exception:
                pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pres.SaveAs(str(output_path), 24)
    slide_count = pres.Slides.Count
    shape_count = sum(slide.Shapes.Count for slide in pres.Slides)
    pres.Close()
    app.Quit()
    return slide_count, shape_count


def main():
    output = Path(os.environ.get(
        "ALGORITHM_PPT_OUTPUT",
        str(ROOT / "DQN-CR-MODE_and_DRL-Init-CR-MODE_detailed.pptx"),
    ))
    slides, shapes = build_presentation(output)
    print(json.dumps({
        "output": str(output.resolve()),
        "slides": slides,
        "shapes": shapes,
        "size_bytes": output.stat().st_size,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
