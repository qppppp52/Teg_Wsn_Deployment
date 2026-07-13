from __future__ import annotations

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon


ROOT = Path(__file__).resolve().parents[1]
DESKTOP = Path(os.environ.get("FLOWCHART_OUTPUT_DIR", Path.home() / "Desktop"))
SOURCE_PPT_PATH = DESKTOP / "DQN-CR-MODE与DRL-Init-CR-MODE算法方案详解.pptx"
PPT_PATH = DESKTOP / "DQN-CR-MODE与DRL-Init-CR-MODE算法方案详解_含流程图.pptx"
FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT = font_manager.FontProperties(fname=str(FONT_PATH)) if FONT_PATH.exists() else None

NAVY = "#17324D"
BLUE = "#276FBF"
CYAN = "#2A9D8F"
GREEN = "#5B8E3E"
AMBER = "#E9A23B"
RED = "#C94C4C"
PURPLE = "#7656A3"
INK = "#263442"
PALE = "#F4F7FA"
WHITE = "#FFFFFF"


def _text(ax, x, y, value, size=12, color=INK, weight="normal", ha="center", va="center"):
    ax.text(
        x,
        y,
        value,
        fontproperties=FONT,
        fontsize=size,
        color=color,
        fontweight=weight,
        ha=ha,
        va=va,
        linespacing=1.25,
        zorder=5,
    )


def _box(ax, x, y, w, h, value, fill=WHITE, edge=BLUE, size=11, radius=0.12, lw=1.6):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.035,rounding_size={radius}",
        facecolor=fill,
        edgecolor=edge,
        linewidth=lw,
        zorder=2,
    )
    ax.add_patch(patch)
    _text(ax, x + w / 2, y + h / 2, value, size=size)
    return patch


def _decision(ax, cx, cy, w, h, value, fill="#FFF7E8", edge=AMBER, size=10.5):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=fill, edgecolor=edge, linewidth=1.7, zorder=2))
    _text(ax, cx, cy, value, size=size)


def _arrow(ax, start, end, color="#607487", label=None, curve=0.0, lw=1.7):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={curve}",
        shrinkA=2,
        shrinkB=2,
        zorder=3,
    )
    ax.add_patch(arrow)
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        _text(ax, mx, my + 0.12, label, size=9, color=color)


def _setup(title, subtitle):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=160)
    fig.patch.set_facecolor(PALE)
    ax.set_facecolor(PALE)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    _text(ax, 0.55, 8.48, title, size=25, color=NAVY, weight="bold", ha="left")
    _text(ax, 0.57, 8.03, subtitle, size=11.5, color="#5C6E7E", ha="left")
    ax.plot([0.55, 15.45], [7.76, 7.76], color="#CCD6DF", linewidth=1.2)
    return fig, ax


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches=None)
    plt.close(fig)


def make_dqn_flow(path):
    fig, ax = _setup(
        "DQN-CR-MODE 算法流程图",
        "DQN 在 CR-MODE 的每一代决策修复/进化策略；正式评估时策略冻结，不进行回放写入或梯度更新。",
    )

    _text(ax, 0.72, 7.43, "A. 离线训练与 checkpoint 选择", size=13, color=PURPLE, weight="bold", ha="left")
    train_y = 6.5
    boxes = [
        (0.7, 1.9, "训练配置\n120 episodes，seeds 1000–1119"),
        (3.05, 2.15, "环境交互\n状态 + action mask"),
        (5.65, 2.2, "ε-greedy 选动作\n执行 1 代 CR-MODE"),
        (8.3, 2.15, "经验回放 + DDQN\n更新 online/target Q 网络"),
        (10.9, 2.05, "定期验证\nseeds 142–144"),
        (13.4, 1.85, "保存最佳 checkpoint\n及训练证据"),
    ]
    for i, (x, w, t) in enumerate(boxes):
        _box(ax, x, train_y, w, 0.72, t, fill="#F4EFFA", edge=PURPLE, size=9.5)
        if i:
            px, pw, _ = boxes[i - 1]
            _arrow(ax, (px + pw, train_y + 0.36), (x, train_y + 0.36), color=PURPLE)

    _text(ax, 0.72, 6.14, "B. 冻结正式评估（与 CR-MODE / DRL-Init-CR-MODE 按相同 NP、代数和评估预算比较）", size=13, color=BLUE, weight="bold", ha="left")

    _box(ax, 0.7, 5.05, 2.05, 0.72, "读取小场景配置\n构建温度/信道/约束", fill="#EAF3FB", edge=BLUE, size=9.7)
    _box(ax, 3.15, 5.05, 2.15, 0.72, "加载最佳 DQN\n校验状态/动作维度", fill="#EAF3FB", edge=BLUE, size=9.7)
    _box(ax, 5.7, 5.05, 2.05, 0.72, "设为 eval 模式\nε=0，禁止训练", fill="#E9F6F3", edge=CYAN, size=9.7)
    _box(ax, 8.15, 5.05, 2.25, 0.72, "加载共享 Gen0\nNP=30，建立 Pareto 档案", fill="#EDF6E9", edge=GREEN, size=9.7)
    for x1, x2 in [(2.75, 3.15), (5.3, 5.7), (7.75, 8.15)]:
        _arrow(ax, (x1, 5.41), (x2, 5.41), color=BLUE)
    _arrow(ax, (14.32, 6.5), (4.25, 5.78), color=PURPLE, label="仅加载", curve=0.08)

    # Generational loop.
    _box(ax, 0.7, 3.56, 2.25, 0.9, "19 维状态 s_t\n可行性/目标/压力/多样性\n修复统计/进度", fill=WHITE, edge=BLUE, size=9.2)
    _box(ax, 3.35, 3.56, 2.05, 0.9, "构造 action mask\n屏蔽当前不合理策略", fill="#FFF7E8", edge=AMBER, size=9.4)
    _box(ax, 5.8, 3.56, 2.05, 0.9, "argmax Q(s_t,a)\n从 10 个策略动作中选择", fill="#F4EFFA", edge=PURPLE, size=9.4)
    _box(ax, 8.25, 3.56, 2.35, 0.9, "动作映射\nF / CR / 变异方式 / 修复顺序\n功率与多样性策略", fill="#F4EFFA", edge=PURPLE, size=9.1)
    _box(ax, 11.0, 3.56, 2.25, 0.9, "GenerationExecutor\n变异→交叉→修复→评估\n30 个 trial", fill="#EDF6E9", edge=GREEN, size=9.1)
    _box(ax, 13.65, 3.56, 1.65, 0.9, "约束 Pareto\n环境选择\n+档案更新", fill="#EDF6E9", edge=GREEN, size=9.2)
    row = [(0.7, 2.95), (3.35, 5.4), (5.8, 7.85), (8.25, 10.6), (11.0, 13.25), (13.65, 15.3)]
    for (_, end), (start, _) in zip(row, row[1:]):
        _arrow(ax, (end, 4.01), (start, 4.01))
    _arrow(ax, (9.28, 5.05), (9.28, 4.48), color=BLUE)

    _box(ax, 11.0, 2.16, 2.25, 0.78, "计算 reward r_t\n得到 s_(t+1) 与新 mask", fill="#FFF7E8", edge=AMBER, size=9.5)
    _box(ax, 8.25, 2.16, 2.35, 0.78, "写入动作/Q值/reward\nFR、CV、Coverage、Rsum", fill=WHITE, edge=BLUE, size=9.3)
    _decision(ax, 6.72, 2.55, 1.8, 1.0, "g < 60？", size=10)
    _arrow(ax, (14.48, 3.55), (12.12, 2.95), curve=-0.15)
    _arrow(ax, (11.0, 2.55), (10.6, 2.55))
    _arrow(ax, (8.25, 2.55), (7.62, 2.55))
    _arrow(ax, (6.72, 3.05), (1.82, 3.54), label="是：g=g+1", curve=-0.2, color=AMBER)

    _box(ax, 5.45, 0.78, 2.55, 0.82, "输出 Pareto 解集\n推荐解 + 收敛曲线", fill="#EAF3FB", edge=BLUE, size=10)
    _box(ax, 8.45, 0.78, 3.15, 0.82, "DQN 有效性证据\ncheckpoint_loaded / ε=0 / 参数哈希不变", fill="#FCEFEF", edge=RED, size=9.6)
    _arrow(ax, (6.72, 2.05), (6.72, 1.62), label="否：结束", color=AMBER)
    _arrow(ax, (8.0, 1.19), (8.45, 1.19), color=BLUE)
    _text(ax, 0.72, 0.28, "核心归因：性能差异来自“每代的自适应进化决策”，不是额外评估预算。", size=10.5, color=NAVY, weight="bold", ha="left")
    _save(fig, path)


def make_drl_init_flow(path):
    fig, ax = _setup(
        "DRL-Init-CR-MODE 算法流程图",
        "PPO 仅负责生成高质量初始种群；Gen1–Gen60 与基线 CR-MODE 共用同一进化内核。",
    )

    _box(ax, 0.65, 6.6, 2.15, 0.82, "读取场景配置\n温度预处理 + 候选点特征", fill="#EAF3FB", edge=BLUE, size=9.8)
    _decision(ax, 4.2, 7.0, 2.05, 1.05, "PPO checkpoint\n兼容且允许复用？", size=9.7)
    _box(ax, 5.9, 6.6, 2.05, 0.82, "加载 PPO 策略\n校验 config hash", fill="#E9F6F3", edge=CYAN, size=9.7)
    _arrow(ax, (2.8, 7.01), (3.16, 7.01), color=BLUE)
    _arrow(ax, (5.22, 7.01), (5.9, 7.01), label="是", color=CYAN)

    # PPO training lane.
    _box(ax, 3.2, 5.05, 2.0, 0.86, "PPO 训练\n1000 episodes", fill="#F4EFFA", edge=PURPLE, size=10)
    _box(ax, 5.65, 5.05, 2.25, 0.86, "候选点序列决策\n96×23 特征 + 15 维全局状态\n联合 action mask", fill="#F4EFFA", edge=PURPLE, size=8.8)
    _box(ax, 8.35, 5.05, 2.15, 0.86, "Attention Actor-Critic\n候选点 × 部署角色", fill="#F4EFFA", edge=PURPLE, size=9.2)
    _box(ax, 10.95, 5.05, 2.0, 0.86, "步进奖励 + 终局奖励\n终局解码/修复/评估", fill="#FFF7E8", edge=AMBER, size=9.1)
    _box(ax, 13.4, 5.05, 1.95, 0.86, "Rollout=512\nGAE + clipped PPO\n参数更新", fill="#F4EFFA", edge=PURPLE, size=9.1)
    _arrow(ax, (4.2, 6.48), (4.2, 5.93), label="否 / 强制重训", color=PURPLE)
    for a, b in [(5.2, 5.65), (7.9, 8.35), (10.5, 10.95), (12.95, 13.4)]:
        _arrow(ax, (a, 5.48), (b, 5.48), color=PURPLE)
    _arrow(ax, (14.37, 5.94), (3.55, 5.94), color=PURPLE, label="继续下一 rollout", curve=0.12)

    _text(ax, 0.72, 4.62, "A. 用 PPO 生成初始种群 Gen0", size=13, color=CYAN, weight="bold", ha="left")
    _box(ax, 0.7, 3.62, 2.3, 0.8, "设定 NP=30 的混合配额\n18 DRL + 6 heuristic + 6 random", fill="#E9F6F3", edge=CYAN, size=9.5)
    _box(ax, 3.45, 3.62, 2.1, 0.8, "PPO 顺序生成部署\n候选点选择 + 角色分配", fill="#E9F6F3", edge=CYAN, size=9.2)
    _box(ax, 6.0, 3.62, 2.25, 0.8, "终局解码/修复/评估\n质量与多样性过滤", fill="#EDF6E9", edge=GREEN, size=9.2)
    _decision(ax, 9.72, 4.02, 1.9, 1.0, "接受数达标？", size=9.8)
    _box(ax, 11.15, 3.62, 1.85, 0.8, "未达标：重试\n最多 30 次", fill="#FFF7E8", edge=AMBER, size=9.2)
    _box(ax, 13.45, 3.62, 1.85, 0.8, "合并三类来源\n得到 Gen0", fill="#E9F6F3", edge=CYAN, size=9.5)
    _arrow(ax, (6.92, 6.6), (4.5, 4.44), color=CYAN, label="冻结策略", curve=0.08)
    for a, b in [(3.0, 3.45), (5.55, 6.0), (8.25, 8.77)]:
        _arrow(ax, (a, 4.02), (b, 4.02), color=CYAN)
    _arrow(ax, (10.67, 4.02), (11.15, 4.02), label="否", color=AMBER)
    _arrow(ax, (13.0, 4.02), (13.45, 4.02), label="是/回退补齐", color=CYAN)

    _text(ax, 0.72, 3.2, "B. 进入共享 CR-MODE 进化阶段", size=13, color=GREEN, weight="bold", ha="left")
    _box(ax, 0.7, 2.15, 2.35, 0.82, "Gen0 统一评估\n建立约束 Pareto 档案", fill="#EDF6E9", edge=GREEN, size=9.7)
    _box(ax, 3.5, 2.15, 2.35, 0.82, "固定/共享 CR-MODE 策略\nF、CR、变异和修复口径一致", fill="#EDF6E9", edge=GREEN, size=9.2)
    _box(ax, 6.3, 2.15, 2.25, 0.82, "每代 30 个 trial\n变异→交叉→修复→评估", fill="#EDF6E9", edge=GREEN, size=9.2)
    _box(ax, 9.0, 2.15, 2.15, 0.82, "约束 Pareto 选择\n更新种群与档案", fill="#EDF6E9", edge=GREEN, size=9.4)
    _decision(ax, 12.45, 2.56, 1.65, 0.95, "g < 60？", size=10)
    _box(ax, 13.7, 2.15, 1.62, 0.82, "输出 Pareto\n与推荐解", fill="#EAF3FB", edge=BLUE, size=9.7)
    _arrow(ax, (14.38, 3.6), (1.88, 2.98), color=CYAN, label="Gen0", curve=0.04)
    for a, b in [(3.05, 3.5), (5.85, 6.3), (8.55, 9.0), (11.15, 11.62)]:
        _arrow(ax, (a, 2.56), (b, 2.56), color=GREEN)
    _arrow(ax, (12.45, 3.04), (4.67, 3.04), color=GREEN, label="是：g=g+1", curve=0.07)
    _arrow(ax, (13.28, 2.56), (13.7, 2.56), label="否", color=BLUE)

    _box(ax, 0.7, 0.68, 4.0, 0.82, "有效性门禁\npolicy_loaded / PPO 确实更新 / DRL 接受比例达标", fill="#FCEFEF", edge=RED, size=9.4)
    _box(ax, 5.15, 0.68, 4.15, 0.82, "来源级证据\nDRL / heuristic / random 的生成、接受、回退统计", fill=WHITE, edge=BLUE, size=9.3)
    _box(ax, 9.75, 0.68, 5.55, 0.82, "公平性口径\n主进化阶段预算一致；PPO 训练/初始化成本单独报告", fill="#FFF7E8", edge=AMBER, size=9.6)
    _text(ax, 0.72, 0.22, "核心归因：性能差异来自“初始种群质量”；PPO 不参与 Gen1–Gen60 的逐代决策。", size=10.5, color=NAVY, weight="bold", ha="left")
    _save(fig, path)


def make_comparison_flow(path):
    fig, ax = _setup(
        "DQN-CR-MODE 与 DRL-Init-CR-MODE 流程对比图",
        "两者的 DRL 介入时机不同：前者学习“每代如何进化”，后者学习“Gen0 如何生成”。",
    )
    _box(ax, 5.65, 6.82, 4.7, 0.62, "共享输入：场景、温度、信道、约束、NP=30、G=60、目标口径", fill=WHITE, edge=NAVY, size=10)

    # Lane headers.
    ax.add_patch(FancyBboxPatch((0.55, 5.92), 7.25, 0.58, boxstyle="round,pad=0.02,rounding_size=0.1", facecolor="#EAF3FB", edgecolor=BLUE, linewidth=1.4))
    ax.add_patch(FancyBboxPatch((8.2, 5.92), 7.25, 0.58, boxstyle="round,pad=0.02,rounding_size=0.1", facecolor="#E9F6F3", edgecolor=CYAN, linewidth=1.4))
    _text(ax, 4.17, 6.21, "DQN-CR-MODE：逐代决策增强", size=13, color=BLUE, weight="bold")
    _text(ax, 11.82, 6.21, "DRL-Init-CR-MODE：初始种群增强", size=13, color=CYAN, weight="bold")
    _arrow(ax, (8.0, 6.82), (4.17, 6.52), color=NAVY, curve=0.08)
    _arrow(ax, (8.0, 6.82), (11.82, 6.52), color=NAVY, curve=-0.08)

    left = [
        (5.02, "离线训练 DDQN\n训练集 + 独立验证集"),
        (3.95, "加载共享 Gen0\n冻结 DQN，ε=0"),
        (2.88, "每代构造状态与 mask\nDQN 选 1/10 进化策略"),
        (1.81, "执行一代共享 GenerationExecutor\n评估、选择、档案更新"),
        (0.74, "60 代结束\nPareto + 动作/Q值/reward 证据"),
    ]
    right = [
        (5.02, "训练/加载 PPO\n候选点序列部署策略"),
        (3.95, "PPO 生成 DRL 候选解\n与 heuristic / random 混合"),
        (2.88, "质量和多样性过滤\n形成特有 Gen0"),
        (1.81, "进入共享 CR-MODE 内核\nPPO 不再参与逐代决策"),
        (0.74, "60 代结束\nPareto + 初始化来源/成本证据"),
    ]
    for lane, x, color, fill in [(left, 0.85, BLUE, "#F7FBFF"), (right, 8.5, CYAN, "#F5FBF9")]:
        for i, (y, txt) in enumerate(lane):
            _box(ax, x, y, 6.65, 0.73, txt, fill=fill, edge=color, size=9.7)
            if i:
                py, _ = lane[i - 1]
                _arrow(ax, (x + 3.325, py), (x + 3.325, y + 0.75), color=color)

    # Center divider and attribution labels.
    ax.plot([8.0, 8.0], [0.58, 5.7], color="#CBD5DE", linewidth=1.1, linestyle="--")
    _text(ax, 7.82, 4.32, "学习对象：进化决策", size=9.5, color=BLUE, ha="right")
    _text(ax, 8.18, 4.32, "学习对象：部署初始化", size=9.5, color=CYAN, ha="left")
    _text(ax, 7.82, 2.18, "DRL 调用：Gen1–Gen60", size=9.5, color=BLUE, ha="right")
    _text(ax, 8.18, 2.18, "DRL 调用：Gen0 之前/期间", size=9.5, color=CYAN, ha="left")
    _text(ax, 0.65, 0.2, "解读原则：两算法不是“同一 DRL 的两个名字”，而是对 CR-MODE 两个不同环节的学习增强。", size=10.3, color=NAVY, weight="bold", ha="left")
    _save(fig, path)


def append_to_ppt(images):
    if not SOURCE_PPT_PATH.exists():
        raise FileNotFoundError(f"PPT not found: {SOURCE_PPT_PATH}")
    shutil.copy2(SOURCE_PPT_PATH, PPT_PATH)
    import win32com.client  # type: ignore

    app = win32com.client.DispatchEx("PowerPoint.Application")
    app.Visible = True
    presentation = app.Presentations.Open(str(PPT_PATH), WithWindow=False)
    try:
        for index in range(presentation.Slides.Count, 0, -1):
            slide = presentation.Slides(index)
            try:
                if slide.Tags.Item("CODEX_FLOWCHART"):
                    slide.Delete()
            except Exception:
                pass
        width = presentation.PageSetup.SlideWidth
        height = presentation.PageSetup.SlideHeight
        for label, image_path in images:
            slide = presentation.Slides.Add(presentation.Slides.Count + 1, 12)
            slide.Tags.Add("CODEX_FLOWCHART", label)
            slide.Shapes.AddPicture(str(image_path), False, True, 0, 0, width, height)
        presentation.Save()
    finally:
        presentation.Close()
        app.Quit()


def main():
    dqn = DESKTOP / "DQN-CR-MODE算法流程图.png"
    drl = DESKTOP / "DRL-Init-CR-MODE算法流程图.png"
    compare = DESKTOP / "DQN-CR-MODE与DRL-Init-CR-MODE流程对比图.png"
    make_dqn_flow(dqn)
    make_drl_init_flow(drl)
    make_comparison_flow(compare)
    append_to_ppt([("DQN", dqn), ("DRL_INIT", drl), ("COMPARE", compare)])
    for item in (dqn, drl, compare, PPT_PATH):
        print(f"{item}\t{item.stat().st_size}")


if __name__ == "__main__":
    main()


