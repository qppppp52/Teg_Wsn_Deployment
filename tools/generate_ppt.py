#!/usr/bin/env python
"""
生成科研汇报 PPT：密闭空间中基于 TEG 热电采能的 WSN 传感器与 AP 协同部署优化算法研究
算法主线：DRL-CR-MODE（CR-MODE + DRL 策略调控）
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from pptx import Presentation
from pptx.util import Inches, Pt, Emu, Cm
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
FIG_DIR = os.path.join(ROOT, "results", "figures")
OUT_PATH = os.path.join(ROOT, "results", "DRL_CR_MODE_汇报.pptx")

# ── 颜色主题 ──
C_PRIMARY = RGBColor(0x0D, 0x47, 0xA1)       # 深蓝
C_SECONDARY = RGBColor(0x21, 0x96, 0xF3)     # 亮蓝
C_ACCENT = RGBColor(0xC6, 0x28, 0x28)         # 红
C_DARK = RGBColor(0x21, 0x21, 0x21)           # 深灰文字
C_LIGHT_BG = RGBColor(0xE3, 0xF2, 0xFD)      # 浅蓝背景
C_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
C_GRAY = RGBColor(0x75, 0x75, 0x75)
C_GREEN = RGBColor(0x2E, 0x7D, 0x32)
C_ORANGE = RGBColor(0xE6, 0x51, 0x00)

prs = Presentation()
prs.slide_width = Inches(13.333)   # 16:9 宽屏
prs.slide_height = Inches(7.5)

# ── 工具函数 ──
def add_blank_slide():
    layout = prs.slide_layouts[6]  # blank
    return prs.slides.add_slide(layout)

def add_textbox(slide, left, top, width, height, text="", font_size=18,
                bold=False, color=C_DARK, alignment=PP_ALIGN.LEFT,
                font_name="Microsoft YaHei", line_spacing=1.3):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = alignment
    p.space_after = Pt(line_spacing * font_size - font_size)
    return tf

def add_title_bar(slide, title_text, subtitle_text=""):
    """顶部标题栏"""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0),
        prs.slide_width, Inches(1.1)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = C_PRIMARY
    shape.line.fill.background()
    add_textbox(slide, 0.6, 0.15, 12, 0.6, title_text,
                font_size=30, bold=True, color=C_WHITE)
    if subtitle_text:
        add_textbox(slide, 0.6, 0.7, 12, 0.35, subtitle_text,
                    font_size=14, color=RGBColor(0xBB, 0xDE, 0xFB))
    # 底部细线
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(1.1),
        prs.slide_width, Inches(0.03)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = C_SECONDARY
    line.line.fill.background()

def add_body_text(slide, left, top, width, height, bullets, font_size=16,
                  color=C_DARK, line_spacing=1.5, bold_prefix=True):
    """添加要点列表"""
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, bullet in enumerate(bullets):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        # 支持 bold prefix: "**关键词**：说明"
        if bold_prefix and "**" in bullet:
            parts = bullet.split("**")
            for j, part in enumerate(parts):
                if j % 2 == 1:  # bold part
                    run = p.add_run()
                    run.text = part
                    run.font.bold = True
                    run.font.size = Pt(font_size)
                    run.font.color.rgb = color
                    run.font.name = "Microsoft YaHei"
                else:
                    run = p.add_run()
                    run.text = part
                    run.font.size = Pt(font_size)
                    run.font.color.rgb = color
                    run.font.name = "Microsoft YaHei"
        else:
            p.text = "• " + bullet
            p.font.size = Pt(font_size)
            p.font.color.rgb = color
            p.font.name = "Microsoft YaHei"
        p.space_after = Pt(4)
        p.alignment = PP_ALIGN.LEFT
    return tf

def add_image_safe(slide, img_rel_path, left, top, width, height=None):
    """安全插入图片"""
    img_path = os.path.join(ROOT, img_rel_path)
    if os.path.exists(img_path):
        if height:
            return slide.shapes.add_picture(img_path, Inches(left), Inches(top),
                                            Inches(width), Inches(height))
        else:
            return slide.shapes.add_picture(img_path, Inches(left), Inches(top),
                                            Inches(width))
    else:
        # 占位框
        shape = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(left), Inches(top),
            Inches(width), Inches(height if height else width * 0.6)
        )
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor(0xF5, 0xF5, 0xF5)
        shape.line.color.rgb = C_GRAY
        add_textbox(slide, left + 0.2, top + (height or width*0.6)/2 - 0.2,
                    width - 0.4, 0.4, f"[图片缺失: {img_rel_path}]",
                    font_size=12, color=C_GRAY, alignment=PP_ALIGN.CENTER)
        return shape

def add_flow_box(slide, left, top, width, height, text, fill_color=C_LIGHT_BG,
                 border_color=C_SECONDARY, font_size=12, bold=False, color=C_DARK):
    """流程图节点"""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top),
        Inches(width), Inches(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.color.rgb = border_color
    shape.line.width = Pt(1.5)
    tf = shape.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = "Microsoft YaHei"
    p.alignment = PP_ALIGN.CENTER
    return shape

def add_arrow(slide, left, top, width, height):
    """右箭头"""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RIGHT_ARROW, Inches(left), Inches(top),
        Inches(width), Inches(height)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = C_PRIMARY
    shape.line.fill.background()
    return shape

def add_page_number(slide, num, total=18):
    add_textbox(slide, 12.2, 7.1, 1.0, 0.3, f"{num}/{total}",
                font_size=10, color=C_GRAY, alignment=PP_ALIGN.RIGHT)


# ═══════════════════════════════════════════════════════════
# Slide 1: 标题页
# ═══════════════════════════════════════════════════════════
slide = add_blank_slide()
# 背景
bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
bg.fill.solid()
bg.fill.fore_color.rgb = C_PRIMARY
bg.line.fill.background()

# 装饰条
dec = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(2.8),
                              prs.slide_width, Inches(0.06))
dec.fill.solid()
dec.fill.fore_color.rgb = C_SECONDARY
dec.line.fill.background()

# 标题
add_textbox(slide, 1.5, 1.2, 10.5, 1.2,
            "密闭空间中基于TEG热电采能的WSN传感器与AP\n协同部署优化算法研究",
            font_size=36, bold=True, color=C_WHITE, alignment=PP_ALIGN.CENTER)

# 算法名称
add_textbox(slide, 1.5, 3.1, 10.5, 0.8,
            "DRL-CR-MODE：深度强化学习调控的约束修复型\n多目标差分进化算法",
            font_size=22, bold=False, color=RGBColor(0xBB, 0xDE, 0xFB),
            alignment=PP_ALIGN.CENTER)

# 汇报信息
add_textbox(slide, 1.5, 4.6, 10.5, 0.5,
            "科研进展汇报",
            font_size=18, color=C_WHITE, alignment=PP_ALIGN.CENTER)
add_textbox(slide, 1.5, 5.5, 10.5, 0.5,
            "2026年6月",
            font_size=16, color=RGBColor(0x90, 0xCA, 0xF9), alignment=PP_ALIGN.CENTER)

# 底部装饰线
dec2 = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1.5), Inches(6.3),
                               Inches(10.3), Inches(0.03))
dec2.fill.solid()
dec2.fill.fore_color.rgb = C_SECONDARY
dec2.line.fill.background()


# ═══════════════════════════════════════════════════════════
# Slide 2: 研究背景与应用场景
# ═══════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, "研究背景与应用场景")

bullets_left = [
    "**密闭空间监测需求**：工业罐体、舰船舱室、地下管廊等封闭环境需部署无线传感器网络(WSN)进行环境感知与状态监测",
    "**TEG热电采能**：密闭空间壁面存在显著温差，热电发生器(TEG)可将温差直接转换为电能，为传感器和AP持续供电，实现能量自维持",
    "**WSN部署挑战**：传感器负责感知覆盖，AP负责数据汇聚与转发，二者协同部署直接影响系统覆盖率和通信吞吐量",
    "**问题本质**：在壁面候选点中同时选择传感器和AP位置，优化覆盖率与吞吐量，同时满足温度场、能量、散热片等多重约束",
]
add_body_text(slide, 0.6, 1.4, 7.5, 5.5, bullets_left, font_size=16)

# 右侧：场景示意图文字框
shape = slide.shapes.add_shape(
    MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.5), Inches(1.5),
    Inches(4.3), Inches(5.2)
)
shape.fill.solid()
shape.fill.fore_color.rgb = C_LIGHT_BG
shape.line.color.rgb = C_SECONDARY
tf = shape.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "密闭空间 WSN 部署示意"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = C_PRIMARY
p.font.name = "Microsoft YaHei"
p.alignment = PP_ALIGN.CENTER

bullets_box = [
    "正六面体密闭空间 5×5×5 m³",
    "壁面网格离散化：间距 1.0 m",
    "候选部署点：150 个",
    "最大部署：15 传感器 + 3 AP",
    "TEG 从壁面温差采集能量",
    "传感器感知半径：Rs = 1.5 m",
    "AP 接入容量：Cmax = 10",
]
for b in bullets_box:
    p = tf.add_paragraph()
    p.text = "▸ " + b
    p.font.size = Pt(13)
    p.font.color.rgb = C_DARK
    p.font.name = "Microsoft YaHei"
    p.space_after = Pt(6)

add_page_number(slide, 2)


# ═══════════════════════════════════════════════════════════
# Slide 3: 问题难点 — 多约束耦合
# ═══════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, "问题难点：覆盖、通信、能量等多约束耦合")

constraints_data = [
    ("部署数量约束", "传感器≤15，AP≤3，总数受限但解空间为组合爆炸", C_PRIMARY),
    ("传感器/AP互斥", "同一候选点不能同时部署传感器和AP：x_i + y_i ≤ 1", C_ACCENT),
    ("通信连接约束", "每个传感器必须且仅连接一个AP，且需满足SNR门限和发射功率上限", C_PRIMARY),
    ("AP接入容量", "每个AP最多服务Cmax=10个传感器，超出则通信质量下降", C_ACCENT),
    ("能量中性约束", "P_harvest = n_sink × P_grid ≥ P_cons，节点采集能量必须≥消耗能量", C_PRIMARY),
    ("散热片占用约束", "散热片只能放置在节点邻域网格内，且互不重叠", C_ACCENT),
    ("AP非空服务约束", "已部署AP必须至少服务一个传感器，否则该AP不应部署", C_PRIMARY),
]

for i, (title, desc, color) in enumerate(constraints_data):
    col = i % 3
    row = i // 3
    left = 0.5 + col * 4.2
    top = 1.4 + row * 2.8

    # 圆角矩形卡片
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top),
        Inches(3.9), Inches(2.4)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = C_WHITE
    shape.line.color.rgb = color
    shape.line.width = Pt(2)

    # 顶部色条
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(left + 0.05), Inches(top + 0.05),
        Inches(3.8), Inches(0.05)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = color
    bar.line.fill.background()

    add_textbox(slide, left + 0.2, top + 0.2, 3.5, 0.5, title,
                font_size=16, bold=True, color=color, alignment=PP_ALIGN.CENTER)
    add_textbox(slide, left + 0.2, top + 0.8, 3.5, 1.3, desc,
                font_size=13, color=C_DARK, alignment=PP_ALIGN.CENTER)

add_textbox(slide, 0.6, 7.0, 12, 0.3,
            "▸ 核心难点：部署位置选择不仅影响覆盖率和通信吞吐量，还受到温度场分布、能量采集能力、散热片物理占位等多物理场约束的耦合限制",
            font_size=14, bold=True, color=C_ACCENT)

add_page_number(slide, 3)


# ═══════════════════════════════════════════════════════════
# Slide 4: 优化目标 — 双目标Pareto优化
# ═══════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, "优化目标：最大化覆盖率(Coverage)与系统吞吐量(Rsum)")

# 目标1
shape = slide.shapes.add_shape(
    MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(1.5),
    Inches(5.8), Inches(2.5)
)
shape.fill.solid()
shape.fill.fore_color.rgb = C_LIGHT_BG
shape.line.color.rgb = C_PRIMARY
shape.line.width = Pt(1.5)
tf = shape.text_frame
tf.word_wrap = True

p = tf.paragraphs[0]
p.text = "目标1：最大化覆盖率 (Coverage)"
p.font.size = Pt(18)
p.font.bold = True
p.font.color.rgb = C_PRIMARY
p.font.name = "Microsoft YaHei"
p.alignment = PP_ALIGN.CENTER

bullets_cov = [
    "Coverage = 被至少一个传感器覆盖的目标点数 / 总目标点数",
    "目标点：壁面上随机分布的80个空间采样点",
    "传感器感知模型：二值覆盖模型（距离≤Rs=1.5m视为覆盖）",
    "部署更多传感器 → 覆盖更多目标点，但受能量和散热片约束",
]
for b in bullets_cov:
    p = tf.add_paragraph()
    p.text = "• " + b
    p.font.size = Pt(13)
    p.font.color.rgb = C_DARK
    p.font.name = "Microsoft YaHei"
    p.space_after = Pt(4)

# 目标2
shape2 = slide.shapes.add_shape(
    MSO_SHAPE.ROUNDED_RECTANGLE, Inches(7.0), Inches(1.5),
    Inches(5.8), Inches(2.5)
)
shape2.fill.solid()
shape2.fill.fore_color.rgb = RGBColor(0xFF, 0xF3, 0xE0)
shape2.line.color.rgb = C_ORANGE
shape2.line.width = Pt(1.5)
tf2 = shape2.text_frame
tf2.word_wrap = True

p = tf2.paragraphs[0]
p.text = "目标2：最大化系统吞吐量 (Rsum)"
p.font.size = Pt(18)
p.font.bold = True
p.font.color.rgb = C_ORANGE
p.font.name = "Microsoft YaHei"
p.alignment = PP_ALIGN.CENTER

bullets_rsum = [
    "Rsum = Σ 所有传感器-AP链路的香农容量之和",
    "链路速率取决于SNR，SNR取决于发射功率、信道增益和路径损耗",
    "传感器选择信道条件最好的AP连接，受AP接入容量限制",
    "提高发射功率 → 提升吞吐量，但增加功耗 → 需要更多散热片",
]
for b in bullets_rsum:
    p = tf2.add_paragraph()
    p.text = "• " + b
    p.font.size = Pt(13)
    p.font.color.rgb = C_DARK
    p.font.name = "Microsoft YaHei"
    p.space_after = Pt(4)

# Pareto 说明
shape3 = slide.shapes.add_shape(
    MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.5), Inches(4.5),
    Inches(10.3), Inches(1.6)
)
shape3.fill.solid()
shape3.fill.fore_color.rgb = RGBColor(0xE8, 0xF5, 0xE9)
shape3.line.color.rgb = C_GREEN
tf3 = shape3.text_frame
tf3.word_wrap = True
p = tf3.paragraphs[0]
p.text = "双目标 Pareto 优化问题"
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = C_GREEN
p.font.name = "Microsoft YaHei"
p.alignment = PP_ALIGN.CENTER

pareto_bullets = [
    "Coverage ↑ 和 Rsum ↑ 存在固有冲突：分配更多资源给覆盖 → 单链路速率可能下降；追求极端吞吐量 → 可能牺牲边缘覆盖",
    "不存在唯一最优解，求解一组 Pareto 非支配解集，供决策者根据偏好选择",
    "优化问题形式：max [Coverage(x,y), Rsum(x,y)]  s.t.  g_k(x,y) ≤ 0, k=1,...,7 (约束条件)",
]
for b in pareto_bullets:
    p = tf3.add_paragraph()
    p.text = "• " + b
    p.font.size = Pt(12)
    p.font.color.rgb = C_DARK
    p.font.name = "Microsoft YaHei"
    p.space_after = Pt(3)

add_textbox(slide, 0.6, 6.5, 12, 0.5,
            "▸ 这是一个受多种工程约束限制的双目标离散组合优化问题，属于 NP-hard 问题",
            font_size=14, bold=True, color=C_ACCENT)

add_page_number(slide, 4)


# ═══════════════════════════════════════════════════════════
# Slide 5: 基础算法 MODE 原理
# ═══════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, "基础算法：MODE 多目标差分进化")

# MODE流程
flow_steps = [
    ("种群初始化", "随机+贪心混合初始化\nNP=80个体\n编码：ρ_s, ρ_a连续向量"),
    ("变异", "随机选择3个不同个体\nv = x_r1 + F·(x_r2 - x_r3)\nF=0.5 缩放因子"),
    ("交叉", "变异向量与目标向量\n按CR=0.8交叉概率\n生成试验向量"),
    ("解码+评价", "将连续编码解码为\n离散部署方案\n计算Coverage和Rsum"),
    ("Pareto选择", "基于Pareto支配关系\n选择优胜者进入下一代\n更新外部档案"),
]

for i, (name, desc) in enumerate(flow_steps):
    left = 0.3 + i * 2.6
    top = 1.5

    # 节点
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top),
        Inches(2.3), Inches(2.2)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = C_WHITE
    shape.line.color.rgb = C_PRIMARY
    shape.line.width = Pt(1.5)

    # 顶部编号
    num_shape = slide.shapes.add_shape(
        MSO_SHAPE.OVAL, Inches(left + 0.85), Inches(top - 0.25),
        Inches(0.6), Inches(0.5)
    )
    num_shape.fill.solid()
    num_shape.fill.fore_color.rgb = C_PRIMARY
    num_shape.line.fill.background()
    nf = num_shape.text_frame
    nf.paragraphs[0].text = str(i + 1)
    nf.paragraphs[0].font.size = Pt(16)
    nf.paragraphs[0].font.bold = True
    nf.paragraphs[0].font.color.rgb = C_WHITE
    nf.paragraphs[0].font.name = "Microsoft YaHei"
    nf.paragraphs[0].alignment = PP_ALIGN.CENTER

    add_textbox(slide, left + 0.1, top + 0.35, 2.1, 0.45, name,
                font_size=14, bold=True, color=C_PRIMARY, alignment=PP_ALIGN.CENTER)
    add_textbox(slide, left + 0.1, top + 0.85, 2.1, 1.2, desc,
                font_size=11, color=C_DARK, alignment=PP_ALIGN.CENTER)

    # 箭头
    if i < 4:
        add_arrow(slide, left + 2.3, top + 0.85, 0.3, 0.3)

# 说明
add_textbox(slide, 0.6, 4.2, 12, 1.5,
            "MODE 的核心机制：通过差分变异探索解空间，Pareto非支配排序逐代搜索两个目标之间的折中解集",
            font_size=15, bold=True, color=C_DARK)

add_textbox(slide, 0.6, 4.8, 12, 1.5,
            "▸ 种群规模 NP=80，最大代数 Tmax=150，外部 Pareto 档案容量 200\n"
            "▸ 变异策略：standard_rand；F_init=0.5 (范围 0.3–0.9)；CR_init=0.8 (范围 0.1–0.9)\n"
            "▸ Pareto 档案维护：拥挤距离剪枝，保证解的多样性和均匀分布",
            font_size=13, color=C_GRAY)

add_textbox(slide, 0.6, 6.0, 12, 0.8,
            "▸ 关键问题：MODE本身只负责在解空间中搜索非支配解，不了解复杂工程约束，容易产生大量不可行解！",
            font_size=15, bold=True, color=C_ACCENT)

add_page_number(slide, 5)


# ═══════════════════════════════════════════════════════════
# Slide 6: 普通 MODE 的不足
# ═══════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, "普通MODE直接用于本问题的不足")

bullets = [
    "**不了解工程约束**：MODE只负责在连续空间中搜索，变异和交叉产生的个体解码后可能违反多种约束，例如部署过量传感器、AP空置、能量赤字等",
    "**可行解比例极低**：随机初始化+无约束引导搜索 → 大部分个体不可行 → 种群进化方向迷失 → Pareto前沿质量差",
    "**约束耦合复杂**：能量约束涉及TEG采能、功耗、散热片数量三者耦合；通信约束涉及SNR、连接关系、AP容量三者耦合；单一惩罚函数难以有效引导搜索",
    "**离散-连续混合**：部署位置是离散选择(C(150, 15)种组合)，但差分进化操作在连续空间 → 解码到离散空间后可能丢失优秀解信息",
    "**目标冲突加剧**：若无约束修复，搜索偏向保守解（如少部署节点），但牺牲了覆盖率和吞吐量",
]
add_body_text(slide, 0.6, 1.4, 12, 4.0, bullets, font_size=16)

# 对比示意图
shape = slide.shapes.add_shape(
    MSO_SHAPE.ROUNDED_RECTANGLE, Inches(2.0), Inches(5.2),
    Inches(9.3), Inches(1.8)
)
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(0xFF, 0xEB, 0xEE)
shape.line.color.rgb = C_ACCENT
tf = shape.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "结论：需要一种机制，将工程约束知识嵌入到 MODE 搜索过程中"
p.font.size = Pt(18)
p.font.bold = True
p.font.color.rgb = C_ACCENT
p.font.name = "Microsoft YaHei"
p.alignment = PP_ALIGN.CENTER

p = tf.add_paragraph()
p.text = "→ 在每个个体解码后，对不可行个体进行闭环约束修复，使其尽可能变为可行解 → CR-MODE"
p.font.size = Pt(15)
p.font.color.rgb = C_DARK
p.font.name = "Microsoft YaHei"
p.alignment = PP_ALIGN.CENTER

add_page_number(slide, 6)


# ═══════════════════════════════════════════════════════════
# Slide 7: CR-MODE 改进思想
# ═══════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, "CR-MODE：约束修复型多目标差分进化算法")

add_textbox(slide, 0.6, 1.3, 12, 0.5,
            "核心思想：MODE 搜索 + 解码 + 约束修复 + 重新评价 + Pareto 更新",
            font_size=20, bold=True, color=C_PRIMARY)

# 改进点卡片
improvements = [
    ("约束修复闭环", "每个个体解码为部署方案后，\n按固定顺序对部署、通信、\n容量、能量、散热片、AP服务\n逐项检查和修复",
     C_PRIMARY),
    ("修复后重新评价", "修复后的解重新计算CV和\n目标函数值，确保评价的是\n修复后的真实质量",
     C_SECONDARY),
    ("修复结果反馈种群", "修复后的解参与Pareto选择；\n若部署改变，反馈修正\n个体编码的ρ_s, ρ_a值\n引导后续搜索方向",
     C_GREEN),
    ("迭代修复直至可行", "最多5次闭环修复迭代，\n每次修复后检查CV是否下降，\n不下降则停止，避免\n修复引入新的约束违反",
     C_ORANGE),
]

for i, (title, desc, color) in enumerate(improvements):
    left = 0.4 + i * 3.2
    top = 2.1

    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top),
        Inches(2.9), Inches(3.2)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = C_WHITE
    shape.line.color.rgb = color
    shape.line.width = Pt(2)

    # 顶部色块
    header = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(left), Inches(top),
        Inches(2.9), Inches(0.6)
    )
    header.fill.solid()
    header.fill.fore_color.rgb = color
    header.line.fill.background()
    add_textbox(slide, left + 0.1, top + 0.05, 2.7, 0.5, title,
                font_size=16, bold=True, color=C_WHITE, alignment=PP_ALIGN.CENTER)

    add_textbox(slide, left + 0.15, top + 0.8, 2.6, 2.2, desc,
                font_size=13, color=C_DARK, alignment=PP_ALIGN.LEFT)

# CR-MODE 与 MODE 对比
comp_shape = slide.shapes.add_shape(
    MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(5.6),
    Inches(12.3), Inches(1.6)
)
comp_shape.fill.solid()
comp_shape.fill.fore_color.rgb = C_LIGHT_BG
comp_shape.line.color.rgb = C_PRIMARY
tf = comp_shape.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "普通 MODE：搜索 → 解码 → 评价 → 选择 → (可能大量不可行解混入Pareto档案)"
p.font.size = Pt(13)
p.font.color.rgb = C_GRAY
p.font.name = "Microsoft YaHei"
p.space_after = Pt(4)

p = tf.add_paragraph()
p.text = "CR-MODE：  搜索 → 解码 → 约束修复 → 重新评价 → CV计算 → 可行解鉴定 → Pareto选择"
p.font.size = Pt(13)
p.font.bold = True
p.font.color.rgb = C_PRIMARY
p.font.name = "Microsoft YaHei"
p.space_after = Pt(4)

p = tf.add_paragraph()
p.text = "关键区别：CR-MODE不是简单惩罚不可行解，而是[修复]不可行解，使其变为可行解后再参与进化"
p.font.size = Pt(13)
p.font.color.rgb = C_ACCENT
p.font.name = "Microsoft YaHei"

add_page_number(slide, 7)


# ═══════════════════════════════════════════════════════════
# Slide 8: CR-MODE 约束修复流程图
# ═══════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, "CR-MODE 约束修复流程（Constraint Repair Pipeline）")

# 流程图
flow_items = [
    ("部署解码", "ρ_s, ρ_a\n→ x, y", C_PRIMARY),
    ("通信连接", "分配传感器\n→ AP连接", C_SECONDARY),
    ("功率分配", "分配发射\n功率p_tx", C_PRIMARY),
    ("能量检查", "检查P_harvest\n≥ P_cons", C_ACCENT),
    ("散热片分配", "分配散热片\n满足能量需求", C_ORANGE),
    ("AP非空检查", "删除无服务\n的空AP", C_GREEN),
    ("CV计算", "综合约束\n违反量评估", C_PRIMARY),
    ("可行解评价", "计算Coverage\n和Rsum", C_GREEN),
]

for i, (name, desc, color) in enumerate(flow_items):
    left = 0.2 + i * 1.6
    top = 1.6

    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top),
        Inches(1.4), Inches(2.0)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()

    add_textbox(slide, left + 0.05, top + 0.15, 1.3, 0.5, name,
                font_size=13, bold=True, color=C_WHITE, alignment=PP_ALIGN.CENTER)
    add_textbox(slide, left + 0.05, top + 0.75, 1.3, 1.0, desc,
                font_size=11, color=RGBColor(0xF5, 0xF5, 0xF5),
                alignment=PP_ALIGN.CENTER)

    # 箭头
    if i < 7:
        add_arrow(slide, left + 1.4, top + 0.75, 0.2, 0.2)

# 循环反馈
feedback_box = slide.shapes.add_shape(
    MSO_SHAPE.ROUNDED_RECTANGLE, Inches(3.5), Inches(4.2),
    Inches(6.3), Inches(1.0)
)
feedback_box.fill.solid()
feedback_box.fill.fore_color.rgb = RGBColor(0xFF, 0xF8, 0xE1)
feedback_box.line.color.rgb = C_ORANGE
feedback_box.line.width = Pt(1.5)
tf = feedback_box.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "迭代修复循环（最多5轮）：CV下降则继续 → CV不下降或可行则停止 → 避免修复引入新违反"
p.font.size = Pt(14)
p.font.bold = True
p.font.color.rgb = C_ORANGE
p.font.name = "Microsoft YaHei"
p.alignment = PP_ALIGN.CENTER

# 修复顺序说明
repair_order = [
    "① 部署修复：数量超限剪枝 + 传感器/AP互斥消解",
    "② 链路修复：清理无效连接 + 传感器一对一重连 + 功率上限修正",
    "③ 容量修复：AP超载时保留高速率传感器连接",
    "④ 能量修复：重算功率与散热片需求 + 增加散热片 + 不足则降功率或删节点",
    "⑤ AP服务修复：删除未连接任何传感器的空AP",
    "⑥ 散热片终分配：确保散热片分配与当前部署一致",
]
add_body_text(slide, 0.6, 5.5, 12, 1.8, repair_order, font_size=12, line_spacing=1.3)

add_page_number(slide, 8)


# ═══════════════════════════════════════════════════════════
# Slide 9: DRL-CR-MODE 总体框架
# ═══════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, "DRL-CR-MODE：深度强化学习调控的约束修复型MODE（最终算法）")

add_textbox(slide, 0.6, 1.3, 12, 0.5,
            "最终算法 = CR-MODE（基础搜索+约束修复） + DRL策略控制器（外层智能调控）",
            font_size=18, bold=True, color=C_PRIMARY)

# 三列结构
cols = [
    ("CR-MODE 基础层", [
        "种群初始化（混合策略）",
        "差分变异 + 二项交叉",
        "解码为离散部署方案",
        "闭环约束修复（6项）",
        "约束违反评价 + Pareto选择",
        "外部Pareto档案维护",
    ], C_PRIMARY),
    ("DRL 策略控制层", [
        "智能体观察种群状态",
        "DQN网络决策动作",
        "动态调整F、CR参数",
        "切换变异/交叉策略",
        "调整修复/功率策略",
        "引导跳出局部最优",
    ], C_SECONDARY),
    ("信息交互接口", [
        "状态s：FR, CV, Coverage,",
        "  Rsum, Pareto改善量",
        "动作a：调整参数/策略",
        "奖励r：Pareto改善",
        "  + FR提升 - CV惩罚",
        "周期性触发DRL决策",
        "  （每5-10代一次）",
    ], C_GREEN),
]

for i, (title, items, color) in enumerate(cols):
    left = 0.4 + i * 4.2
    top = 2.0

    # 列标题
    header = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top),
        Inches(3.9), Inches(0.5)
    )
    header.fill.solid()
    header.fill.fore_color.rgb = color
    header.line.fill.background()
    add_textbox(slide, left + 0.1, top + 0.05, 3.7, 0.4, title,
                font_size=15, bold=True, color=C_WHITE, alignment=PP_ALIGN.CENTER)

    # 内容
    body = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top + 0.5),
        Inches(3.9), Inches(3.2)
    )
    body.fill.solid()
    body.fill.fore_color.rgb = C_WHITE
    body.line.color.rgb = color
    body.line.width = Pt(1.5)
    tf = body.text_frame
    tf.word_wrap = True
    for j, item in enumerate(items):
        if j == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = "• " + item
        p.font.size = Pt(12)
        p.font.color.rgb = C_DARK
        p.font.name = "Microsoft YaHei"
        p.space_after = Pt(3)

    # 箭头连接
    if i < 2:
        add_arrow(slide, left + 3.9, top + 1.8, 0.3, 0.3)

add_textbox(slide, 0.6, 6.0, 12, 1.2,
            "▸ DRL-CR-MODE 的优势：CR-MODE保证解可行性 → DRL动态调控搜索策略 → 自动化超参调节 → 提升Pareto前沿质量和收敛速度\n"
            "▸ 当前阶段：先展示CR-MODE第一阶段结果，验证约束修复型MODE能否解决部署问题，为后续DRL模块接入提供基线（baseline）",
            font_size=14, bold=False, color=C_DARK)

add_page_number(slide, 9)


# ═══════════════════════════════════════════════════════════
# Slide 10: DRL 作用原理
# ═══════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, "DRL策略控制器：状态、动作与奖励设计")

# 三列：State, Action, Reward
sections = [
    ("状态空间 S", [
        "FR (可行解比例) ∈ [0,1]",
        "CV_mean (平均约束违反量)",
        "CV_min (最小约束违反量)",
        "Coverage_best (最优覆盖率)",
        "Rsum_best (最优吞吐量)",
        "Pareto_front_size (前沿规模)",
        "Delta_HV (超体积改善量)",
        "Population_diversity (多样性)",
        "Gen_progress (进化进度 t/Tmax)",
    ], C_PRIMARY),
    ("动作空间 A", [
        "ΔF ∈ {-0.05, 0, +0.05}",
        "ΔCR ∈ {-0.1, 0, +0.1}",
        "切换变异策略:",
        "  rand/1, best/1, current-to-best/1",
        "切换修复策略:",
        "  aggressive / balanced / conservative",
        "切换功率策略:",
        "  max_throughput / energy_balanced / conservative",
    ], C_SECONDARY),
    ("奖励函数 R", [
        "R = w1·ΔHV (超体积改善)",
        "   + w2·ΔFR (可行解比例提升)",
        "   + w3·ΔCoverage_best",
        "   + w4·ΔRsum_best",
        "   - w5·ΔCV (约束违反增加惩罚)",
        "   - w6·stagnation (停滞惩罚)",
        "引导智能体学习何时探索、",
        "何时收敛、何时调整策略",
    ], C_GREEN),
]

for i, (title, items, color) in enumerate(sections):
    left = 0.4 + i * 4.2
    top = 1.4

    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top),
        Inches(3.9), Inches(5.3)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = C_WHITE
    shape.line.color.rgb = color
    shape.line.width = Pt(2)

    # 标题
    header = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(left), Inches(top),
        Inches(3.9), Inches(0.5)
    )
    header.fill.solid()
    header.fill.fore_color.rgb = color
    header.line.fill.background()
    add_textbox(slide, left + 0.1, top + 0.05, 3.7, 0.4, title,
                font_size=16, bold=True, color=C_WHITE, alignment=PP_ALIGN.CENTER)

    tf = shape.text_frame
    tf.word_wrap = True
    for j, item in enumerate(items):
        if j == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = "• " + item
        p.font.size = Pt(11)
        p.font.color.rgb = C_DARK
        p.font.name = "Microsoft YaHei"
        p.space_after = Pt(2)

add_textbox(slide, 0.6, 6.9, 12, 0.3,
            "▸ DQN网络结构：3层全连接MLP + ReLU + 经验回放 + 目标网络 + ε-greedy探索",
            font_size=13, bold=False, color=C_GRAY)

add_page_number(slide, 10)


# ═══════════════════════════════════════════════════════════
# Slide 11: 第一阶段实验说明
# ═══════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, "第一阶段实验：CR-MODE 基础模块验证")

add_textbox(slide, 0.6, 1.3, 12, 0.5,
            "为什么先跑 CR-MODE，而不是直接跑完整 DRL-CR-MODE？",
            font_size=20, bold=True, color=C_PRIMARY)

reasons = [
    ("先验证约束修复有效性",
     "核心问题：约束修复型MODE能否在密闭空间TEG部署问题中生成足够比例的可行解？\n"
     "若CR-MODE可行解比例极低，说明修复机制本身需要改进，加入DRL也无意义",
     C_PRIMARY),
    ("建立性能基线",
     "CR-MODE的结果作为baseline，后续DRL-CR-MODE必须优于CR-MODE才有意义\n"
     "消融对比：CR-MODE vs DRL-CR-MODE → 量化DRL策略调控的贡献",
     C_SECONDARY),
    ("分离修复效果与调控效果",
     "若直接跑DRL-CR-MODE，无法区分可行解的提升来自约束修复还是DRL智能调控\n"
     "分阶段实验 → 清晰归因 → 更严谨的科研论证",
     C_GREEN),
    ("降低调试复杂度",
     "CR-MODE参数少（F、CR、NP、Tmax），调优相对简单\n"
     "在此基础上加入DRL，可以专注于DRL的state/action/reward设计调优",
     C_ORANGE),
]

for i, (title, desc, color) in enumerate(reasons):
    left = 0.4 + (i % 2) * 6.3
    top = 2.0 + (i // 2) * 2.6

    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top),
        Inches(5.9), Inches(2.2)
    )
    shape.fill.solid()
    shape.fill.fore_color.rgb = C_WHITE
    shape.line.color.rgb = color
    shape.line.width = Pt(2)

    add_textbox(slide, left + 0.2, top + 0.1, 5.5, 0.4, f"理由 {i+1}：{title}",
                font_size=16, bold=True, color=color)
    add_textbox(slide, left + 0.2, top + 0.7, 5.5, 1.3, desc,
                font_size=13, color=C_DARK, line_spacing=1.4)

add_page_number(slide, 11)


# ═══════════════════════════════════════════════════════════
# Slide 12: 实验场景与参数
# ═══════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, "实验场景与参数设置")

# 参数表格
params = [
    ("空间尺寸", "5.0 × 5.0 × 5.0 m³ 正六面体"),
    ("网格间距", "1.0 m"),
    ("候选部署点", "150 个（6面 × 约25点/面）"),
    ("目标点", "80 个（随机分布壁面）"),
    ("传感器感知半径", "Rs = 1.5 m"),
    ("最大传感器数", "15 个"),
    ("最大AP数", "3 个"),
    ("AP接入容量", "Cmax = 10 个传感器/AP"),
    ("温度场模型", "局部表面最高温模型，2-3个热源，热源温度326.15K"),
    ("环境温度", "T_amb = 295.15 K, T_max = 330.15 K"),
    ("TEG模型", "热阻模型：R_TEG=0.12 K/W, R_int=0.8 K/W"),
    ("信道模型", "κ-μ衰落 (κ=3.0, μ=2.0)，载波2.4 GHz，路径损耗指数α=2.0"),
    ("SNR门限", "γ_th = 10 dB"),
    ("发射功率上限", "P_tx_max = 0.5 W"),
    ("MODE参数", "NP=80, Tmax=150, F∈[0.3,0.9], CR∈[0.1,0.9]"),
    ("约束修复", "固定顺序修复，最多5次迭代，CV<0.05为可行"),
]

left_col = params[:8]
right_col = params[8:]

for side, items, x_start in [("left", left_col, 0.4), ("right", right_col, 6.7)]:
    table_shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x_start), Inches(1.4),
        Inches(6.2), Inches(len(items) * 0.52 + 0.1)
    )
    table_shape.fill.solid()
    table_shape.fill.fore_color.rgb = C_WHITE
    table_shape.line.color.rgb = C_PRIMARY
    table_shape.line.width = Pt(1)

    for i, (key, val) in enumerate(items):
        y = 1.5 + i * 0.52
        add_textbox(slide, x_start + 0.15, y, 2.3, 0.45, key,
                    font_size=11, bold=True, color=C_PRIMARY)
        add_textbox(slide, x_start + 2.4, y, 3.6, 0.45, val,
                    font_size=11, color=C_DARK)

add_page_number(slide, 12)


# ═══════════════════════════════════════════════════════════
# Slide 13: 温度分布与候选点筛选
# ═══════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, "温度分布与候选点筛选结果")

add_textbox(slide, 0.6, 1.3, 12, 0.5,
            "基于ANSYS温度场数据 + 候选点筛选预处理",
            font_size=16, bold=False, color=C_GRAY)

# 候选点分类
add_textbox(slide, 0.6, 1.8, 5.5, 0.4, "候选点分类统计", font_size=18, bold=True, color=C_PRIMARY)

cand_bullets = [
    "**总候选点数**：150 个（网格间距1.0m，六面体6个内表面）",
    "**传感器候选点 Ls**：76 个（排除温度过低无法供能的点）",
    "**AP候选点 La**：29 个（排除无通信可达性的点）",
    "**筛选策略**：基于TEG采能模型和链路可行性矩阵联合筛选",
    "**温度范围**：295.15 K（环境）~ 330.15 K（热源附近）",
    "**关键观察**：热源附近候选点采能功率高，但散热片竞争激烈",
]
add_body_text(slide, 0.6, 2.3, 5.8, 4.5, cand_bullets, font_size=13)

# 右边插入部署图作为场景展示
add_image_safe(slide, "results/figures/deployment.png", 6.8, 1.6, 6.2, 5.0)

add_page_number(slide, 13)


# ═══════════════════════════════════════════════════════════
# Slide 14: 收敛结果
# ═══════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, "CR-MODE 收敛曲线：可行解比例 FR 与约束违反 CV")

# FR图
add_image_safe(slide, "results/figures/convergence_fr.png", 0.3, 1.3, 6.3, 2.8)
# CV图
add_image_safe(slide, "results/figures/convergence_cv.png", 6.9, 1.3, 6.3, 2.8)

# Coverage 和 Rsum
add_image_safe(slide, "results/figures/convergence_coverage.png", 0.3, 4.3, 6.3, 2.8)
add_image_safe(slide, "results/figures/convergence_rsum.png", 6.9, 4.3, 6.3, 2.8)

add_page_number(slide, 14)


# ═══════════════════════════════════════════════════════════
# Slide 15: Pareto前沿图
# ═══════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, "CR-MODE Pareto 前沿：Coverage 与 Rsum 的权衡关系")

# 两个Pareto图
add_image_safe(slide, "results/figures/pareto_cr_mode.png", 0.3, 1.3, 6.3, 5.5)
add_image_safe(slide, "results/figures/pareto_cr_mode_front.png", 6.9, 1.3, 6.3, 5.5)

add_textbox(slide, 0.6, 7.0, 12, 0.3,
            "▸ 左：Pareto档案全部可行解  |  右：去重后的非支配前沿（带数值标注）",
            font_size=12, color=C_GRAY)

add_page_number(slide, 15)


# ═══════════════════════════════════════════════════════════
# Slide 16: 空间部署图
# ═══════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, "Pareto解的空间部署可视化")

# 三张部署图
add_image_safe(slide, "results/figures/selected_solution_max_Coverage_3d.png", 0.15, 1.2, 4.3, 3.0)
add_image_safe(slide, "results/figures/selected_solution_knee_3d.png", 4.55, 1.2, 4.3, 3.0)
add_image_safe(slide, "results/figures/selected_solution_max_Rsum_3d.png", 8.95, 1.2, 4.3, 3.0)

# 标签
labels = ["最大覆盖率解 (Max Coverage)", "膝点解 (Knee Point)", "最大吞吐量解 (Max Rsum)"]
for i, lbl in enumerate(labels):
    add_textbox(slide, 0.15 + i * 4.4, 4.3, 4.3, 0.4, lbl,
                font_size=14, bold=True, color=C_PRIMARY, alignment=PP_ALIGN.CENTER)

# 图例说明
legend_items = [
    "● 传感器 (红点) — 部署在壁面网格上",
    "▲ AP (蓝三角) — 负责数据汇聚",
    "★ 热源 (黄星) — 高温区域，TEG采能功率高",
    "橙色面片 — 传感器散热片占位",
    "蓝色面片 — AP散热片占位",
    "灰色虚线 — 传感器→AP通信链路",
    "背景色块 — 壁面温度场 (蓝冷→红热)",
]
for i, item in enumerate(legend_items):
    col = i // 3
    row = i % 3
    add_textbox(slide, 0.5 + col * 6.3, 4.9 + row * 0.55, 6.0, 0.5, "• " + item,
                font_size=11, color=C_DARK)

add_page_number(slide, 16)


# ═══════════════════════════════════════════════════════════
# Slide 17: 总结
# ═══════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, "CR-MODE 阶段结论")

conclusions = [
    ("可行解生成能力",
     "CR-MODE成功在150代内收敛到全部可行的Pareto解集，Pareto档案中所有解均为可行解(CV<0.05)；约束修复机制有效将不可行个体转化为可行个体",
     C_GREEN),
    ("约束修复有效性",
     "6项约束修复按固定顺序（部署→通信→容量→能量→散热片→AP服务）闭环执行，修复后CV快速下降；迭代修复最多5轮可达到可行状态，证明修复策略适合本问题",
     C_PRIMARY),
    ("Coverage与Rsum权衡",
     "Pareto前沿清晰展示了覆盖率与吞吐量之间的权衡关系：高覆盖率解需要更多传感器覆盖边缘区域，但单链路速率受限；高吞吐量解优先在高温区域部署，牺牲部分边缘覆盖",
     C_SECONDARY),
    ("部署合理性",
     "传感器倾向部署在热源附近（高采能功率），AP倾向部署在几何中心（最大化通信可达性）；散热片分布在节点邻域网格内，符合能量中性约束",
     C_ORANGE),
]

for i, (title, desc, color) in enumerate(conclusions):
    top = 1.4 + i * 1.5
    # 编号
    num_shape = slide.shapes.add_shape(
        MSO_SHAPE.OVAL, Inches(0.5), Inches(top + 0.1),
        Inches(0.5), Inches(0.5)
    )
    num_shape.fill.solid()
    num_shape.fill.fore_color.rgb = color
    num_shape.line.fill.background()
    nf = num_shape.text_frame
    nf.paragraphs[0].text = str(i + 1)
    nf.paragraphs[0].font.size = Pt(18)
    nf.paragraphs[0].font.bold = True
    nf.paragraphs[0].font.color.rgb = C_WHITE
    nf.paragraphs[0].font.name = "Microsoft YaHei"
    nf.paragraphs[0].alignment = PP_ALIGN.CENTER

    add_textbox(slide, 1.2, top, 3.0, 0.45, title,
                font_size=18, bold=True, color=color)
    add_textbox(slide, 1.2, top + 0.5, 11.5, 0.85, desc,
                font_size=13, color=C_DARK, line_spacing=1.4)

add_page_number(slide, 17)


# ═══════════════════════════════════════════════════════════
# Slide 18: 后续工作
# ═══════════════════════════════════════════════════════════
slide = add_blank_slide()
add_title_bar(slide, "后续工作：DRL-CR-MODE 完整算法实现与消融对比")

future_work = [
    ("接入DRL策略控制器",
     "实现DQN智能体，根据种群状态（FR、CV、Coverage、Rsum）动态调整MODE的F、CR、变异策略和修复策略；训练智能体学会何时探索、何时收敛、何时切换策略",
     C_PRIMARY),
    ("消融对比实验",
     "CR-MODE vs DRL-CR-MODE，量化DRL策略调控对Pareto前沿质量、收敛速度、可行解比例的提升效果；验证DRL模块的独立贡献",
     C_SECONDARY),
    ("超参数敏感性分析",
     "对比不同DRL触发周期、epsilon-greedy衰减策略、奖励权重配置下的算法性能，选择最优超参数组合",
     C_GREEN),
    ("大规模场景扩展",
     "将算法扩展到更大空间（如10x10x10 m^3）、更多候选点、更多传感器/AP的场景，验证算法的可扩展性和泛化能力",
     C_ORANGE),
    ("与其他算法对比",
     "与NSGA-II-CR、MOPSO-CR等约束修复型基线算法进行公平对比，从Pareto前沿质量、收敛速度、可行解比例等维度全面评估",
     C_ACCENT),
]

for i, (title, desc, color) in enumerate(future_work):
    top = 1.3 + i * 1.2
    # 编号
    num_shape = slide.shapes.add_shape(
        MSO_SHAPE.OVAL, Inches(0.5), Inches(top + 0.05),
        Inches(0.45), Inches(0.45)
    )
    num_shape.fill.solid()
    num_shape.fill.fore_color.rgb = color
    num_shape.line.fill.background()
    nf = num_shape.text_frame
    nf.paragraphs[0].text = str(i + 1)
    nf.paragraphs[0].font.size = Pt(16)
    nf.paragraphs[0].font.bold = True
    nf.paragraphs[0].font.color.rgb = C_WHITE
    nf.paragraphs[0].font.name = "Microsoft YaHei"
    nf.paragraphs[0].alignment = PP_ALIGN.CENTER

    add_textbox(slide, 1.2, top - 0.05, 3.5, 0.45, title,
                font_size=17, bold=True, color=color)
    add_textbox(slide, 1.2, top + 0.4, 11.5, 0.7, desc,
                font_size=12, color=C_DARK, line_spacing=1.3)

# 时间线
add_textbox(slide, 0.6, 7.0, 12, 0.3,
            "▸ 计划时间线：CR-MODE调优完成 → DQN训练与调参 → 消融实验 → 对比实验 → 论文撰写",
            font_size=13, bold=True, color=C_PRIMARY)

add_page_number(slide, 18)


# ── 保存 ──
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
prs.save(OUT_PATH)
print(f"PPT已生成：{OUT_PATH}")
print(f"共 {len(prs.slides)} 页")
