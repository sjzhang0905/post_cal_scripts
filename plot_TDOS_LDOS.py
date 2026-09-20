#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import re
import glob
import os
import matplotlib.pyplot as plt
import argparse

# -------- 用户可调参数 --------
TDOS_FILE = "TDOS.dat"        # 自旋体系: 1列能量, 2列UP, 3列DW；非自旋: 1列能量, 2列总DOS
PDOS_NONSPIN_PATTERN = "PDOS_*.dat"   # 不含 _UP/_DW 的非自旋文件
PDOS_SPIN_UP_PATTERN  = "PDOS_*_UP.dat"
PDOS_SPIN_DW_PATTERN  = "PDOS_*_DW.dat"

GAUSS_SIGMA_POINTS = 2.0      # 对 y 进行高斯平滑的 σ（单位：数据点数, 设为 0 或 None 取消）
LINEWIDTH = 1.2
MIRROR_SPIN = True           # 默认：不镜像；若 True，自旋下(DW)镜像到负方向
UP_LINESTYLE = '-'            # 自旋向上线型
DW_LINESTYLE = '-'           # 自旋向下线型（与 UP 同色）

XRANGE = (-6, 4)              # x 轴范围
YMAX_FIXED = 50            # y 轴上限；为 None 时自动估计；MIRROR_SPIN=True 时上下对称
# --------------------------------

def gaussian_smooth_points(y, sigma_pts):
    """对 y 进行高斯平滑（σ为点数）。"""
    if sigma_pts is None or sigma_pts <= 0:
        return y
    half_w = int(np.ceil(6 * sigma_pts))
    xx = np.arange(-half_w, half_w + 1, dtype=float)
    kernel = np.exp(-0.5 * (xx / sigma_pts) ** 2)
    kernel /= kernel.sum()
    return np.convolve(y, kernel, mode="same")

def load_tdos_dat_maybe_spin(fname, skiprows=1):
    """从 TDOS.dat 加载数据：
       - 若列数 >= 3: 视为 (E, UP, DW) 自旋
       - 若列数 == 2: 视为 (E, TOTAL) 非自旋
       返回 (mode, x, dict_y)
    """
    data = np.loadtxt(fname, skiprows=skiprows)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    ncol = data.shape[1]
    x = data[:, 0]
    if ncol >= 3:
        y_up = data[:, 1]
        y_dw = data[:, 2]
        return 'spin', x, {'up': y_up, 'dw': y_dw}
    elif ncol >= 2:
        y = data[:, 1]
        return 'nonspin', x, {'total': y}
    else:
        raise ValueError(f"{fname} 列数不足，无法解析。")

def load_xy_two_cols(fname, x_col=0, y_col=1, skiprows=1):
    data = np.loadtxt(fname, skiprows=skiprows)
    x = data[:, x_col]
    y = data[:, y_col]
    return x, y

def load_tdos_any():
    """优先使用 TDOS_FILE；若为三列则按自旋解析；否则尝试 TDOS_UP/DW；
       返回 (mode, x, dict_y)
       mode: 'nonspin' | 'spin' | 'none'
    """
    if os.path.exists(TDOS_FILE):
        try:
            mode, x, ydict = load_tdos_dat_maybe_spin(TDOS_FILE, skiprows=1)
            return mode, x, ydict
        except Exception:
            pass  # 若 TDOS.dat 异常，继续其他途径

    up = "TDOS_UP.dat"
    dw = "TDOS_DW.dat"
    if os.path.exists(up) and os.path.exists(dw):
        x_up, y_up = load_xy_two_cols(up, x_col=0, y_col=1, skiprows=1)
        x_dw, y_dw = load_xy_two_cols(dw, x_col=0, y_col=1, skiprows=1)
        if len(x_up) != len(x_dw) or np.max(np.abs(x_up - x_dw)) > 1e-12:
            y_dw = np.interp(x_up, x_dw, y_dw)
        return 'spin', x_up, {'up': y_up, 'dw': y_dw}

    if os.path.exists(TDOS_FILE):
        # 如果上面的解析失败，这里再尝试按两列读取
        x, y = load_xy_two_cols(TDOS_FILE, x_col=0, y_col=1, skiprows=1)
        return 'nonspin', x, {'total': y}

    return 'none', None, {}

def extract_elem_nonspin(fname):
    # 匹配 PDOS_Xx.dat（不含 _UP/_DW）
    m = re.match(r".*PDOS_([A-Za-z][a-z]?)\.dat$", fname)
    return m.group(1) if m else None

def extract_elem_spin(fname):
    # 匹配 PDOS_Xx_UP.dat / PDOS_Xx_DW.dat
    m = re.match(r".*PDOS_([A-Za-z][a-z]?)_(UP|DW)\.dat$", fname)
    if m:
        elem = m.group(1)
        tag = m.group(2)  # 'UP' or 'DW'
        return elem, tag
    return None, None

def ensure_on_grid(x_ref, x_src, y_src):
    """若网格不同，插值到 x_ref 上。"""
    if len(x_ref) != len(x_src) or np.max(np.abs(x_ref - x_src)) > 1e-12:
        return np.interp(x_ref, x_src, y_src)
    return y_src

def collect_pdos(x_grid):
    """收集 PDOS，返回结构化数据：
       {
         'spin': True/False,
         'elems': {
             'Fe': {'up': y1, 'dw': y2} 或 {'total': y}
             ...
         }
       }
    """
    result = {'spin': False, 'elems': {}}

    # 先搜 spin 文件
    up_files = sorted(glob.glob(PDOS_SPIN_UP_PATTERN))
    dw_files = sorted(glob.glob(PDOS_SPIN_DW_PATTERN))

    up_map = {}
    for f in up_files:
        elem, tag = extract_elem_spin(f)
        if elem and tag == 'UP':
            up_map[elem] = f
    dw_map = {}
    for f in dw_files:
        elem, tag = extract_elem_spin(f)
        if elem and tag == 'DW':
            dw_map[elem] = f

    spin_elems = sorted(set(up_map.keys()) & set(dw_map.keys()))
    if spin_elems:
        result['spin'] = True

    # 加载 spin elems
    for elem in spin_elems:
        x_up, y_up = load_xy_two_cols(up_map[elem], x_col=0, y_col=-1, skiprows=1)
        x_dw, y_dw = load_xy_two_cols(dw_map[elem], x_col=0, y_col=-1, skiprows=1)
        y_up = ensure_on_grid(x_grid, x_up, y_up)
        y_dw = ensure_on_grid(x_grid, x_dw, y_dw)
        y_up = gaussian_smooth_points(y_up, GAUSS_SIGMA_POINTS)
        y_dw = gaussian_smooth_points(y_dw, GAUSS_SIGMA_POINTS)
        result['elems'][elem] = {'up': y_up, 'dw': y_dw}

    # 再搜 non-spin（排除已经有 spin 的元素）
    for f in sorted(glob.glob(PDOS_NONSPIN_PATTERN)):
        # 跳过含 _UP/_DW 的文件
        if re.search(r"_(UP|DW)\.dat$", f):
            continue
        elem = extract_elem_nonspin(f)
        if not elem or elem in result['elems']:
            continue
        x_p, y_p = load_xy_two_cols(f, x_col=0, y_col=-1, skiprows=1)
        y_p = ensure_on_grid(x_grid, x_p, y_p)
        y_p = gaussian_smooth_points(y_p, GAUSS_SIGMA_POINTS)
        result['elems'][elem] = {'total': y_p}

    return result

def assign_colors(elems_order):
    """为元素分配颜色（同元素 UP/DW 共用同色）。"""
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    colors = {}
    for i, elem in enumerate(elems_order):
        colors[elem] = color_cycle[i % len(color_cycle)] if color_cycle else None
    return colors

def main():
    # --- CLI: optional ymax overrides YMAX_FIXED ---
    # Usage:
    #   python plot_dos.py        # uses default YMAX_FIXED
    #   python plot_dos.py 120    # sets YMAX_FIXED=120
    global YMAX_FIXED
    parser = argparse.ArgumentParser(
        description='Plot TDOS/PDOS with optional fixed y-axis limit (YMAX_FIXED).'
    )
    parser.add_argument(
        'ymax', nargs='?', type=float,
        help='Optional y-axis max to override YMAX_FIXED (positive number).'
    )
    args = parser.parse_args()
    if args.ymax is not None:
        if args.ymax <= 0:
            raise ValueError('ymax must be a positive number')
        YMAX_FIXED = args.ymax

    # 1) 读取 TDOS（自动检测 spin 与非 spin，包括三列 TDOS.dat 场景）
    tdos_mode, x_t, tdos_data = load_tdos_any()

    # 如果没有 TDOS，尝试从任一 PDOS 获取 x 网格
    if tdos_mode == 'none':
        candidates = (sorted(glob.glob(PDOS_SPIN_UP_PATTERN)) +
                      sorted(glob.glob(PDOS_SPIN_DW_PATTERN)) +
                      sorted(glob.glob(PDOS_NONSPIN_PATTERN)))
        if not candidates:
            raise FileNotFoundError("未找到 TDOS 和 PDOS 数据文件。")
        x0, _ = load_xy_two_cols(candidates[0], x_col=0, y_col=-1, skiprows=1)
        x_t = x0
    else:
        # 平滑 TDOS
        if tdos_mode == 'nonspin':
            tdos_data['total'] = gaussian_smooth_points(tdos_data['total'], GAUSS_SIGMA_POINTS)
        else:
            tdos_data['up'] = gaussian_smooth_points(tdos_data['up'], GAUSS_SIGMA_POINTS)
            tdos_data['dw'] = gaussian_smooth_points(tdos_data['dw'], GAUSS_SIGMA_POINTS)

    # 2) 收集 PDOS 并插值/平滑
    pdos = collect_pdos(x_t)
    spin_present = (tdos_mode == 'spin') or pdos['spin']

    # 3) 作图
    fig, ax = plt.subplots(figsize=(10, 6), dpi=800)

    # --- 颜色映射（对元素而言保持一致） ---
    elems_order = sorted(pdos['elems'].keys())
    elem_colors = assign_colors(elems_order)

    # --- 先画 TDOS ---
    if tdos_mode == 'nonspin' and 'total' in tdos_data:
        ax.plot(x_t, tdos_data['total'], color='black', lw=LINEWIDTH, label='Total')
    elif tdos_mode == 'spin':
        y_up = tdos_data['up']
        y_dw = tdos_data['dw'] if MIRROR_SPIN else tdos_data['dw']
        # 只在一条总 DOS 线上打图例
        ax.plot(x_t, y_up, color='black', lw=LINEWIDTH, linestyle=UP_LINESTYLE, label='Total')
        ax.plot(x_t, y_dw, color='black', lw=LINEWIDTH, linestyle=DW_LINESTYLE, label='_nolegend_')

    # --- 再画各元素（图例只显示元素名，不重复 ↑/↓） ---
    for elem in elems_order:
        color = elem_colors.get(elem, None)
        data = pdos['elems'][elem]
        if 'total' in data:
            ax.plot(x_t, data['total'], lw=LINEWIDTH, label=elem, color=color)
        else:
            y_up = data['up']
            y_dw = data['dw'] if MIRROR_SPIN else data['dw']
            # 只给一条线（一般 UP）加图例，另一条不进入图例
            ax.plot(x_t, y_up, lw=LINEWIDTH, label=elem, color=color, linestyle=UP_LINESTYLE)
            ax.plot(x_t, y_dw, lw=LINEWIDTH, label='_nolegend_', color=color, linestyle=DW_LINESTYLE)

    # 坐标轴范围
    ax.set_xlim(*XRANGE)

    # y 轴范围自动或固定
    if YMAX_FIXED is not None:
        if spin_present and MIRROR_SPIN:
            ax.set_ylim(-YMAX_FIXED, YMAX_FIXED)
        else:
            ax.set_ylim(0, YMAX_FIXED)
    else:
        # 自动估计：确保在 MIRROR_SPIN=False 且存在负值时也能显示自旋向下
        ys = [l.get_ydata() for l in ax.get_lines()]
        if not ys:
            ax.set_ylim(0.0, 1.0)
        else:
            y_max = max(float(np.nanmax(y)) for y in ys)
            y_min = min(float(np.nanmin(y)) for y in ys)
            if spin_present and MIRROR_SPIN:
                bound = max(abs(y_max), abs(y_min))
                bound = 1.05 * bound if bound > 0 else 1.0
                ax.set_ylim(-bound, bound)
            else:
                lower = 1.05 * y_min if y_min < 0 else 0.0
                upper = 1.05 * y_max if y_max > 0 else 1.0
                ax.set_ylim(lower, upper)

    # 刻度与坐标轴标签
    plt.xticks(fontsize=22, fontweight='bold')
    plt.yticks(fontsize=22, fontweight='bold')
    plt.xlabel(r'${E}$-$E_{f}$ (eV)', fontsize=28, fontweight='bold')
    plt.ylabel('DOS (states / eV)', fontsize=28, fontweight='bold')

    # 竖直/水平灰色虚线：x=0 与 y=0
    ax.axvline(0.0, color='gray', linestyle='--', linewidth=1)
    ax.axhline(0.0, color='gray', linestyle='--', linewidth=1)

    # 图例（更紧凑：减少行距与内边距；仅元素名与 Total）
    handles, labels = ax.get_legend_handles_labels()
    # Matplotlib 会自动忽略 label = '_nolegend_'
    plt.legend(prop={'size': 24, 'weight': 'bold'},
               labelcolor='linecolor',
               frameon=False,
               labelspacing=0.2,
               borderaxespad=0.2,
               handlelength=2.0,
               handletextpad=0.4,
               ncol=2)

    plt.tight_layout()
    plt.savefig('DOS.svg')
    # plt.show()

if __name__ == "__main__":
    main()
