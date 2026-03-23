import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from matplotlib.lines import Line2D
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from PIL import Image
import json
import traceback

# ========== 字体设置：全部使用 Times New Roman ==========
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 10

# ========== 参数配置 ==========
img_dir  = "output"
save_dir = "result"
os.makedirs(save_dir, exist_ok=True)

output_file = os.path.join(save_dir, "fitting_results.json")
error_log   = os.path.join(save_dir, "error_log.txt")


# ========== 核心拟合函数 ==========
def sinfun_k(p, x):
    """正弦函数: A*sin(k*x + phi) + y0"""
    A, k, phi, y0 = p
    return A * np.sin(k * x + phi) + y0


# ========== 基于 FFT 的初始参数估计 ==========
def fft_init_params(x_region, y_region):
    y0_est    = np.median(y_region)
    y_centered = y_region - y0_est
    A_est     = np.std(y_centered) * np.sqrt(2)

    N     = 512
    x_uni = np.linspace(x_region.min(), x_region.max(), N)
    y_uni = np.interp(x_uni, x_region, y_centered)

    fft_vals = np.fft.rfft(y_uni)
    freqs    = np.fft.rfftfreq(N, d=(x_uni[1] - x_uni[0]))

    mag = np.abs(fft_vals)
    mag[0] = 0
    peak_idx = np.argmax(mag)
    freq_est = freqs[peak_idx]

    k_est   = 2 * np.pi * freq_est if freq_est > 0 else (2 * np.pi / (x_region.ptp() + 1e-6))
    phi_est = np.angle(fft_vals[peak_idx])

    return [A_est, k_est, phi_est, y0_est]


# ========== 带离群点剔除的最小二乘拟合（返回 iter_records）==========
def fit_region_robust(x_region, y_region, init_params, max_iter=5):
    x_fit  = x_region.copy().astype(float)
    y_fit  = y_region.copy().astype(float)
    params = list(init_params)

    A_init = abs(params[0]) if params[0] != 0 else 10.0
    k_init = abs(params[1]) if params[1] != 0 else 0.05

    bounds_lo = [0,      1e-4,  -np.pi * 4, -np.inf]
    bounds_hi = [np.inf, np.pi,  np.pi * 4,  np.inf]

    params[0] = A_init
    params[1] = k_init

    r_thresh    = None
    iter_records = []          # ← 新增：收集每轮状态

    for iteration in range(max_iter):
        def residuals(p):
            return y_fit - sinfun_k(p, x_fit)

        try:
            res = least_squares(
                residuals, params,
                bounds=(bounds_lo, bounds_hi),
                max_nfev=2000,
                ftol=1e-5, xtol=1e-5, gtol=1e-5,
                method='trf'
            )
            params = list(res.x)
        except Exception as e:
            print(f"    [警告] 拟合迭代 {iteration} 出错: {e}")
            break

        y_pred   = sinfun_k(params, x_fit)
        res_abs  = np.abs(y_fit - y_pred)
        cur_rmse = float(np.sqrt(np.mean(res_abs ** 2)))

        if r_thresh is None:
            r_thresh = np.percentile(res_abs, 85)
        else:
            r_thresh = np.percentile(res_abs, 75)

        idx_keep = res_abs < max(r_thresh, 3.0)
        idx_drop = ~idx_keep

        # ── 记录本轮状态 ──
        iter_records.append({
            "iter":   iteration,
            "x_keep": x_fit[idx_keep].copy(),
            "y_keep": y_fit[idx_keep].copy(),
            "x_drop": x_fit[idx_drop].copy(),
            "y_drop": y_fit[idx_drop].copy(),
            "params": list(params),
            "rmse":   cur_rmse,
        })

        if np.sum(idx_keep) < max(5, len(x_fit) * 0.3):
            break

        x_fit = x_fit[idx_keep]
        y_fit = y_fit[idx_keep]

    return params, iter_records          # ← 同时返回记录


# ========== 区域分割 ==========
def find_curve_regions(mat_bin, min_gap=20, min_height=10):
    row_count  = np.sum(mat_bin, axis=1)
    has_signal = row_count > 0

    regions   = []
    in_region = False
    row_start = 0
    gap_count = 0

    for j in range(len(has_signal)):
        if has_signal[j]:
            if not in_region:
                row_start = j
                in_region = True
            gap_count = 0
        else:
            if in_region:
                gap_count += 1
                if gap_count >= min_gap:
                    row_end = j - gap_count
                    if row_end - row_start >= min_height:
                        regions.append((row_start, row_end))
                    in_region = False
                    gap_count = 0

    if in_region:
        row_end = len(has_signal) - 1
        if row_end - row_start >= min_height:
            regions.append((row_start, row_end))

    return regions


# ========== 列中位数细化 ==========
def thin_by_column_median(x_region, y_region, min_pts_per_col=1):
    xs = np.unique(x_region)
    x_thin, y_thin = [], []
    for xi in xs:
        mask = x_region == xi
        if np.sum(mask) >= min_pts_per_col:
            x_thin.append(xi)
            y_thin.append(np.median(y_region[mask]))
    return np.array(x_thin, dtype=float), np.array(y_thin, dtype=float)


# ========== 多初值策略（返回 multi_init_records）==========
def fit_with_multi_init(x_region, y_region):
    init_fft = fft_init_params(x_region, y_region)

    y0_scan   = np.median(y_region)
    A_scan    = (y_region.max() - y_region.min()) / 2.0
    x_span    = x_region.ptp()
    init_scan = [A_scan, 2 * np.pi / max(x_span / 2, 1), 0.0, y0_scan]

    init_small = [A_scan * 0.5, init_fft[1], init_fft[2], y0_scan]

    candidates  = [init_fft,    init_scan,    init_small]
    cand_labels = ["FFT init",  "scan init",  "small init"]

    best_params = None
    best_rmse   = float('inf')
    multi_init_records = []          # ← 新增：收集候选结果

    for label, init in zip(cand_labels, candidates):
        try:
            params, _ = fit_region_robust(x_region, y_region, init)
            y_pred    = sinfun_k(params, x_region)
            rmse      = float(np.sqrt(np.mean((y_region - y_pred) ** 2)))

            multi_init_records.append({
                "label":  label,
                "params": list(params),
                "rmse":   rmse,
                "best":   False,
            })

            if rmse < best_rmse:
                best_rmse   = rmse
                best_params = params
        except Exception:
            continue

    # 标记最优候选
    if multi_init_records:
        best_idx = int(np.argmin([r["rmse"] for r in multi_init_records]))
        multi_init_records[best_idx]["best"] = True

    return best_params, best_rmse, multi_init_records   # ← 多返回一个


# ========== 2×2 诊断图 ==========
def plot_diagnostic_2x2(
    x_region,
    y_region,
    iter_records,
    multi_init_records,
    save_path=None,
    region_id=0,
    img_name="",
):
    """绘制并保存 2×2 过程性诊断图。"""

    AXES_BG = (209/255, 217/255, 238/255)
    KEEP_C  = '#2E7D6E'
    DROP_C  = '#C0392B'
    CURVE_C = ['#1a5276', '#117a65', '#6e2f8a']
    BEST_C  = '#1a5276'
    OTHER_C = ['#E67E22', '#C0392B', '#8E44AD']
    SCAT_C  = '#7F8C8D'

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.patch.set_facecolor('white')

    x_dense   = np.linspace(x_region.min(), x_region.max(), 500)
    iter_show = iter_records[:3]

    # ── 子图 ①②③：逐轮离群点剔除 ────────────────────────────────
    for idx, rec in enumerate(iter_show):
        row, col = divmod(idx, 2)
        ax = axes[row][col]
        ax.set_facecolor(AXES_BG)

        if len(rec['x_drop']) > 0:
            ax.scatter(rec['x_drop'], rec['y_drop'],
                       s=10, c=DROP_C, alpha=0.55, linewidths=0)

        ax.scatter(rec['x_keep'], rec['y_keep'],
                   s=10, c=KEEP_C, alpha=0.75, linewidths=0)

        ax.plot(x_dense, sinfun_k(rec['params'], x_dense),
                color=CURVE_C[idx % len(CURVE_C)], linewidth=2.0, zorder=5)

        ax.set_title(
            f"Iter {rec['iter']}: RMSE = {rec['rmse']:.2f},  "
            f"kept = {len(rec['x_keep'])} pts",
            pad=6
        )
        ax.set_xlabel("X  (pixels)")
        ax.set_ylabel("Y  (pixels)")

        ax.legend(handles=[
            mpatches.Patch(color=KEEP_C, alpha=0.75,
                           label=f"kept  ({len(rec['x_keep'])} pts)"),
            mpatches.Patch(color=DROP_C, alpha=0.55,
                           label=f"dropped  ({len(rec['x_drop'])} pts)"),
            Line2D([0], [0], color=CURVE_C[idx % len(CURVE_C)],
                   linewidth=2, label='fit curve'),
        ], loc='upper right', framealpha=0.85, edgecolor='#888888')

        for spine in ax.spines.values():
            spine.set_edgecolor('#888888')

    # ── 子图 ④：多初值候选对比 ──────────────────────────────────
    ax4 = axes[1][1]
    ax4.set_facecolor(AXES_BG)

    ax4.scatter(x_region, y_region,
                s=8, c=SCAT_C, alpha=0.35, linewidths=0, zorder=2)

    legend_handles4 = [
        mpatches.Patch(color=SCAT_C, alpha=0.4, label='skeleton pts'),
    ]
    other_idx = 0
    for rec in multi_init_records:
        y_c = sinfun_k(rec['params'], x_dense)
        if rec['best']:
            label_str = f"{rec['label']}  RMSE={rec['rmse']:.2f}  $\\checkmark$"
            ax4.plot(x_dense, y_c,
                     color=BEST_C, linewidth=2.5, linestyle='-', zorder=6)
            legend_handles4.append(
                Line2D([0], [0], color=BEST_C, linewidth=2.5, label=label_str)
            )
        else:
            c = OTHER_C[other_idx % len(OTHER_C)]
            label_str = f"{rec['label']}  RMSE={rec['rmse']:.2f}"
            ax4.plot(x_dense, y_c,
                     color=c, linewidth=1.5, linestyle='--', alpha=0.75, zorder=4)
            legend_handles4.append(
                Line2D([0], [0], color=c, linewidth=1.5,
                       linestyle='--', alpha=0.75, label=label_str)
            )
            other_idx += 1

    ax4.set_title("Multi-init candidates comparison", pad=6)
    ax4.set_xlabel("X  (pixels)")
    ax4.set_ylabel("Y  (pixels)")
    ax4.legend(handles=legend_handles4, loc='upper right',
               framealpha=0.85, edgecolor='#888888')
    for spine in ax4.spines.values():
        spine.set_edgecolor('#888888')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"    ✓ 诊断图已保存: {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ========== 主处理流程 ==========
all_results = {}

img_list = sorted([f for f in os.listdir(img_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

if not img_list:
    print(f"⚠️  在 '{img_dir}' 目录中未找到图片文件！")
else:
    print(f"共找到 {len(img_list)} 张图片，开始处理...\n")

for img_name in img_list:
    try:
        print(f"\n{'='*60}")
        print(f"处理图片: {img_name}")
        print(f"{'='*60}")

        img_path = os.path.join(img_dir, img_name)
        img  = Image.open(img_path).convert("L")
        W, H = img.size
        mat  = np.array(img)

        mat_bin = (mat > 128).astype(np.uint8)

        white_count = np.sum(mat_bin)
        print(f"  图像尺寸: {W}×{H}，检测到白点数量: {white_count}")

        if white_count == 0:
            print(f"  ⚠️  未检测到白色曲线，跳过。")
            all_results[img_name] = []
            continue

        # ---- 可视化准备 ----
        plt.close('all')
        fig, ax = plt.subplots(figsize=(12, 7))
        axes_bg = (209/255, 217/255, 238/255)
        ax.set_facecolor(axes_bg)
        fig.patch.set_facecolor('white')

        y_idx_all, x_idx_all = np.where(mat_bin == 1)
        y_idx_all_flipped = H - 1 - y_idx_all
        ax.scatter(x_idx_all, y_idx_all_flipped, s=3, c='black', alpha=0.8, linewidths=0)

        regions = find_curve_regions(mat_bin, min_gap=20, min_height=10)
        print(f"  检测到 {len(regions)} 个曲线区域")

        img_results = []
        palette = ['#3A6B4A', '#2E5F8A', '#A03050', '#8A7040']

        for region_idx, (row_start, row_end) in enumerate(regions):
            try:
                sub = mat_bin[row_start:row_end + 1, :]
                y_rel, x_idx = np.where(sub == 1)
                x_raw = x_idx.astype(float)

                y_image = (y_rel + row_start).astype(float)
                y_raw   = H - 1 - y_image

                n_pts = len(x_raw)
                print(f"\n  → 区域 {region_idx}  行[{row_start}~{row_end}]  点数: {n_pts}")

                if n_pts < 8:
                    print(f"    ⚠️  点数不足，跳过")
                    continue

                x_region, y_region = thin_by_column_median(x_raw, y_raw)
                if len(x_region) < 5:
                    print(f"    ⚠️  细化后点数不足，跳过")
                    continue

                sort_idx = np.argsort(x_region)
                x_region = x_region[sort_idx]
                y_region = y_region[sort_idx]

                # ---- 多初值拟合（现在多返回 multi_init_records）----
                params, rmse, multi_init_records = fit_with_multi_init(x_region, y_region)

                if params is None:
                    print(f"    ❌ 拟合失败")
                    continue

                A_fit, k_fit, phi_fit, y0_fit = params
                P_fit = 2 * np.pi / abs(k_fit) if k_fit != 0 else np.inf

                rmse_limit = max(25.0, (row_end - row_start) * 0.8)
                if rmse > rmse_limit:
                    print(f"    ⚠️  RMSE={rmse:.2f} 超过限制 {rmse_limit:.2f}，跳过")
                    continue

                # ---- 主图：绘制拟合曲线 ----
                color   = palette[region_idx % len(palette)]
                x_dense = np.linspace(x_region.min(), x_region.max(), 500)
                y_dense = sinfun_k(params, x_dense)
                ax.plot(x_dense, y_dense, linewidth=2.5, color=color)

                # ---- 诊断图：取最优候选的 iter_records ----
                best_label = next(
                    (r["label"] for r in multi_init_records if r["best"]), "FFT init"
                )
                cand_label_map = {
                    "FFT init":   fft_init_params(x_region, y_region),
                    "scan init":  [
                        (y_region.max()-y_region.min())/2.0,
                        2*np.pi / max(x_region.ptp()/2, 1),
                        0.0,
                        np.median(y_region)
                    ],
                    "small init": None,   # 占位，下面直接取 best params
                }
                best_init = cand_label_map.get(best_label)
                if best_init is None:
                    best_init = params
                _, iter_records = fit_region_robust(x_region, y_region, best_init)

                diag_path = os.path.join(
                    save_dir,
                    f"{os.path.splitext(img_name)[0]}_region{region_idx}_diag.png"
                )
                plot_diagnostic_2x2(
                    x_region           = x_region,
                    y_region           = y_region,
                    iter_records       = iter_records[-3:],   # 取最后 3 轮
                    multi_init_records = multi_init_records,
                    save_path          = diag_path,
                    region_id          = region_idx,
                    img_name           = img_name,
                )

                # ---- 保存 JSON 结果 ----
                region_result = {
                    "region_id":   region_idx,
                    "row_start":   int(row_start),
                    "row_end":     int(row_end),
                    "point_count": int(n_pts),
                    "parameters": {
                        "A":   float(A_fit),
                        "k":   float(k_fit),
                        "phi": float(phi_fit),
                        "y0":  float(y0_fit),
                        "P":   float(P_fit),
                    },
                    "rmse": float(rmse)
                }
                img_results.append(region_result)
                print(f"    ✓ 拟合成功  A={A_fit:.2f}, k={k_fit:.4f}, "
                      f"phi={phi_fit:.3f}, y0={y0_fit:.2f}  RMSE={rmse:.2f}")

            except Exception as e:
                print(f"    ❌ 区域 {region_idx} 异常: {e}")
                traceback.print_exc()
                continue

        # ---- 主图坐标轴样式 ----
        ax.set_xlabel("X  (pixels)", color='black', fontname='Times New Roman')
        ax.set_ylabel("Y  (pixels)", color='black', fontname='Times New Roman')
        ax.tick_params(colors='black')
        for spine in ax.spines.values():
            spine.set_edgecolor('#888888')
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()
        save_img_path = os.path.join(
            save_dir, f"{os.path.splitext(img_name)[0]}_fit.png")
        plt.savefig(save_img_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        print(f"\n  ✓ 可视化已保存: {save_img_path}")
        all_results[img_name] = img_results

        if img_results:
            print(f"  ✓ {img_name} 完成，共识别 {len(img_results)} 条曲线")
        else:
            print(f"  ⚠️  {img_name} 未识别到有效曲线，请检查参数或图像质量")

    except Exception as e:
        print(f"\n❌ {img_name} 处理失败: {e}")
        traceback.print_exc()
        with open(error_log, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n图片: {img_name}\n错误: {e}\n")
            f.write(traceback.format_exc())
        plt.close('all')

# ========== 保存 JSON 结果 ==========
plt.close('all')
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print(f"\n{'='*60}")
print(f"✓ JSON 结果已保存: {output_file}")
print(f"✓ 拟合图已保存至: {save_dir}/")
if os.path.exists(error_log) and os.path.getsize(error_log) > 0:
    print(f"⚠  错误日志: {error_log}")
print(f"{'='*60}\n")