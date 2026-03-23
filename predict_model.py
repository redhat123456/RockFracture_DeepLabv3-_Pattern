"""
predict_with_denoising.py  (重构版)
=====================================
岩缝分割预测  +  空间聚类降噪  +  完整可视化
 
修改说明：
  - sample_N.png：第3列（Prediction raw）和第4列（Denoised）
    改为「白=裂缝，黑=背景」，视觉更突出
  - 标题字体全部使用 Times New Roman，字号 18pt
  - 新增 noise_removal_rate.png：所有样本的噪点去除率随样本序号变化折线图
  - 其余可视化图（cluster_centers、param_sweep、all_summary）暂时不输出
"""
 
import os
import time
import numpy as np
from scipy import ndimage
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import cv2
 
# ── 全局字体设置 ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':        'Times New Roman',
    'font.size':          20,
    'axes.titlesize':     20,
    'axes.labelsize':     20,
    'xtick.labelsize':    20,
    'ytick.labelsize':    20,
    'legend.fontsize':    20,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
})
 

# ============================================================
# 1. 图像预处理
# ============================================================

def preprocess_image(img_path):
    """CLAHE + 双边滤波 + 高通增强"""
    try:
        data = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        bilateral = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        highpass = cv2.filter2D(bilateral, -1, kernel)
        return cv2.addWeighted(bilateral, 1.0, highpass, 0.7, 0)
    except Exception as e:
        print(f"  [preprocess] 读取出错: {e}")
        return None


# ============================================================
# 2. 模型定义（DeepLabV3+ with EdgeAware CBAM）
# ============================================================

def sobel_edge(x):
    if x.shape[1] == 3:
        x_gray = 0.299*x[:,0:1,:,:] + 0.587*x[:,1:2,:,:] + 0.114*x[:,2:3,:,:]
    else:
        x_gray = x
    kernel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32,
                             device=x.device).unsqueeze(0).unsqueeze(0)
    kernel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32,
                             device=x.device).unsqueeze(0).unsqueeze(0)
    edge_x = F.conv2d(x_gray, kernel_x, padding=1)
    edge_y = F.conv2d(x_gray, kernel_y, padding=1)
    return torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(x_cat))


class EdgeAwareCBAM(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, ratio)
        self.spatial_att = SpatialAttention()

    def forward(self, x, img_input=None):
        x_out = self.channel_att(x)
        if img_input is not None:
            edge_map = sobel_edge(img_input)
            edge_map = F.interpolate(edge_map, size=x_out.shape[2:],
                                     mode='bilinear', align_corners=False)
            sa_out = self.spatial_att(x_out) * (1 + edge_map)
        else:
            sa_out = self.spatial_att(x_out)
        return sa_out


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv3_6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv3_12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.conv3_18 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv3_6(x)
        x3 = self.conv3_12(x)
        x4 = self.conv3_18(x)
        return self.project(torch.cat([x1, x2, x3, x4], dim=1))


class DeepLabV3Plus_Edge(nn.Module):
    def __init__(self, n_classes=2):
        super().__init__()
        # backbone 拆散存储，键名为 layer0.* layer1.* ... （与 checkpoint 完全一致）
        self.backbone = models.resnet50(
            pretrained=False,
            replace_stride_with_dilation=[False, True, True])
        layers = list(self.backbone.children())
        self.layer0 = nn.Sequential(*layers[:4])   # conv1+bn1+relu+maxpool
        self.layer1 = layers[4]
        self.layer2 = layers[5]
        self.layer3 = layers[6]
        self.layer4 = layers[7]

        self.aspp = ASPP(2048, 256)

        self.low_level_conv = nn.Sequential(
            nn.Conv2d(256, 256, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.cbam = EdgeAwareCBAM(512)

        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, n_classes, 1))

    def forward(self, x):
        original_x = x
        h, w = x.shape[2:]
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        aspp_out = self.aspp(x4)
        aspp_out = F.interpolate(aspp_out, size=x1.shape[2:],
                                 mode='bilinear', align_corners=False)
        low = self.low_level_conv(x1)
        x = torch.cat([aspp_out, low], dim=1)
        x = self.cbam(x, img_input=original_x)
        x = self.decoder(x)
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)


# ============================================================
# 3. 空间聚类降噪
# ============================================================
 
def spatial_cluster_denoise(pred,
                             merge_dist=3,
                             min_cluster_pixels=150,
                             use_boundary_dist=True):
    labeled, n = ndimage.label(pred)
    if n == 0:
        return pred.copy().astype(np.uint8), {
            'original_pixels': 0, 'removed_pixels': 0, 'kept_pixels': 0,
            'n_components': 0, 'n_clusters': 0, 'clusters': [],
            'components': [], 'labeled': labeled,
            'merge_dist': merge_dist, 'min_cluster_pixels': min_cluster_pixels,
        }
 
    components = []
    for lid in range(1, n + 1):
        comp   = (labeled == lid)
        coords = np.where(comp)
        sz     = int(comp.sum())
        ymin, ymax = int(coords[0].min()), int(coords[0].max())
        xmin, xmax = int(coords[1].min()), int(coords[1].max())
        components.append({
            'lid': lid, 'sz': sz,
            'cy': (ymin+ymax)//2, 'cx': (xmin+xmax)//2,
            'ymin': ymin, 'ymax': ymax, 'xmin': xmin, 'xmax': xmax,
        })
 
    N = len(components)
    _masks = {}
    def get_mask(i):
        if i not in _masks:
            _masks[i] = (labeled == components[i]['lid'])
        return _masks[i]
 
    parent = list(range(N))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[rb] = ra
 
    for i in range(N):
        for j in range(i + 1, N):
            ci, cj = components[i], components[j]
            row_gap = max(0, max(ci['ymin'], cj['ymin']) - min(ci['ymax'], cj['ymax']))
            col_gap = max(0, max(ci['xmin'], cj['xmin']) - min(ci['xmax'], cj['xmax']))
            bbox_d  = (row_gap**2 + col_gap**2) ** 0.5
            if bbox_d > merge_dist + 1:
                continue
            if use_boundary_dist:
                dt = ndimage.distance_transform_edt(~get_mask(i))
                d  = float(dt[get_mask(j)].min())
            else:
                d = bbox_d
            if d <= merge_dist:
                union(i, j)
 
    cluster_px  = {}
    cluster_mem = {}
    for i, comp in enumerate(components):
        r = find(i)
        cluster_px[r]  = cluster_px.get(r, 0) + comp['sz']
        cluster_mem[r] = cluster_mem.get(r, []) + [i]
 
    out = np.zeros_like(pred, dtype=np.uint8)
    removed = kept = 0
    clusters_info = []
 
    for r, total_px in cluster_px.items():
        keep    = (total_px >= min_cluster_pixels)
        members = cluster_mem[r]
        clusters_info.append({
            'n_components': len(members),
            'total_pixels': total_px,
            'keep':         keep,
            'cy': float(np.mean([components[m]['cy'] for m in members])),
            'cx': float(np.mean([components[m]['cx'] for m in members])),
            'members': members,
        })
        for m in members:
            mask_m = get_mask(m)
            if keep:
                out[mask_m] = 1;  kept    += components[m]['sz']
            else:
                removed += components[m]['sz']
 
    return out, {
        'original_pixels':    int(pred.sum()),
        'removed_pixels':     removed,
        'kept_pixels':        kept,
        'n_components':       n,
        'n_clusters':         len(cluster_px),
        'clusters':           sorted(clusters_info, key=lambda x: -x['total_pixels']),
        'components':         components,
        'labeled':            labeled,
        'merge_dist':         merge_dist,
        'min_cluster_pixels': min_cluster_pixels,
    }
 
 
# ============================================================
# 4. 工具函数
# ============================================================
 
def _title_font():
    return {'fontsize': 18, 'fontfamily': 'Times New Roman', 'fontweight': 'bold'}
 
def _sub_font():
    return {'fontsize': 18, 'fontfamily': 'Times New Roman'}
 
 
def _show_black_crack(ax, mask, title='', subtitle=''):
    """黑=裂缝，白=背景（GT 列专用）"""
    disp = ((1 - mask.astype(float)) * 255).astype(np.uint8)
    ax.imshow(disp, cmap='gray', vmin=0, vmax=255,
              aspect='auto', interpolation='nearest')
    if title:
        ax.set_title(title, pad=8, **_title_font())
    if subtitle:
        ax.text(0.5, -0.02, subtitle, transform=ax.transAxes,
                ha='center', va='top', color='#444', **_sub_font())
    ax.axis('off')
 
 
def _show_white_crack(ax, mask, title='', subtitle=''):
    """白=裂缝，黑=背景（Prediction / Denoised 列专用）"""
    disp = (mask.astype(float) * 255).astype(np.uint8)
    ax.imshow(disp, cmap='gray', vmin=0, vmax=255,
              aspect='auto', interpolation='nearest')
    if title:
        ax.set_title(title, pad=8, **_title_font())
    if subtitle:
        ax.text(0.5, -0.02, subtitle, transform=ax.transAxes,
                ha='center', va='top', color='#ccc', **_sub_font())
    ax.axis('off')
 
 
def evaluate(pred, truth):
    p = pred.astype(bool); t = truth.astype(bool)
    tp=(p&t).sum(); fp=(p&~t).sum(); fn=(~p&t).sum(); tn=(~p&~t).sum()
    iou  = tp/(tp+fp+fn+1e-9);  prec = tp/(tp+fp+1e-9)
    rec  = tp/(tp+fn+1e-9);     f1   = 2*prec*rec/(prec+rec+1e-9)
    return dict(IoU=float(iou), Precision=float(prec),
                Recall=float(rec), F1=float(f1),
                TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn))
 
 
# ============================================================
# 5. 图①：四列对比图
#    列1: 原图（彩色）
#    列2: GT          黑=裂缝 白=背景
#    列3: Prediction  白=裂缝 黑=背景  ← 新规则
#    列4: Denoised    白=裂缝 黑=背景  ← 新规则
# ============================================================
 
def plot_four_panel(img_orig, gt_mask, pred, denoised, stats,
                    sample_name, save_path):
    m_raw = evaluate(pred,     gt_mask)
    m_dn  = evaluate(denoised, gt_mask)
    pct   = 100 * stats['removed_pixels'] / max(stats['original_pixels'], 1)
 
    H, W_img = img_orig.shape[:2]
    dpi  = 100
    col_w = max(W_img, 80)
 
    fig, axes = plt.subplots(1, 4,
                             figsize=(col_w * 4 / dpi, H / dpi),
                             dpi=dpi)
 
    fig.suptitle(
        f'merge_dist = {stats["merge_dist"]} px     '
        f'min_cluster_pixels = {stats["min_cluster_pixels"]}     '
        f'{stats["n_components"]} components  →  {stats["n_clusters"]} clusters',
        y=1.02, **_title_font())
 
    # 列1：原图
    axes[0].imshow(img_orig, aspect='auto', interpolation='lanczos')
    axes[0].set_title('Original Image', pad=8, **_title_font())
    axes[0].axis('off')
 
    # 列2：GT  黑=裂缝
    _show_black_crack(
        axes[1], gt_mask,
        title='Ground Truth\n(black = crack)')
 
    # 列3：Prediction  白=裂缝
    _show_white_crack(
        axes[2], pred,
        title='Prediction (raw)\n(white = crack)')
 
    # 列4：Denoised  白=裂缝
    _show_white_crack(
        axes[3], denoised,
        title=f'Denoised  (removed {pct:.2f}%)\n(white = crack)')
 
    plt.subplots_adjust(wspace=0.03, left=0.01, right=0.99,
                        top=0.93, bottom=0.03)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  [saved] {save_path}')
 
 
# ============================================================
# 6. 图②：噪点去除率随样本序号变化图（3个子图）
# ============================================================
 
def plot_noise_removal_rate(removal_records, save_path):
    """
    三个子图：
      ① 去除率折线（%）
      ② 像素数分解（总量 / 保留 / 去除）
      ③ 连通域数量柱状图
    """
    if not removal_records:
        return
 
    idxs       = [r['idx']            for r in removal_records]
    rates      = [r['removal_rate']   for r in removal_records]
    orig_px    = [r['original_pixels'] / 1000 for r in removal_records]
    removed_px = [r['removed_pixels']  / 1000 for r in removal_records]
    kept_px    = [r['kept_pixels']     / 1000 for r in removal_records]
    n_comps    = [r['n_components']    for r in removal_records]
    names      = [os.path.splitext(r['name'])[0] for r in removal_records]
 
    x       = np.array(idxs)
    xlabels = [f'#{i}\n{n}' for i, n in zip(idxs, names)]
    
    fig, axes = plt.subplots(3, 1,
                             figsize=(max(10, len(idxs) * 1.4), 10),
                             facecolor='white')
 
    # ── 子图①：去除率折线 ────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(x, rates, 'o-',
            color='#C0392B', lw=2.5, ms=9,
            markerfacecolor='white', markeredgewidth=2.2,
            zorder=4)
    ax.fill_between(x, rates, alpha=0.10, color='#E74C3C')
    for xi, ri in zip(x, rates):
        ax.annotate(f'{ri:.2f}%',
                    xy=(xi, ri),
                    xytext=(0, 12), textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=18, fontfamily='Times New Roman',
                    color='#922B21')
    ax.set_ylabel('Removal rate (%)', **_title_font())
    ax.set_title('Noise removal rate per sample', pad=6, **_title_font())
    ax.set_xticks(x); ax.set_xticklabels(xlabels,
                                          fontsize=18, fontfamily='Times New Roman')
    ax.set_ylim(0, max(max(rates) * 1.40, 1))
    ax.yaxis.grid(True, alpha=0.28, linewidth=0.8)
    ax.set_axisbelow(True)
 
    # ── 子图②：像素数分解 ────────────────────────────────────────────────
    ax = axes[1]
    ax.fill_between(x, orig_px,    color='#AEB6BF', alpha=0.35)
    ax.fill_between(x, kept_px,    color='#1E8449', alpha=0.50)
    ax.fill_between(x, removed_px, color='#C0392B', alpha=0.50)
    ax.plot(x, orig_px,    'o--', color='#717D7E', lw=1.8, ms=6,
            label='Total predicted')
    ax.plot(x, kept_px,    's-',  color='#1A5632', lw=2.2, ms=7,
            label='Kept (crack)')
    ax.plot(x, removed_px, 'D-',  color='#922B21', lw=2.2, ms=7,
            label='Removed (noise)')
    ax.set_ylabel('Pixels (× 1000)', **_title_font())
    ax.set_title('Pixel count breakdown per sample', pad=6, **_title_font())
    ax.set_xticks(x); ax.set_xticklabels(xlabels,
                                          fontsize=18, fontfamily='Times New Roman')
    ax.legend(fontsize=18, frameon=False, ncol=3,
              prop={'family': 'Times New Roman', 'size': 14})
    ax.yaxis.grid(True, alpha=0.28, linewidth=0.8)
    ax.set_axisbelow(True)
 
    # ── 子图③：连通域数量 ────────────────────────────────────────────────
    ax = axes[2]
    bars = ax.bar(x, n_comps, width=0.5,
                  color='#2471A3', alpha=0.78, linewidth=0)
    for bar, nc in zip(bars, n_comps):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(n_comps) * 0.015,
                str(nc),
                ha='center', va='bottom',
                fontsize=18, fontfamily='Times New Roman',
                color='#1A5276')
    ax.set_ylabel('# components', **_title_font())
    ax.set_title('Number of connected components per sample', pad=6, **_title_font())
    ax.set_xticks(x); ax.set_xticklabels(xlabels,
                                          fontsize=18, fontfamily='Times New Roman')
    ax.yaxis.grid(True, alpha=0.28, linewidth=0.8)
    ax.set_axisbelow(True)
 
    plt.tight_layout(pad=1.8)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  [saved] {save_path}')
 
 
# ============================================================
# 7. 主流程
# ============================================================
 
def main():
    IMAGE_DIR = 'dataset/origin/'
    MASK_DIR  = 'dataset/mask/'
    CKPT_PATH = 'training_outputs/checkpoints/best_model.pth'
    OUT_VIS   = 'prediction_results/'
    OUT_MASK  = 'output/'
 
    CLUSTER_PARAMS = dict(
        merge_dist          = 10,
        min_cluster_pixels  = 150,
        use_boundary_dist   = True,
    )
 
    os.makedirs(OUT_VIS, exist_ok=True)
    os.makedirs(OUT_MASK, exist_ok=True)
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
 
    model = DeepLabV3Plus_Edge(n_classes=2).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()
    print('模型加载成功')
 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
 
    image_list = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ])
 
    removal_records = []
 
    with torch.no_grad():
        for idx, name in enumerate(image_list):
            img_path = os.path.join(IMAGE_DIR, name)
            print(f'\n[{idx+1}/{len(image_list)}] {name}')
 
            pre = preprocess_image(img_path)
            if pre is None:
                print('  预处理失败，跳过'); continue
 
            pre_rgb  = cv2.cvtColor(pre, cv2.COLOR_GRAY2RGB)
            img_orig = np.array(Image.open(img_path).convert('RGB'))
            orig_h, orig_w = img_orig.shape[:2]
 
            stem      = os.path.splitext(name)[0]
            mask_path = os.path.join(MASK_DIR, f'{stem}_mask.png')
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f'Mask 未找到: {mask_path}')
            gt_gray = np.array(Image.open(mask_path).convert('L'))
            gt_mask = (gt_gray < 128).astype(np.uint8)
 
            inp = transform(
                Image.fromarray(pre_rgb).resize((256, 256), Image.BILINEAR)
            ).unsqueeze(0).to(device)
 
            # ── 调试：第一张图打印概率统计 ──────────────────────────────
            if idx == 0:
                probs_dbg = torch.softmax(model(inp), dim=1)
                cp = probs_dbg[0, 1].cpu().numpy()
                print(f'  [调试] 裂缝概率: '
                      f'min={cp.min():.3f}  max={cp.max():.3f}  mean={cp.mean():.3f}')
                print(f'  → 若 mean > 0.8：模型严重过预测，需调整阈值或重训练')
                print(f'  → 若 mean < 0.2：标签方向反了，pred = 1-argmax')
 
            # 标签取反（根据实际情况选择是否保留此行）
            pred_small = 1 - torch.argmax(model(inp), dim=1).cpu().numpy()[0]
            pred = np.array(
                Image.fromarray(pred_small.astype(np.uint8))
                     .resize((orig_w, orig_h), Image.NEAREST)
            )
 
            t0 = time.perf_counter()
            pred_dn, stats = spatial_cluster_denoise(pred, **CLUSTER_PARAMS)
            elapsed = (time.perf_counter() - t0) * 1000
 
            pct = 100 * stats['removed_pixels'] / max(stats['original_pixels'], 1)
            print(f'  预测裂缝: {stats["original_pixels"]:,}px')
            print(f'  去除噪点: {stats["removed_pixels"]:,}px ({pct:.2f}%)'
                  f'  [{stats["n_components"]} comps → {stats["n_clusters"]} clusters]')
            print(f'  保留裂缝: {stats["kept_pixels"]:,}px   耗时: {elapsed:.0f}ms')
 
            m_raw = evaluate(pred,    gt_mask)
            m_dn  = evaluate(pred_dn, gt_mask)
            print(f'  Raw  IoU={m_raw["IoU"]:.4f}  F1={m_raw["F1"]:.4f}')
            print(f'  Dn   IoU={m_dn["IoU"]:.4f}   F1={m_dn["F1"]:.4f}')
 
            removal_records.append({
                'idx':             idx + 1,
                'name':            name,
                'original_pixels': stats['original_pixels'],
                'removed_pixels':  stats['removed_pixels'],
                'kept_pixels':     stats['kept_pixels'],
                'n_components':    stats['n_components'],
                'n_clusters':      stats['n_clusters'],
                'removal_rate':    pct,
            })
 
            # 图①：四列对比
            plot_four_panel(
                img_orig, gt_mask, pred, pred_dn, stats,
                sample_name=f'sample_{idx+1}',
                save_path=os.path.join(OUT_VIS, f'sample_{idx+1}.png'))
 
            # 保存降噪 mask
            Image.fromarray((pred_dn * 255).astype(np.uint8)).save(
                os.path.join(OUT_MASK, f'denoised_prediction_{idx+1}.png'))
            print(f'  mask 已保存: output/denoised_prediction_{idx+1}.png')
 
    # 图②：噪点去除率折线图
    if removal_records:
        print('\n生成噪点去除率折线图...')
        plot_noise_removal_rate(
            removal_records,
            save_path=os.path.join(OUT_VIS, 'noise_removal_rate.png'))
 
    print(f'\n{"="*55}')
    print('完成！')
    print(f'  可视化输出: {OUT_VIS}')
    print(f'  降噪 Mask : {OUT_MASK}')
    print(f'{"="*55}')
 
 
# ============================================================
# 8. 入口
# ============================================================
 
if __name__ == '__main__':
    main()