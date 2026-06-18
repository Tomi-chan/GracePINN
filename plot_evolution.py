import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['figure.max_open_warning'] = 0

from scipy.interpolate import griddata
from scipy.spatial import cKDTree
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # Vũ khí bí mật để khóa chiều cao colorbar

# ==========================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN 
# ==========================================================
BASE_DIR = os.path.abspath("runs")
METHOD_NAME = "06.18-10.17.32-grace_0"
PDE_IDX = 6
# Tên PDE chuẩn theo mặc định của framework PINNacle
PDE_NAME = 'NS2D_LongTime'

LOG_FOLDER = os.path.join(BASE_DIR, METHOD_NAME, f"{PDE_IDX}-0", "gracepinn_logs")
OUTPUT_DIR = os.path.join(LOG_FOLDER, "visualizations")

# ==========================================================
# 2. CẤU HÌNH ĐỒ THỊ 
# ==========================================================
MODE = 'step' 
STEP_COUNT = 10         
SPECIFIC_LIST = [1000, 10000, 20000] 

PLOT_TYPE = 'interpolate' 
GRID_RES = 400            
MASKING_MULTIPLIER = 1.3  

MY_CMAP = 'nipy_spectral'         
VMIN = 0.0
VMAX = 1.0

# ==========================================================
# CẤU HÌNH LÁT CẮT THỜI GIAN
# ==========================================================
TIME_SLICES = ['first', 0.5, 'last'] 
SLICE_TOLERANCE = 0.05

# ==========================================================
# 3. TỪ ĐIỂN CẤU HÌNH TỰ ĐỘNG THEO PDE
# ==========================================================
PDE_DICT = {
    'Burgers1D': {'xlabel': 'x', 'ylabel': 't', 'aspect': 'auto'},
    'Burgers2D': {'xlabel': 'x', 'ylabel': 'y', 'aspect': 'equal'},
    'Poisson2D_Classic': {'xlabel': 'x', 'ylabel': 'y', 'aspect': 'equal'},
    'Poisson2D_ManyArea': {'xlabel': 'x', 'ylabel': 'y', 'aspect': 'equal'},
    'NS2D_LidDriven': {'xlabel': 'x', 'ylabel': 'y', 'aspect': 'equal'},
    'NS2D_BackStep': {'xlabel': 'x', 'ylabel': 'y', 'aspect': 'equal'},
    'NS2D_LongTime': {'xlabel': 'x', 'ylabel': 'y', 'aspect': 'equal'},
}

current_pde_cfg = PDE_DICT.get(PDE_NAME, {'xlabel': 'Dim 1', 'ylabel': 'Dim 2', 'aspect': 'auto'})
ASPECT_RATIO = current_pde_cfg['aspect']
X_LABEL = current_pde_cfg['xlabel']

# ==========================================================
# 4. HỆ THỐNG XỬ LÝ
# ==========================================================
def get_epoch_list(mode, step_count, specific_list):
    TOTAL_EPOCHS = 20000
    if mode == 'all':
        return list(range(1000, TOTAL_EPOCHS + 1, 1000))
    elif mode == 'step':
        interval = max(1000, TOTAL_EPOCHS // step_count)
        interval = (interval // 1000) * 1000
        return list(range(interval, TOTAL_EPOCHS + 1, interval))
    elif mode == 'specific':
        return specific_list
    return []

def draw_plot(ax, x, y, z):
    if PLOT_TYPE == 'scatter':
        sc = ax.scatter(x, y, c=z, s=1.5, alpha=1.0, cmap=MY_CMAP, vmin=VMIN, vmax=VMAX, edgecolors='none')
        ax.set_aspect(ASPECT_RATIO) 
    else:
        xi = np.linspace(x.min(), x.max(), GRID_RES)
        yi = np.linspace(y.min(), y.max(), GRID_RES)
        X, Y = np.meshgrid(xi, yi)
        
        Z = griddata((x, y), z, (X, Y), method='linear')
        area = (x.max() - x.min()) * (y.max() - y.min())
        avg_spacing = np.sqrt(area / max(len(x), 1))
        
        tree = cKDTree(np.c_[x, y])
        dists, _ = tree.query(np.c_[X.ravel(), Y.ravel()])
        mask = (dists > MASKING_MULTIPLIER * avg_spacing).reshape(X.shape)
        Z[mask] = np.nan 
        
        sc = ax.imshow(Z, extent=[x.min(), x.max(), y.min(), y.max()], 
                       origin='lower', cmap=MY_CMAP, vmin=VMIN, vmax=VMAX, aspect=ASPECT_RATIO)
    ax.set_xlabel(X_LABEL, fontsize=8)
    return sc

def plot_evolution():
    print(f"[*] Đang kiểm tra thư mục log: {LOG_FOLDER}")
    if not os.path.exists(LOG_FOLDER): return
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    epochs_to_plot = get_epoch_list(MODE, STEP_COUNT, SPECIFIC_LIST)
    valid_epochs = [ep for ep in epochs_to_plot if os.path.exists(os.path.join(LOG_FOLDER, f"epoch_{ep}.npz"))]
    if not valid_epochs: return

    sample_data = np.load(os.path.join(LOG_FOLDER, f"epoch_{valid_epochs[0]}.npz"))
    dim = sample_data['coords'].shape[1]
    actual_slices = TIME_SLICES if dim >= 3 else [None]

    num_epochs = len(valid_epochs)
    num_slices = len(actual_slices)
    total_rows = num_epochs * num_slices

    print(f"\n[*] Bài toán: {PDE_NAME} ({dim}D)")
    base_titles = ["Residual ($R_{norm}$)", "Roughness ($L_{norm}$)", "Difficulty ($D$)"]

    # =========================================================
    # 1. XUẤT ẢNH RIÊNG TỪNG EPOCH
    # =========================================================
    for epoch in valid_epochs:
        data = np.load(os.path.join(LOG_FOLDER, f"epoch_{epoch}.npz"))
        coords = data['coords']
        print(f"  -> Đang xử lý Ảnh Riêng cho Epoch {epoch}...")

        fig = plt.figure(figsize=(10, 3 * num_slices))
        # Đưa số cột về đúng 3 (Không chừa cột thứ 4 cho ImageGrid quản lý nữa)
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(num_slices, 3),
                         axes_pad=(0.3, 0.2)) 

        sc_map = None
        for slice_idx, t_val in enumerate(actual_slices):
            if dim >= 3:
                x_all, y_all, t_all = coords[:, 0], coords[:, 1], coords[:, -1]
                t_target = t_all.max() if t_val == 'last' else (t_all.min() if t_val == 'first' else float(t_val))
                tol = SLICE_TOLERANCE * (t_all.max() - t_all.min()) if (t_all.max() - t_all.min()) > 0 else 1e-5
                mask = np.abs(t_all - t_target) <= tol
                x, y = x_all[mask], y_all[mask]
                
                cols = [data['r_norm'][mask], data['l_norm'][mask], data['difficulty'][mask]] if len(x) > 0 else [np.array([])]*3
                print(f"    [+] Lát cắt t = {t_target:.2f} | Lọc được: {len(x)} điểm")
                y_label_str = f"Epoch {epoch}\n\nt = {t_target:.2f}" if slice_idx == 0 else f"t = {t_target:.2f}"
            else:
                x, y = coords[:, 0], coords[:, 1]
                cols = [data['r_norm'], data['l_norm'], data['difficulty']]
                y_label_str = f"Epoch {epoch}" if slice_idx == 0 else ""
                if slice_idx == 0: print(f"    [+] Không gian 2D | Tổng số điểm: {len(x)}")

            for col_idx in range(3):
                ax = grid[slice_idx * 3 + col_idx]
                if len(x) > 0:
                    sc = draw_plot(ax, x, y, cols[col_idx])
                    if sc_map is None: sc_map = sc

                if slice_idx == 0:
                    ax.set_title(base_titles[col_idx], fontsize=10, pad=10)
                if col_idx == 0:
                    ax.set_ylabel(y_label_str, fontsize=8, fontweight='bold', labelpad=15)

        # KHÓA CHIỀU CAO THANH MÀU: Đặt cạnh ô Difficulty hàng đầu tiên (index = 2)
        if sc_map:
            ax_target = grid[2]
            cax = inset_axes(ax_target, width="5%", height="100%", loc='lower left',
                             bbox_to_anchor=(1.05, 0., 1., 1.), bbox_transform=ax_target.transAxes, borderpad=0)
            cbar = fig.colorbar(sc_map, cax=cax)

        fig.savefig(os.path.join(OUTPUT_DIR, f"{PLOT_TYPE}_epoch_{epoch}.png"), dpi=200, bbox_inches='tight')
        plt.close(fig)

    # =========================================================
    # 2. XUẤT ẢNH CHUNG TẤT CẢ EPOCH
    # =========================================================
    print(f"\n[*] Đang kết xuất tệp ảnh chung (Tất cả Epochs)...")
    fig_all = plt.figure(figsize=(10, 3 * total_rows))
    grid_all = ImageGrid(fig_all, 111,
                         nrows_ncols=(total_rows, 3),
                         axes_pad=(0.3, 0.2))

    sc_map_all = None
    grid_row_idx = 0

    for epoch in valid_epochs:
        data = np.load(os.path.join(LOG_FOLDER, f"epoch_{epoch}.npz"))
        coords = data['coords']

        for slice_idx, t_val in enumerate(actual_slices):
            if dim >= 3:
                x_all, y_all, t_all = coords[:, 0], coords[:, 1], coords[:, -1]
                t_target = t_all.max() if t_val == 'last' else (t_all.min() if t_val == 'first' else float(t_val))
                tol = SLICE_TOLERANCE * (t_all.max() - t_all.min()) if (t_all.max() - t_all.min()) > 0 else 1e-5
                mask = np.abs(t_all - t_target) <= tol
                x, y = x_all[mask], y_all[mask]
                
                cols = [data['r_norm'][mask], data['l_norm'][mask], data['difficulty'][mask]] if len(x) > 0 else [np.array([])]*3
                y_label_str = f"Epoch {epoch}\n\nt = {t_target:.2f}" if slice_idx == 0 else f"t = {t_target:.2f}"
            else:
                x, y = coords[:, 0], coords[:, 1]
                cols = [data['r_norm'], data['l_norm'], data['difficulty']]
                y_label_str = f"Epoch {epoch}" if slice_idx == 0 else ""

            for col_idx in range(3):
                ax = grid_all[grid_row_idx * 3 + col_idx]
                if len(x) > 0:
                    sc = draw_plot(ax, x, y, cols[col_idx])
                    if sc_map_all is None: sc_map_all = sc

                if grid_row_idx == 0:
                    ax.set_title(base_titles[col_idx], fontsize=10, pad=10)
                if col_idx == 0:
                    ax.set_ylabel(y_label_str, fontsize=8, fontweight='bold', labelpad=15)
                    
            grid_row_idx += 1

    # KHÓA CHIỀU CAO THANH MÀU TRÊN ẢNH TỔNG: Gắn đúng vào ô vuông Difficulty đầu tiên (index = 2)
    if sc_map_all:
        ax_target_all = grid_all[2]
        cax_all = inset_axes(ax_target_all, width="5%", height="100%", loc='lower left',
                             bbox_to_anchor=(1.05, 0., 1., 1.), bbox_transform=ax_target_all.transAxes, borderpad=0)
        cbar_all = fig_all.colorbar(sc_map_all, cax=cax_all)

    fig_all.savefig(os.path.join(OUTPUT_DIR, f"{PLOT_TYPE}_all_epochs.png"), dpi=200, bbox_inches='tight')
    plt.close(fig_all)
    
    print(f"\n[HOÀN THÀNH TỐI ƯU] Đã xong! Ảnh cực kỳ gọn gàng nằm tại:\n {OUTPUT_DIR}")

if __name__ == "__main__":
    plot_evolution()