import numpy as np
import lmdb
import random
import io
import shutil
import re
import cv2
import pickle
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch

DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
_GRID_CACHE = {}

def _get_uv_grid(h, w, device):
    key = (h, w, str(device))
    if key not in _GRID_CACHE:
        v, u = torch.meshgrid(
            torch.arange(h, device=device, dtype=torch.float32),
            torch.arange(w, device=device, dtype=torch.float32),
            indexing="ij",
        )
        _GRID_CACHE[key] = (u, v)
    return _GRID_CACHE[key]

@torch.no_grad()
def splat_once_gpu(depth_np, rgb_np, fx, fy, cx, cy, R_np, t_np, device=DEVICE):
    depth = torch.from_numpy(depth_np).to(device=device, dtype=torch.float32)
    rgb = torch.from_numpy(rgb_np).to(device=device, dtype=torch.float32)

    h, w = depth.shape
    u, v = _get_uv_grid(h, w, device)

    valid = torch.isfinite(depth) & (depth > 0)
    if valid.sum() == 0:
        return np.zeros((h, w), dtype=np.float32), np.zeros((h, w, 3), dtype=np.uint8)

    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    P = torch.stack([x, y, z], dim=-1)[valid]      # [N,3]
    C = rgb[valid]                                 # [N,3]

    R = torch.as_tensor(R_np, device=device, dtype=torch.float32)
    t = torch.as_tensor(t_np, device=device, dtype=torch.float32)

    Pn = (P - t) @ R.T
    xn, yn, zn = Pn[:, 0], Pn[:, 1], Pn[:, 2]

    front = zn > 0
    if front.sum() == 0:
        return np.zeros((h, w), dtype=np.float32), np.zeros((h, w, 3), dtype=np.uint8)

    xn, yn, zn = xn[front], yn[front], zn[front]
    C = C[front]

    uf = fx * xn / zn + cx
    vf = fy * yn / zn + cy

    u0 = torch.floor(uf).to(torch.int64)
    v0 = torch.floor(vf).to(torch.int64)
    du = uf - u0.float()
    dv = vf - v0.float()

    ui = torch.stack([u0, u0 + 1, u0, u0 + 1], dim=1)   # [N,4]
    vi = torch.stack([v0, v0, v0 + 1, v0 + 1], dim=1)
    ww = torch.stack(
        [(1 - du) * (1 - dv), du * (1 - dv), (1 - du) * dv, du * dv],
        dim=1,
    )

    zn4 = zn[:, None].expand(-1, 4)
    invz4 = (1.0 / zn)[:, None].expand(-1, 4)
    c4 = C[:, None, :].expand(-1, 4, -1)

    inb = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h) & (ww > 0)

    ui = ui[inb]
    vi = vi[inb]
    ww = ww[inb]
    zn4 = zn4[inb]
    invz4 = invz4[inb]
    c4 = c4[inb]

    idx = vi * w + ui
    n_pix = h * w

    zbuf = torch.full((n_pix,), float("inf"), device=device, dtype=torch.float32)
    zbuf.scatter_reduce_(0, idx, zn4, reduce="amin", include_self=True)

    vis = zn4 <= (zbuf[idx] + 1e-4)
    idx = idx[vis]
    ww = ww[vis]
    invz4 = invz4[vis]
    c4 = c4[vis]

    rgb_acc = torch.zeros((n_pix, 3), device=device, dtype=torch.float32)
    rgb_wacc = torch.zeros((n_pix,), device=device, dtype=torch.float32)
    invz_acc = torch.zeros((n_pix,), device=device, dtype=torch.float32)
    invz_wacc = torch.zeros((n_pix,), device=device, dtype=torch.float32)

    rgb_acc.index_add_(0, idx, c4 * ww[:, None])
    rgb_wacc.index_add_(0, idx, ww)
    invz_acc.index_add_(0, idx, invz4 * ww)
    invz_wacc.index_add_(0, idx, ww)

    rgb_new = torch.zeros_like(rgb_acc)
    valid_rgb = rgb_wacc > 1e-6
    rgb_new[valid_rgb] = rgb_acc[valid_rgb] / rgb_wacc[valid_rgb, None]
    rgb_new = rgb_new.reshape(h, w, 3).clamp(0, 255).byte().cpu().numpy()

    depth_new = torch.zeros((n_pix,), device=device, dtype=torch.float32)
    valid_d = invz_wacc > 1e-6
    depth_new[valid_d] = invz_wacc[valid_d] / invz_acc[valid_d]
    depth_new = depth_new.reshape(h, w).cpu().numpy().astype(np.float32)
    depth_new[depth_new > 8.0] = 0.0

    hole_mask = (~valid_d.reshape(h, w).cpu().numpy()).astype(np.uint8) * 255
    rgb_new = cv2.inpaint(rgb_new, hole_mask, 3, cv2.INPAINT_NS)
    depth_new = cv2.inpaint(depth_new, hole_mask, 3, cv2.INPAINT_NS)

    return depth_new, rgb_new

def load_rlbench_depth(path, near, far):
    depth_rgb = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth_rgb is None:
        raise FileNotFoundError(f"Depth image not found at {path}")
    
    depth_rgb = depth_rgb.astype(np.uint32)
    depth_int = (
        depth_rgb[:, :, 0] +
        depth_rgb[:, :, 1] * 256 +
        depth_rgb[:, :, 2] * 256 * 256
    )
    depth = depth_int.astype(np.float32) / (256**3 - 1)
    depth = near + depth * (far - near)
    return depth

def euler_to_matrix(roll, pitch, yaw):
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])

    Ry = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx

# ---------------- CPU 数据准备阶段 ----------------
def cpu_prepare_one(epoch, depth_path, rgb_path, pkl_path, indice, is_left, params):
    """这部分运行在 CPU ProcessPool 中，用于处理慢速的 I/O (imread, pickle load)"""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    obs = data[indice]
    
    if is_left:
        K = obs.misc['wrist_left_camera_intrinsics']
        near = obs.misc['wrist_left_camera_near']
        far = obs.misc['wrist_left_camera_far']
    else:
        K = obs.misc['wrist_right_camera_intrinsics']
        near = obs.misc['wrist_right_camera_near']
        far = obs.misc['wrist_right_camera_far']

    fx, fy = -K[0][0], -K[1][1]
    cx, cy = K[0][2], K[1][2]

    depth = load_rlbench_depth(depth_path, near=near, far=far)

    rgb = cv2.imread(rgb_path)
    if rgb is None:
        raise FileNotFoundError(f"RGB image not found at {rgb_path}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    return {
        "epoch": epoch,
        "depth": depth,
        "rgb": rgb,
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "params": params
    }

# ---------------- GPU 前向计算阶段 ----------------
def get_double_changed_loaded(data_dict):
    """这部分运行在主进程中，用于将预加载好的矩阵打入 GPU 极速演算"""
    depth = data_dict["depth"]
    rgb = data_dict["rgb"]
    fx, fy, cx, cy = data_dict["fx"], data_dict["fy"], data_dict["cx"], data_dict["cy"]
    delta_X, delta_Y, delta_Z, roll, pitch, yaw = data_dict["params"]

    R = euler_to_matrix(roll, pitch, yaw)
    t = np.array([delta_X, delta_Y, delta_Z], dtype=np.float32)

    # 第一次变换
    depth_new, rgb_new = splat_once_gpu(depth, rgb, fx, fy, cx, cy, R, t, device=DEVICE)

    rgb_org = rgb
    depth_org = depth

    # 第二次变换
    depth_mid = depth_new.copy()
    depth_mid[depth_mid <= 1e-5] = np.inf

    R2 = euler_to_matrix(-roll, -pitch, -yaw)
    t2 = np.array([-delta_X, -delta_Y, -delta_Z], dtype=np.float32)

    depth_new2, rgb_new2 = splat_once_gpu(depth_mid, rgb_new, fx, fy, cx, cy, R2, t2, device=DEVICE)
    
    return rgb_org, rgb_new2


if __name__ == "__main__":
    # 使用 spawn 保证配合 CUDA 多线程时不会出底层死锁
    ctx = mp.get_context("spawn")
    
    # all parameters except for gpu
    num_demo = 25
    num_data = 500 # 50000
    lmdb_path = '/home/zsh/dcoda/D-Aug/data/co_lift_ball_test.lmdb'
    data_path = '/home/zsh/dcoda/RLBench/tools/data/rlbench_data_test/coordinated_lift_ball/all_variations/episodes'
    lower_bound = 0.02
    upper_bound = 0.05
    angle_range = 0.5
    cutoff_index = 50

    if os.path.exists(lmdb_path):
        shutil.rmtree(lmdb_path)

    # 1. 预先收集所有任务参数
    tasks = []
    for epoch in range(num_data):
        idx = random.randint(0, num_demo - 1)
        is_left = random.choice([True, False])
        camera_name = 'left' if is_left else 'right'

        ep_depth_path = os.path.join(data_path, f'episode{idx}', f'wrist_{camera_name}_depth')
        ep_rgb_path = os.path.join(data_path, f'episode{idx}', f'wrist_{camera_name}_rgb')
        pkl_path = os.path.join(data_path, f'episode{idx}', 'low_dim_obs.pkl')

        if not os.path.exists(ep_depth_path):
            continue

        files = os.listdir(ep_depth_path)
        pattern = re.compile(r"depth_(\d+)\.png")
        indices = [int(m.group(1)) for f in files if (m := pattern.match(f))]
        
        if not indices:
            continue

        max_index = max(indices)
        indice = random.randint(0, min(max_index, cutoff_index))

        delta_X = random.uniform(lower_bound, upper_bound) * random.choice([-1, 1])
        delta_Y = random.uniform(lower_bound, upper_bound) * random.choice([-1, 1])
        delta_Z = random.uniform(lower_bound, upper_bound) * random.choice([-1, 1])
        roll    = np.deg2rad(random.uniform(-angle_range, angle_range))
        pitch   = np.deg2rad(random.uniform(-angle_range, angle_range))
        yaw     = np.deg2rad(random.uniform(-angle_range, angle_range))

        img_depth_path = os.path.join(ep_depth_path, f"depth_{indice:04d}.png")
        img_rgb_path = os.path.join(ep_rgb_path, f"rgb_{indice:04d}.png")

        tasks.append((
            epoch, img_depth_path, img_rgb_path, pkl_path, indice, is_left, 
            (delta_X, delta_Y, delta_Z, roll, pitch, yaw)
        ))
 
    print(f"Total tasks prepared: {len(tasks)}")

    # 打开 LMDB 准备写入
    env_tmp = lmdb.open(lmdb_path, map_size=int(1e10))
    
    # 2. 启动并发任务
    # 设置 max_workers，一般设置为物理核心数比如 4~8 视你的 CPU 性能定
    # 保持主进程拿数据推 GPU，子进程读取硬盘并解析 pickle 和 img
    with env_tmp.begin(write=True) as txn:
        with ProcessPoolExecutor(max_workers=8, mp_context=ctx) as executor:
            # 提交 CPU 预处理任务
            future_to_epoch = {
                executor.submit(cpu_prepare_one, *task): task[0] 
                for task in tasks
            }

            for future in as_completed(future_to_epoch):
                epoch = future_to_epoch[future]
                try:
                    data_dict = future.result()
                    
                    # 取出预处理好的 numpy 数据，交给主进程的 GPU 进行运算
                    # GPU 运算是在本进程所以不怕显存错乱或者上下文冲突
                    rgb_org, rgb_new = get_double_changed_loaded(data_dict)

                    # 写入 LMDB (仍然是单进程串行写入，保证稳定)
                    ex_buf = io.BytesIO()
                    np.save(ex_buf, rgb_org)
                    txn.put(f'data{epoch:05d}_original_rgb'.encode(), ex_buf.getvalue())

                    ex_buf = io.BytesIO()
                    np.save(ex_buf, rgb_new)
                    txn.put(f'data{epoch:05d}_changed_rgb'.encode(), ex_buf.getvalue())
                    if epoch % 100 == 0:
                        print(f"Finished processing and saved epoch: {epoch}")

                except Exception as exc:
                    print(f"Epoch {epoch} generated an exception: {exc}")

    env_tmp.close()
    print("All tasks completed.")