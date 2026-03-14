import numpy as np
import lmdb
import random
import io
import shutil
import re
import random
import numpy as np
import cv2
import pickle
import os
from PIL import Image

# os.environ["QT_QPA_PLATFORM"] = "offscreen"

def load_rlbench_depth(path, near, far):
    """
    Decode RLBench RGB depth image into metric depth (meters)
    """

    depth_rgb = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    # BGR → RGB
    # depth_rgb = cv2.cvtColor(depth_rgb, cv2.COLOR_BGR2RGB)

    depth_rgb = depth_rgb.astype(np.uint32)

    # 24-bit integer
    depth_int = (
        depth_rgb[:, :, 0] +
        depth_rgb[:, :, 1] * 256 +
        depth_rgb[:, :, 2] * 256 * 256
    )

    depth = depth_int.astype(np.float32) / (256**3 - 1)

    # scale to metric
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

    return Rz @ Ry @ Rx  # ZYX order

def get_double_changed(depth_path, rgb_path, pkl_path, indice, is_left, delta_X, delta_Y, delta_Z, roll, pitch, yaw):
    # load camera intrinsics
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

    fx = -K[0][0]
    fy = -K[1][1]
    cx = K[0][2]
    cy = K[1][2]

    depth = load_rlbench_depth(
        depth_path,
        near=near,
        far=far
    )

    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    R = euler_to_matrix(roll, pitch, yaw)
    t = np.array([delta_X, delta_Y, delta_Z])


    H, W = depth.shape
    # RGB
    rgb_acc    = np.zeros((H, W, 3), dtype=np.float32)
    rgb_wacc   = np.zeros((H, W), dtype=np.float32)

    # Depth (inverse depth)
    invz_acc   = np.zeros((H, W), dtype=np.float32)
    invz_wacc  = np.zeros((H, W), dtype=np.float32)

    zbuf = np.full((H, W), np.inf, dtype=np.float32)

    for v in range(H):
        for u in range(W):

            Z = depth[v, u]
            if not np.isfinite(Z) or Z <= 0:
                continue

            # back-project to 3D
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy

            P = np.array([X, Y, Z])

            # 6DoF transform
            Pn = R @ P - R @ t
            Xn, Yn, Zn = Pn

            if Zn <= 0:
                continue

            # re-project
            uf = fx * Xn / Zn + cx
            vf = fy * Yn / Zn + cy

            u0 = int(np.floor(uf))
            v0 = int(np.floor(vf))

            du = uf - u0
            dv = vf - v0

            color = rgb[v, u]
            invZ  = 1.0 / Zn

            weights = [
                (u0,   v0,   (1-du)*(1-dv)),
                (u0+1, v0,   du*(1-dv)),
                (u0,   v0+1, (1-du)*dv),
                (u0+1, v0+1, du*dv)
            ]

            for ui, vi, w in weights:
                if 0 <= ui < W and 0 <= vi < H and w > 0:

                    if Zn < zbuf[vi, ui] + 1e-4:
                        zbuf[vi, ui] = Zn

                        rgb_acc[vi, ui]  += color * w
                        rgb_wacc[vi, ui] += w

                        invz_acc[vi, ui]  += invZ * w
                        invz_wacc[vi, ui] += w

    # =========================
    # 7. Normalize RGB
    # =========================

    rgb_new = np.zeros_like(rgb_acc)
    valid_rgb = rgb_wacc > 1e-6
    rgb_new[valid_rgb] = rgb_acc[valid_rgb] / rgb_wacc[valid_rgb, None]
    rgb_new = np.clip(rgb_new, 0, 255).astype(np.uint8)

    # =========================
    # 8. Recover depth
    # =========================

    depth_new = np.zeros((H, W), dtype=np.float32)
    valid_d = invz_wacc > 1e-6
    depth_new[valid_d] = invz_wacc[valid_d] / invz_acc[valid_d]

    depth_new[depth_new > 8.0] = 0.0

    # =========================
    # 9. Hole filling
    # =========================

    hole_mask = (~valid_d).astype(np.uint8) * 255

    rgb_new   = cv2.inpaint(rgb_new, hole_mask, 3, cv2.INPAINT_NS)
    depth_new = cv2.inpaint(depth_new, hole_mask, 3, cv2.INPAINT_NS)

   # double time change
    rgb_org = rgb
    depth_org = depth
    rgb = rgb_new
    depth = depth_new

    rgb_new   = cv2.inpaint(rgb_new, hole_mask, 3, cv2.INPAINT_NS)
    depth_new = cv2.inpaint(depth_new, hole_mask, 3, cv2.INPAINT_NS)

    depth[depth <= 1e-5] = float('inf')

    delta_X = -delta_X  # 相机沿X轴平移的距离，单位：米
    delta_Y = -delta_Y  # 相机沿Y轴平移的距离，
    delta_Z = -delta_Z  # 相机沿Z轴平移的距离，单位：米
    roll  = -roll  # x-axis
    pitch = -pitch  # y-axis
    yaw   = -yaw   # z-axis
    R = euler_to_matrix(roll, pitch, yaw)
    t = np.array([delta_X, delta_Y, delta_Z])


    # RGB
    rgb_acc    = np.zeros((H, W, 3), dtype=np.float32)
    rgb_wacc   = np.zeros((H, W), dtype=np.float32)

    # Depth (inverse depth)
    invz_acc   = np.zeros((H, W), dtype=np.float32)
    invz_wacc  = np.zeros((H, W), dtype=np.float32)

    zbuf = np.full((H, W), np.inf, dtype=np.float32)

    for v in range(H):
        for u in range(W):

            Z = depth[v, u]
            if not np.isfinite(Z) or Z <= 0:
                continue

            # back-project to 3D
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy

            P = np.array([X, Y, Z])

            # 6DoF transform
            Pn = R @ P - R @ t
            Xn, Yn, Zn = Pn

            if Zn <= 0:
                continue

            # re-project
            uf = fx * Xn / Zn + cx
            vf = fy * Yn / Zn + cy

            u0 = int(np.floor(uf))
            v0 = int(np.floor(vf))

            du = uf - u0
            dv = vf - v0

            color = rgb[v, u]
            invZ  = 1.0 / Zn

            weights = [
                (u0,   v0,   (1-du)*(1-dv)),
                (u0+1, v0,   du*(1-dv)),
                (u0,   v0+1, (1-du)*dv),
                (u0+1, v0+1, du*dv)
            ]

            for ui, vi, w in weights:
                if 0 <= ui < W and 0 <= vi < H and w > 0:

                    if Zn < zbuf[vi, ui] + 1e-4:
                        zbuf[vi, ui] = Zn

                        rgb_acc[vi, ui]  += color * w
                        rgb_wacc[vi, ui] += w

                        invz_acc[vi, ui]  += invZ * w
                        invz_wacc[vi, ui] += w

    # =========================
    # 7. Normalize RGB
    # =========================

    rgb_new = np.zeros_like(rgb_acc)
    valid_rgb = rgb_wacc > 1e-6
    rgb_new[valid_rgb] = rgb_acc[valid_rgb] / rgb_wacc[valid_rgb, None]
    rgb_new = np.clip(rgb_new, 0, 255).astype(np.uint8)

    # =========================
    # 8. Recover depth
    # =========================

    depth_new = np.zeros((H, W), dtype=np.float32)
    valid_d = invz_wacc > 1e-6
    depth_new[valid_d] = invz_wacc[valid_d] / invz_acc[valid_d]

    depth_new[depth_new > 8.0] = 0.0

    # =========================
    # 9. Hole filling
    # =========================

    hole_mask = (~valid_d).astype(np.uint8) * 255

    rgb_new   = cv2.inpaint(rgb_new, hole_mask, 3, cv2.INPAINT_NS)
    depth_new = cv2.inpaint(depth_new, hole_mask, 3, cv2.INPAINT_NS)

    return depth_org, rgb_org, depth_new, rgb_new

if __name__ == "__main__":
    
    num_demo = 25
    num_data = 500
    lmdb_path = '/home/zsh/dcoda/D-Aug/data/co_lift_ball_test.lmdb'
    data_path = '/home/zsh/dcoda/RLBench/tools/data/rlbench_data_test/coordinated_lift_ball/all_variations/episodes'
    lower_bound = 0.02
    upper_bound = 0.05
    angle_range = 0.5
    cutoff_index = 50

    if os.path.exists(lmdb_path):
        shutil.rmtree(lmdb_path)  # 删除整个 LMDB 目录
    env_tmp = lmdb.open(lmdb_path, map_size=int(1e9))
    with env_tmp.begin(write=True) as txn:
        for epoch in range(num_data):
            print("Epoch:", epoch)
            idx = random.randint(0, num_demo - 1)
            is_left = random.choice([True, False])
            if is_left:
                camera_name = 'left'
            else:
                camera_name = 'right'

            depth_path = os.path.join(data_path, f'episode{idx}', f'wrist_{camera_name}_depth')
            rgb_path = os.path.join(data_path, f'episode{idx}', f'wrist_{camera_name}_rgb')
            pkl_path = os.path.join(data_path, f'episode{idx}', 'low_dim_obs.pkl')

            # 获取文件列表
            files = os.listdir(depth_path)

            # 筛选 depth_*.npy 文件，并提取 indice
            indices = []
            pattern = re.compile(r"depth_(\d+)\.png")  # 正则匹配数字
            for f in files:
                m = pattern.match(f)
                if m:
                    indices.append(int(m.group(1)))

            if indices:
                max_index = max(indices)
            else:
                print("没有找到 depth_*.png 文件")

            indice = random.randint(0, min(max_index, cutoff_index))

            # small camera shift
            delta_X = random.uniform(lower_bound, upper_bound)*random.choice([-1, 1])  # 相机沿X轴平移的距离，单位：米
            delta_Y = random.uniform(lower_bound, upper_bound)*random.choice([-1, 1])  # 相机沿Y轴平移的距离，单位：米
            delta_Z = random.uniform(lower_bound, upper_bound)*random.choice([-1, 1])  # 相机沿Z轴平移的距离，单位：米

            # rotation (degrees)
            roll  = np.deg2rad(random.uniform(-angle_range, angle_range))    # x-axis
            pitch = np.deg2rad(random.uniform(-angle_range, angle_range))    # y-axis
            yaw   = np.deg2rad(random.uniform(-angle_range, angle_range))   # z-axis

            depth_path = os.path.join(depth_path, f"depth_{indice:04d}.png")
            rgb_path = os.path.join(rgb_path, f"rgb_{indice:04d}.png")
            
            depth_org, rgb_org, depth_new, rgb_new = get_double_changed(depth_path, 
                                                                        rgb_path, pkl_path, indice, is_left, delta_X, delta_Y, delta_Z, roll, pitch, yaw)

            rgb_file = os.path.join(rgb_path)
            img_bgr = cv2.imread(rgb_file)  # img 为 numpy.ndarray，BGR 格式
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            ex_arr = rgb_org
            ex_buf = io.BytesIO()
            np.save(ex_buf, ex_arr)
            ex_key = f'data{epoch:05d}_original_rgb'.encode()
            txn.put(ex_key, ex_buf.getvalue())


            ex_arr = rgb_new
            ex_buf = io.BytesIO()
            np.save(ex_buf, ex_arr)
            ex_key = f'data{epoch:05d}_changed_rgb'.encode()
            txn.put(ex_key, ex_buf.getvalue())

    env_tmp.close()