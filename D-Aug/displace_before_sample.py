import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch


DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
GRID_CACHE = {}
MAX_RECORDS = 5000

DATA_JSON_PATH = Path(
    "/home/zsh/dcoda/DMD/instance-data/"
    "260310_coordinated_lift_ball_100_org_data_w_depth_v1_run1/"
    "coordinated_lift_ball_dmd_bimanual_v1/data.json"
)
OUTPUT_DIR = DATA_JSON_PATH.parent / "displaced_all"


def get_uv_grid(height, width, device):
    key = (height, width, str(device))
    if key not in GRID_CACHE:
        v, u = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        GRID_CACHE[key] = (u, v)
    return GRID_CACHE[key]


@torch.no_grad()
def splat_once_gpu(depth_np, rgb_np, fx, fy, cx, cy, rot_np, trans_np, device=DEVICE):
    depth = torch.from_numpy(depth_np).to(device=device, dtype=torch.float32)
    rgb = torch.from_numpy(rgb_np).to(device=device, dtype=torch.float32)

    height, width = depth.shape
    u, v = get_uv_grid(height, width, device)

    valid = torch.isfinite(depth) & (depth > 0)
    if valid.sum() == 0:
        return np.zeros((height, width), dtype=np.float32), np.zeros((height, width, 3), dtype=np.uint8)

    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = torch.stack([x, y, z], dim=-1)[valid]
    colors = rgb[valid]

    rot = torch.as_tensor(rot_np, device=device, dtype=torch.float32)
    trans = torch.as_tensor(trans_np, device=device, dtype=torch.float32)
    points_new = (points - trans) @ rot.T

    xn, yn, zn = points_new[:, 0], points_new[:, 1], points_new[:, 2]
    in_front = zn > 0
    if in_front.sum() == 0:
        return np.zeros((height, width), dtype=np.float32), np.zeros((height, width, 3), dtype=np.uint8)

    xn, yn, zn = xn[in_front], yn[in_front], zn[in_front]
    colors = colors[in_front]

    uf = fx * xn / zn + cx
    vf = fy * yn / zn + cy

    u0 = torch.floor(uf).to(torch.int64)
    v0 = torch.floor(vf).to(torch.int64)
    du = uf - u0.float()
    dv = vf - v0.float()

    ui = torch.stack([u0, u0 + 1, u0, u0 + 1], dim=1)
    vi = torch.stack([v0, v0, v0 + 1, v0 + 1], dim=1)
    weights = torch.stack(
        [(1 - du) * (1 - dv), du * (1 - dv), (1 - du) * dv, du * dv],
        dim=1,
    )

    zn4 = zn[:, None].expand(-1, 4)
    invz4 = (1.0 / zn)[:, None].expand(-1, 4)
    c4 = colors[:, None, :].expand(-1, 4, -1)

    in_bounds = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height) & (weights > 0)
    ui = ui[in_bounds]
    vi = vi[in_bounds]
    weights = weights[in_bounds]
    zn4 = zn4[in_bounds]
    invz4 = invz4[in_bounds]
    c4 = c4[in_bounds]

    idx = vi * width + ui
    n_pix = height * width

    zbuf = torch.full((n_pix,), float("inf"), device=device, dtype=torch.float32)
    zbuf.scatter_reduce_(0, idx, zn4, reduce="amin", include_self=True)

    visible = zn4 <= (zbuf[idx] + 1e-4)
    idx = idx[visible]
    weights = weights[visible]
    invz4 = invz4[visible]
    c4 = c4[visible]

    rgb_acc = torch.zeros((n_pix, 3), device=device, dtype=torch.float32)
    rgb_wacc = torch.zeros((n_pix,), device=device, dtype=torch.float32)
    invz_acc = torch.zeros((n_pix,), device=device, dtype=torch.float32)
    invz_wacc = torch.zeros((n_pix,), device=device, dtype=torch.float32)

    rgb_acc.index_add_(0, idx, c4 * weights[:, None])
    rgb_wacc.index_add_(0, idx, weights)
    invz_acc.index_add_(0, idx, invz4 * weights)
    invz_wacc.index_add_(0, idx, weights)

    rgb_new = torch.zeros_like(rgb_acc)
    valid_rgb = rgb_wacc > 1e-6
    rgb_new[valid_rgb] = rgb_acc[valid_rgb] / rgb_wacc[valid_rgb, None]
    rgb_new = rgb_new.reshape(height, width, 3).clamp(0, 255).byte().cpu().numpy()

    depth_new = torch.zeros((n_pix,), device=device, dtype=torch.float32)
    valid_depth = invz_wacc > 1e-6
    depth_new[valid_depth] = invz_wacc[valid_depth] / invz_acc[valid_depth]
    depth_new = depth_new.reshape(height, width).cpu().numpy().astype(np.float32)
    depth_new[depth_new > 10.0] = 0.0

    hole_mask = (~valid_depth.reshape(height, width).cpu().numpy()).astype(np.uint8) * 255
    rgb_new = cv2.inpaint(rgb_new, hole_mask, 3, cv2.INPAINT_NS)
    depth_new = cv2.inpaint(depth_new, hole_mask, 3, cv2.INPAINT_NS)

    return depth_new, rgb_new


def euler_to_matrix(roll, pitch, yaw):
    rx = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]],
        dtype=np.float32,
    )
    ry = np.array(
        [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]],
        dtype=np.float32,
    )
    rz = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]],
        dtype=np.float32,
    )
    return rz @ ry @ rx


def resolve_dataset_root(data_json_path):
    marker = f"{os.sep}instance-data{os.sep}"
    full = str(data_json_path)
    idx = full.find(marker)
    if idx == -1:
        return data_json_path.parent
    return Path(full[:idx])


def resolve_image_path(raw_path, data_json_path, dataset_root):
    p = Path(raw_path)
    if p.is_absolute() and p.exists():
        return p

    candidate1 = (data_json_path.parent / p).resolve()
    if candidate1.exists():
        return candidate1

    raw_str = str(raw_path)
    if "instance-data/" in raw_str:
        rel = raw_str[raw_str.index("instance-data/") :]
        candidate2 = (dataset_root / rel).resolve()
        if candidate2.exists():
            return candidate2

    candidate3 = (dataset_root / raw_str.lstrip("./")).resolve()
    if candidate3.exists():
        return candidate3

    raise FileNotFoundError(f"Cannot resolve image path: {raw_path}")


def infer_depth_path(image_path):
    stem, _ = os.path.splitext(str(image_path))
    return Path(f"{stem}_depth.png")


def load_depth(depth_path, near, far):
    depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"Depth image not found: {depth_path}")

    if depth_raw.ndim == 3 and depth_raw.shape[2] >= 3:
        depth_u32 = depth_raw.astype(np.uint32)
        depth_int = depth_u32[:, :, 0] + depth_u32[:, :, 1] * 256 + depth_u32[:, :, 2] * 256 * 256
        depth_norm = depth_int.astype(np.float32) / (256**3 - 1)
        return near + depth_norm * (far - near)

    if depth_raw.dtype == np.uint16:
        depth_norm = depth_raw.astype(np.float32) / 65535.0
        return near + depth_norm * (far - near)

    if depth_raw.dtype == np.uint8:
        depth_norm = depth_raw.astype(np.float32) / 255.0
        return near + depth_norm * (far - near)

    depth_float = depth_raw.astype(np.float32)
    if depth_float.max() <= 1.0:
        return near + depth_float * (far - near)
    return depth_float


def get_intrinsics(height, width, focal_y):
    focal = max(abs(float(focal_y)), 1e-6)
    fx = -focal * width
    fy = -focal * height
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    return fx, fy, cx, cy


def process_one_image(
    image_path,
    delta_xyz,
    euler_xyz,
    focal_y,
    depth_info,
):
    rgb_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise FileNotFoundError(f"RGB image not found: {image_path}")

    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    # print(rgb.shape)
    # rgb = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
    depth_path = infer_depth_path(image_path)
    depth_key = depth_path.name
    meta = depth_info.get(depth_key)
    if meta is None:
        raise KeyError(f"Depth info not found for {depth_key}")

    camera_type = "left" if "left" in depth_key else "right"
    near = float(meta[f"wrist_{camera_type}_camera_near"])
    far = float(meta[f"wrist_{camera_type}_camera_far"])

    depth = load_depth(depth_path, near=near, far=far)
    height, width = depth.shape
    fx, fy, cx, cy = get_intrinsics(height, width, focal_y)

    rot = euler_to_matrix(float(euler_xyz[0]), float(euler_xyz[1]), float(euler_xyz[2]))
    trans = np.asarray(delta_xyz, dtype=np.float32)

    _, rgb_new = splat_once_gpu(depth, rgb, fx, fy, cx, cy, rot, trans, device=DEVICE)
    return cv2.cvtColor(rgb_new, cv2.COLOR_RGB2BGR)


def iter_camera_items(item):
    if all(k in item for k in ("img_path", "camera_delta_xyz", "camera_euler_xyz")):
        yield {
            "tag": "single",
            "img_path": item["img_path"],
            "delta": item["camera_delta_xyz"],
            "euler": item["camera_euler_xyz"],
            "focal_y": item.get("focal_y", -0.8660254),
        }
        return

    if all(k in item for k in ("left_img", "camera_delta_xyz_left", "camera_euler_xyz_left")):
        yield {
            "tag": "left",
            "img_path": item["left_img"],
            "delta": item["camera_delta_xyz_left"],
            "euler": item["camera_euler_xyz_left"],
            "focal_y": item.get("wl_focal_y", -0.8660254),
        }

    if all(k in item for k in ("right_img", "camera_delta_xyz_right", "camera_euler_xyz_right")):
        yield {
            "tag": "right",
            "img_path": item["right_img"],
            "delta": item["camera_delta_xyz_right"],
            "euler": item["camera_euler_xyz_right"],
            "focal_y": item.get("wr_focal_y", -0.8660254),
        }


def main():
    data_json_path = DATA_JSON_PATH
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(data_json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not records:
        raise ValueError(f"No records found in {data_json_path}")

    records = records[:MAX_RECORDS]
    dataset_root = resolve_dataset_root(data_json_path)

    first_cam = None
    for item in records:
        cams = list(iter_camera_items(item))
        if cams:
            first_cam = cams[0]
            break

    if first_cam is None:
        raise ValueError("No valid camera items found in the first 50 records.")

    first_img = resolve_image_path(first_cam["img_path"], data_json_path, dataset_root)
    depth_info_path = first_img.parent.parent / "depth_info.json"
    if not depth_info_path.exists():
        raise FileNotFoundError(f"Depth info file not found: {depth_info_path}")

    with open(depth_info_path, "r", encoding="utf-8") as f:
        depth_info = json.load(f)

    saved = 0
    for idx, item in enumerate(records):
        for cam in iter_camera_items(item):
            try:
                image_path = resolve_image_path(cam["img_path"], data_json_path, dataset_root)
                displaced_bgr = process_one_image(
                    image_path=image_path,
                    delta_xyz=cam["delta"],
                    euler_xyz=cam["euler"],
                    focal_y=cam["focal_y"],
                    depth_info=depth_info,
                )

                out_name = f"{idx:06d}_{cam['tag']}.png"
                out_path = output_dir / out_name
                cv2.imwrite(str(out_path), displaced_bgr)
                saved += 1
            except Exception as exc:
                print(f"[WARN] record={idx}, camera={cam['tag']} failed: {exc}")
        if idx % 100 ==0:
            print(f"Processed records: {idx}")
    print(f"Processed records: {len(records)}")
    print(f"Saved displaced images: {saved}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()