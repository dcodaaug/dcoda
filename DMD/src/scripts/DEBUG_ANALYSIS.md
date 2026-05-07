# 代码卡住分析 - `propagate_in_video` 问题

## 问题描述
在 `get_gripper_masks()` 函数的 `propagate_in_video()` 之后代码一直卡住。

## 代码位置
- **文件**: `generate_inference_json_vlms.py`
- **函数**: `get_gripper_masks (第124行)`
- **卡住位置**: 第140行 - `for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):`
- **调用位置**: 第961-962行（main函数中）

## 原因分析

### 1. **SAM2 推理速度慢（最可能原因）**
   - `propagate_in_video()` 是一个**生成器**，逐帧进行神经网络推理
   - 每帧都需要进行特征提取、Transformer计算等
   - 对于100+ 帧的视频，这可能需要**几分钟甚至更长**
   - 如果分辨率高（如128x128），速度会更慢

### 2. **缺少进度反馈**
   - 代码中没有任何日志或进度条
   - 用户无法判断代码是否正在运行还是真的卡住了

### 3. **可能的GPU问题**
   - GPU内存不足导致推理变得非常慢
   - 或导致GPU进程挂起

### 4. **数据加载问题**
   - 从磁盘读取大量图像到内存可能很慢
   - SAM2的视频处理可能涉及额外的数据预处理

## 解决方案

### 方案 A：添加进度监控（推荐，立即见效）
```python
def get_gripper_masks(predictor, images_folder, gripper_mask, frame_indices, which_arm="left"):
    """
    Use SAM2 to generate gripper masks for a trajectory of images.
    """
    inference_state = predictor.init_state(video_path=images_folder)
    predictor.reset_state(inference_state)
    _, out_obj_ids, video_res_masks = predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1,
        mask=gripper_mask,
    )
    
    video_segments = {}
    non_dilated_video_segments = {}
    
    # 获取总帧数
    total_frames = len(frame_indices)
    processed_count = 0
    
    print(f"[{which_arm}] Starting propagation for {total_frames} frames...")
    start_time = time.time()
    
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        local_dict = {}
        non_dilated_local_dict = {}
        for i, out_obj_id in enumerate(out_obj_ids):
            final_mask = (out_mask_logits[i] > 0.0).cpu().numpy().astype(np.uint8)
            half_height = int(final_mask.shape[1]/2)
            final_mask[0, :half_height, :] = 0
            processed_final_mask = cv2.dilate(final_mask, np.ones((23, 15), np.uint8), iterations=1)
            processed_final_mask = cv2.morphologyEx(processed_final_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
            non_dilated_local_dict[out_obj_id] = final_mask
            local_dict[out_obj_id] = processed_final_mask
        
        video_segments[out_frame_idx] = local_dict
        non_dilated_video_segments[out_frame_idx] = non_dilated_local_dict
        
        processed_count += 1
        elapsed = time.time() - start_time
        fps = processed_count / elapsed if elapsed > 0 else 0
        remaining = (total_frames - processed_count) / fps if fps > 0 else 0
        
        print(f"[{which_arm}] Progress: {processed_count}/{total_frames} | "
              f"Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s")
    
    print(f"[{which_arm}] Propagation completed in {time.time() - start_time:.1f}s")
    return video_segments, non_dilated_video_segments
```

### 方案 B：添加超时和错误处理
```python
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("SAM2 propagation timed out")

def get_gripper_masks(predictor, images_folder, gripper_mask, frame_indices, which_arm="left", timeout=3600):
    """
    Use SAM2 to generate gripper masks for a trajectory of images.
    """
    # 设置超时（例如1小时）
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        inference_state = predictor.init_state(video_path=images_folder)
        predictor.reset_state(inference_state)
        _, out_obj_ids, video_res_masks = predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            mask=gripper_mask,
        )
        
        video_segments = {}
        non_dilated_video_segments = {}
        
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            # ... 处理代码 ...
            video_segments[out_frame_idx] = local_dict
            non_dilated_video_segments[out_frame_idx] = non_dilated_local_dict
        
        signal.alarm(0)  # 取消超时
        return video_segments, non_dilated_video_segments
        
    except TimeoutError:
        print(f"ERROR: SAM2 propagation timed out after {timeout}s")
        signal.alarm(0)
        raise
```

### 方案 C：检查GPU状态和内存
```python
import torch

def get_gripper_masks(predictor, images_folder, gripper_mask, frame_indices, which_arm="left"):
    """
    Use SAM2 to generate gripper masks for a trajectory of images.
    """
    # 检查GPU状态
    if torch.cuda.is_available():
        print(f"[{which_arm}] GPU available: {torch.cuda.get_device_name(0)}")
        print(f"[{which_arm}] GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    else:
        print(f"[{which_arm}] WARNING: No GPU available, using CPU (will be very slow)")
    
    # ... 其他代码 ...
```

### 方案 D：验证输入和数据完整性
```python
def get_gripper_masks(predictor, images_folder, gripper_mask, frame_indices, which_arm="left"):
    """
    Use SAM2 to generate gripper masks for a trajectory of images.
    """
    # 检查图像文件是否存在
    import os
    image_files = sorted([f for f in os.listdir(images_folder) if f.endswith('.jpg')])
    print(f"[{which_arm}] Found {len(image_files)} images in {images_folder}")
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {images_folder}")
    
    # 检查初始mask
    if gripper_mask is None or np.sum(gripper_mask) == 0:
        raise ValueError(f"Invalid gripper_mask for {which_arm}")
    
    print(f"[{which_arm}] Initial mask shape: {gripper_mask.shape}, non-zero pixels: {np.sum(gripper_mask)}")
    
    # ... 其他代码 ...
```

## 建议的调试步骤

1. **立即添加进度日志**（方案A）- 可以立即看到是否真的在卡住
2. **检查GPU内存** - 运行 `nvidia-smi` 或在代码中检查
3. **验证输入数据** - 确保图像和mask都正确加载
4. **逐帧测试** - 只处理前2-3帧看是否有问题
5. **检查SAM2版本** - 确保使用的是最新版本
6. **调整batch_size或推理参数** - 如果SAM2支持

## 快速修复建议
在 `get_gripper_masks()` 函数的138行（`for` 循环前）添加：
```python
print(f"Starting {which_arm} gripper mask propagation for {len(frame_indices)} frames...")
import time
start = time.time()
```

在155行（循环内）添加：
```python
if (out_frame_idx + 1) % 10 == 0:  # 每10帧打印一次
    elapsed = time.time() - start
    print(f"[{which_arm}] Processed frame {out_frame_idx + 1} in {elapsed:.1f}s")
```
