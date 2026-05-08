# 卡住问题诊断与修复总结

## 问题位置
`propagate_in_video` 快速完成到 100%，**之后的代码卡住了**

## 根本原因

### 🔴 主要问题：字典访问错误

在 `determine_if_contact_occurred()` 函数（第 ~320 行），代码使用了错误的数组索引：

```python
# ❌ 错误代码
left_gripper_mask = left_gripper_video_segments[idx][1][0]
```

**问题分析：**
- `video_segments[idx]` 返回字典 `{1: array(...)}`
- `[1]` 正确访问了 obj_id=1 对应的掩码数组
- **`[0]` 错误地索引了数组的第一个元素**，而不是整个掩码图像

### 🔴 次要问题：缺少错误处理

如果 SAM2 返回的帧数少于预期，直接访问不存在的字典键会导致 `KeyError` 并无声地失败。

## 应用的修复

### 修复 1：修正数组索引
```python
# ✅ 修复后
left_gripper_mask = left_gripper_video_segments[idx][1]  # 去掉 [0]
right_gripper_mask = right_gripper_video_segments[idx][1]
non_dilated_left_gripper_mask = non_dilated_left_gripper_video_segments[idx][1]
non_dilated_right_gripper_mask = non_dilated_right_gripper_video_segments[idx][1]
```

### 修复 2：添加诊断日志和错误检查
```python
# 检查 idx 是否存在
if idx not in left_gripper_video_segments:
    print(f"ERROR: idx={idx} not found!")
    print(f"Available keys: {list(left_gripper_video_segments.keys())}")
    raise KeyError(...)

# 打印字典结构用于调试
print(f"DEBUG: left_gripper_video_segments[{idx}] = {left_gripper_video_segments[idx]}")
```

### 修复 3：增加进度和状态输出
- SAM2 完成后打印字典键列表
- 主循环开始前显示 frame_indices 内容
- 每个循环迭代显示进度
- describe 函数处理状态

## 修改的文件

| 文件 | 行号 | 修改内容 |
|------|------|---------|
| generate_inference_json_vlms.py | 184-200 | get_gripper_masks() 返回前添加诊断日志 |
| generate_inference_json_vlms.py | 973-983 | 调用 get_gripper_masks() 后添加完成日志 |
| generate_inference_json_vlms.py | ~1000 | 主循环添加进度日志 |
| generate_inference_json_vlms.py | ~1025 | determine_if_contact_occurred() 完成后添加状态 |
| generate_inference_json_vlms.py | ~320 | 修复掩码访问：删除 [0] 索引，添加 KeyError 检查 |

## 预期结果

运行修复后的代码，会在控制台看到：

```
=== Processing LEFT gripper masks ===
[left] Progress: 10/100 | Elapsed: 5.2s | ETA: 46.8s | Speed: 1.92 fps
[left] Progress: 20/100 | Elapsed: 10.4s | ETA: 41.6s | Speed: 1.92 fps
...
[left] Progress: 100/100 | Elapsed: 52.0s | ETA: 0.0s | Speed: 1.92 fps
[left] Dictionary keys: [0, 1, 2, 3, ..., 99]
✓ LEFT masks complete. Keys: [0, 1, 2, 3, ..., 99]

=== Processing RIGHT gripper masks ===
[right] Progress: 100/100 | ...
✓ RIGHT masks complete. Keys: [0, 1, 2, 3, ..., 99]

=== Starting main processing loop ===
frame_indices: [0, 6, 12, 18, 24, ...]
Length: 17

[Loop] idx=0, frame_idx=0
  [Contact] Starting contact determination...
  DEBUG: left_gripper_video_segments[0] = {1: array([[0, 0, 0, ...], ...])}
  ✓ contact=False

[Loop] idx=1, frame_idx=6
  [Contact] Starting contact determination...
  ✓ contact=True
...
```

## 调试命令

如果仍然卡住，运行以下命令来诊断：

```bash
# 只处理前 3 帧
DISPLAY=:99 python scripts/generate_inference_json_vlms.py \
  ... --every_x_frame 6 --sample_rotation \
  2>&1 | head -100  # 只显示前 100 行
```

## 如果问题仍未解决

1. **检查 SAM2 输出帧数**
   - 查看 "Language Dictionary keys" 的数字
   - 与 "Expected: frame_indices" 比较
   
2. **检查掩码数据**
   - 是否为 None 或全零
   - 形状是否为 (128, 128)

3. **检查 GPU 内存**
   ```bash
   nvidia-smi
   ```

4. **减少处理帧率**
   ```bash
   --every_x_frame 12  # 而不是 6
   ```
