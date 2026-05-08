

# D-CODA Pipeline

> Step-by-step instructions for data generation, preprocessing, training, and sampling.

---

## 1. Generate Demos

| | |
|---|---|
| **Env** | `rlbench_py39` |
| **Folder** | `dcoda/RLBench/tools/` |
| **Output** | `dcoda/RLBench/tools/data/rlbench_data/XXXX/all_variations/episodes` |

```bash
python dataset_generator_bimanual.py
```

---

## 2. Convert RLBench Training Data into D-CODA Format

| | |
|---|---|
| **Env** | `rlbench_py39` |
| **Folder** | `dcoda/` |
| **Output** | `dcoda/DMD/instance-data/XXXXXX_coordinated_lift_ball_100_org_data_w_depth_v1_run1/coordinated_lift_ball_dmd_bimanual` |

```bash
bash format.sh
```

---

## 3. Preprocess Data into New Format

| | |
|---|---|
| **Env** | `rlbench_py39` |
| **Folder** | `dcoda/D-Aug/` |
| **Output** | `dcoda/D-Aug/data/XXXX.lmdb` |

```bash
python preprocess_gpu.py
```

---

## 4. Train Diffusion Models

| | |
|---|---|
| **Env** | `dmd-diffusion` |
| **Folder** | `dcoda/D-Aug/dmd_diffusion` |
| **Output** | `dcoda/D-Aug/dmd_diffusion/src/output` |

```bash
bash train.sh
```

---

## 5. Try Sampling

Use `sample.ipynb` or `change_view_demo`.

---

## 6. Generate JSON and HTML

| | |
|---|---|
| **Env** | `sam2` |
| **Folder** | `dcoda/DMD/src/` |
| **Output** | `dcoda/DMD/instance-data/XXXXXX_coordinated_lift_ball_100_org_data_w_depth_v1_run1/coordinated_lift_ball_dmd_bimanual_v1` |
| | `dcoda/DMD/data/traj_plots` |

```bash
bash generate_json_D-Aug.sh
```

---

## 7. Displace Before Sampling

| | |
|---|---|
| **Env** | `sam2` |
| **Folder** | `dcoda/D-Aug/` |
| **Output** | `dcoda/DMD/instance-data/XXXXXX_coordinated_lift_ball_100_org_data_w_depth_v1_run1/coordinated_lift_ball_dmd_bimanual_v1/displaced_all` |

```bash
python displace_before_sample.py
```

---

## 8. Sample

| | |
|---|---|
| **Env** | `dmd-diffusion` |
| **Folder** | `dcoda/DMD/src/` |
| **Output** | `dcoda/DMD/instance-data/XXXXXX_coordinated_lift_ball_100_org_data_w_depth_v1_run1/coordinated_lift_ball_dmd_bimanual_v1/images` |

```bash
bash synthesize_D-Aug.sh
```