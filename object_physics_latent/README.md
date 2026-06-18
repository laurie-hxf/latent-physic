# Object Physics Latent Dataset Tools

该目录中的数据工具用于组织“多个物体、每个物体多条 MuJoCo 轨迹”的训练数据。
所有实现均为新文件，不修改现有单物体 MuJoCo/Newton 数据流程。

## 数据组织

每个物理物体对应一个标准 MuJoCo trajectory NPZ。一个 manifest 将多个 NPZ 组织为：

```text
object-level split:
  train / validation / test

per-object episode split:
  context / query / eval
```

默认每个物体使用：

```text
context: 15%
query:   75%
eval:    10%
```

Manifest 会保存：

```text
object_id
physical_config_id
shape_id
trajectory_npz
object_split
context/query/eval episode indices
friction_spec
trajectory counts and lengths
```

相同 `physical_config_id` 的重复物体实例不会跨越 object train/validation/test split。

## 已生成的正式多物体数据

正式数据已经生成：

```text
mujoco/outputs/object_physics_latent_box_partitions_48x2000_min300/
```

包含：

```text
48 objects
2000 trajectories per object
96000 total trajectories

left_right:  16 objects
front_back:  16 objects
center_ends: 16 objects

context episodes: 14400
query episodes:   72000
eval episodes:     9600

每条轨迹至少包含 300 个有效 simulation steps，也就是至少 301 帧、约 0.6 秒。
物体在 300 steps 后仍未静止时会继续记录到静止，因此低摩擦物体的轨迹可以更长。
```

三个 partition family 使用相同的 `0.20 x 0.10 x 0.05 m` box 外形、`1.0 kg` 质量和
相同惯量：

```text
left_right:
  沿 top-view y 轴分割，即 local x=0
  得到两个 0.10 x 0.10 正方形区域

front_back:
  沿 top-view x 轴分割，即 local y=0
  得到两个 0.20 x 0.05 窄长方形区域

center_ends:
  沿 local x 分成左端、中心、右端
  左右两端共享 ends friction，中心使用 center friction
```

正式 manifest：

```text
mujoco/outputs/object_physics_latent_box_partitions_48x2000_min300/manifest.json
```

旧的允许较早静止终止的数据仍保留在：

```text
mujoco/outputs/object_physics_latent_box_partitions_48x2000/
```

更完整的数据结构、字段说明和读取示例见：

```text
docs/object_physics_latent_dataset_guide.md
```

## 物体预览与图片库

正式数据中的每个物体都有一张预览图。左侧按照实际 MuJoCo XML geom 和 friction
metadata 显示俯视摩擦分区，右侧显示该物体第 0 条真实推动轨迹。

打开可点击的 48 物体 HTML 图片库：

```text
mujoco/outputs/object_physics_latent_box_partitions_48x2000_min300/previews/gallery.html
```

总览图：

```text
mujoco/outputs/object_physics_latent_box_partitions_48x2000_min300/previews/all_objects_contact_sheet.png
mujoco/outputs/object_physics_latent_box_partitions_48x2000_min300/previews/left_right_contact_sheet.png
mujoco/outputs/object_physics_latent_box_partitions_48x2000_min300/previews/front_back_contact_sheet.png
mujoco/outputs/object_physics_latent_box_partitions_48x2000_min300/previews/center_ends_contact_sheet.png
```

重新从已有数据生成图片，不会重新运行 MuJoCo：

```bash
python object_physics_latent/render_dataset_previews.py
```

## 重新生成正式多物体数据

先只生成计划，不运行 MuJoCo：

```bash
python object_physics_latent/generate_box_object_datasets.py \
  --output-root mujoco/outputs/object_physics_latent_box_partitions_48x2000_min300 \
  --num-objects 48 \
  --episodes-per-object 2000 \
  --action-scale rotation \
  --minimum-recorded-steps 300 \
  --workers 8 \
  --seed 0 \
  --plan-only
```

确认 `generation_plan.json` 后，删除 `--plan-only` 正式生成。生成脚本支持
`--skip-existing`，中断后可以继续执行：

```bash
python object_physics_latent/generate_box_object_datasets.py \
  --output-root mujoco/outputs/object_physics_latent_box_partitions_48x2000_min300 \
  --num-objects 48 \
  --episodes-per-object 2000 \
  --action-scale rotation \
  --minimum-recorded-steps 300 \
  --workers 8 \
  --seed 0
```

生成器会均衡生成不同强度的 `left_right`、`front_back` 和 `center_ends` 摩擦物体。
Manifest 和 sampler 不依赖具体 friction family；加入 T 形数据后仍可使用相同数据
接口。

## 从已有 NPZ 构建 Manifest

可以传入一个或多个 NPZ，也可以传入递归包含 NPZ 的目录：

```bash
python object_physics_latent/build_manifest.py \
  mujoco/outputs/object_physics_latent_box40 \
  --output mujoco/outputs/object_physics_latent_box40/manifest.json \
  --seed 0
```

目录扫描会忽略不包含 `trajectories/columns/episode_lengths` 的其他 NPZ。

## 校验与采样

校验 manifest，并实际加载数据采样一个训练 step：

```bash
python object_physics_latent/validate_dataset.py \
  mujoco/outputs/object_physics_latent_box40/manifest.json \
  --split train \
  --objects-per-step 4 \
  --context-trajectories-per-view 4 \
  --query-trajectories-per-view 128 \
  --context-window-steps 300 \
  --query-window-steps 300
```

每个采样物体返回：

```text
context_a:
  padded local-frame encoder features + valid mask

context_b:
  与 context_a 不重叠的 encoder features + valid mask

query_a:
  128 条可直接传给现有 Newton runtime 的 MujocoTrajectory

query_b:
  与 query_a 不重叠的 128 条 MujocoTrajectory
```

Context 和 query 来自固定的不重叠 episode pool。Context 两个 view 互不重叠，query
两个 view 也互不重叠。

## 模型接口

第二步模型已经实现为新的独立文件，不修改现有单物体 Newton 拟合流程：

```text
object_physics_latent/encoder.py
object_physics_latent/friction_decoder.py
object_physics_latent/model.py
object_physics_latent/test_models.py
```

包含：

```text
TrajectoryGRUEncoder:
  单条轨迹 step-MLP + GRU -> trajectory embedding

TrajectorySetEncoder / ObjectPhysicsEncoder:
  多条 context 轨迹 mean pooling -> deterministic unit-norm object latent
  contrastive learning operates directly on the same latent consumed by decoder

LatentConditionedFrictionDecoder:
  point-feature branch with trajectory-latent FiLM modulation -> bounded per-point friction

TrajectoryConditionedFrictionModel:
  context trajectories -> latent -> per-point friction
```

当前模型接口已经支持真实 dataset loader 输出的 `(B, K, T, 12)` context feature 和
valid mask。DINO 特征是可选的；没有 DINO feature 时可以先使用 position-only point
feature，后续接入 Newton/DINO 训练时再传入真实 surface-point DINO feature。

Latent 相关 loss：

```text
same_object_consistency_loss
symmetric_info_nce_loss
latent_regularization_losses
```

正式训练脚本默认使用 FiLM decoder，并加入不依赖真实 friction 标签的 cyclic latent-swap
rollout ranking loss。对于物体 `i` 的同一组 query trajectories，分别使用正确 context
latent `z_i` 和另一个物体的 latent `z_j` 生成 friction field；目标要求正确 latent 的
query rollout loss 更低。使用 `--swap-query-trajectories-per-view` 控制额外 paired
rollout 的计算量。

## 端到端训练入口

第三步训练入口已经放在新文件：

```text
object_physics_latent/train.py
```

它走旧 DINO-MLP 的摩擦系数路线：trajectory encoder 输出 object latent，
latent-conditioned DINO-MLP 只预测每个 surface point 的 `mu`，Newton rollout 继续复用
原来的 surface-point friction kernel。`contact_stiffness`、`contact_damping` 等接触参数
仍然是命令行固定配置，不由模型预测；这个入口不使用 `mu/ke/kd/kf` contact-value field。

快速 dry-run，只检查真实 manifest、DINO 特征对齐、模型维度和采样：

```bash
python object_physics_latent/train.py \
  --experiment-dir outputs/object_physics_latent_dino_mlp_dryrun \
  --device cuda:0 \
  --dry-run \
  --objects-per-step 1 \
  --context-trajectories-per-view 2 \
  --query-trajectories-per-view 2 \
  --context-window-steps 20 \
  --query-window-steps 20 \
  --active-object-limit 1 \
  --active-trajectories-per-object 2 \
  --active-window-steps 20
```

最小训练 smoke：

```bash
python object_physics_latent/train.py \
  --experiment-dir outputs/object_physics_latent_dino_mlp_smoke \
  --device cuda:0 \
  --opt-iters 1 \
  --objects-per-step 1 \
  --context-trajectories-per-view 2 \
  --query-trajectories-per-view 2 \
  --context-window-steps 20 \
  --query-window-steps 20 \
  --active-object-limit 1 \
  --active-trajectories-per-object 2 \
  --active-window-steps 20 \
  --checkpoint-every 1
```

正式训练时把窗口恢复到 300 steps，并把 query batch 提高到 64 或 128：

```bash
python object_physics_latent/train.py \
  --manifest mujoco/outputs/object_physics_latent_box_partitions_48x2000_min300/manifest.json \
  --experiment-dir outputs/object_physics_latent_dino_mlp \
  --device cuda:0 \
  --opt-iters 200 \
  --objects-per-step 2 \
  --context-trajectories-per-view 4 \
  --query-trajectories-per-view 64 \
  --context-window-steps 300 \
  --query-window-steps 300 \
  --surface-point-spacing 0.01 \
  --active-object-limit 4 \
  --active-trajectories-per-object 64
```

训练脚本只训练和保存 checkpoint，不会自动运行 eval。

## Checkpoint Latent 与摩擦可视化

从已有 checkpoint 离线生成每个物体的 latent PCA、latent 向量热图、latent 距离矩阵，
以及底面预测摩擦与 ground truth 的逐物体对比。该命令只读取 checkpoint、manifest、
DINO feature 和 context 轨迹，不运行 Newton rollout：

```bash
python object_physics_latent/visualize_checkpoint.py \
  outputs/object_physics_latent_runs/<experiment>/<experiment>_best.pt \
  --device cuda:0
```

默认输出到：

```text
outputs/object_physics_latent_runs/<experiment>/visualization/<checkpoint_name>/
```

其中 `gallery.html` 是可点击的逐物体图片库，`object_metrics.csv` 保存每个物体的摩擦
恢复误差，`object_latent_friction_data.npz` 保存逐物体 latent、预测 friction、ground
truth friction 和 PCA 坐标，便于后续分析。

## Python 接口

```python
from pathlib import Path

import numpy as np
import torch

from object_physics_latent.dataset import ObjectPhysicsDataset
from object_physics_latent.encoder import encoder_batch_to_torch
from object_physics_latent.friction_decoder import build_point_conditioning_features
from object_physics_latent.model import TrajectoryConditionedFrictionModel


dataset = ObjectPhysicsDataset(
    Path("mujoco/outputs/object_physics_latent_box_partitions_48x2000_min300/manifest.json"),
    cache_size=8,
)
samples = dataset.sample_training_step(
    split="train",
    objects_per_step=4,
    context_trajectories_per_view=4,
    query_trajectories_per_view=128,
    context_window_steps=300,
    query_window_steps=300,
    rng=np.random.default_rng(0),
)

sample = samples[0]
context_features, context_mask = encoder_batch_to_torch(sample.context_a)
point_features_np, point_metadata, _ = build_point_conditioning_features(
    local_surface_points=np.array([[-0.1, -0.05, -0.025], [0.1, 0.05, -0.025]], dtype=np.float32),
    half_extents=np.array([0.1, 0.05, 0.025], dtype=np.float32),
    dino_features=None,
    position_frequencies=2,
)
model = TrajectoryConditionedFrictionModel.from_dimensions(point_feature_dim=point_metadata.input_dim)
output = model(
    context_features=context_features,
    context_valid_mask=context_mask,
    point_features=torch.as_tensor(point_features_np, dtype=torch.float32),
)
print(output.latent.shape, output.friction.shape)
```

Loader 按物体懒加载并使用 LRU cache。压缩 NPZ 必须解压后才能随机访问，因此训练时
应根据 CPU 内存调整 `cache_size`，避免每个 iteration 重复加载相同物体。

## 测试

```bash
python -m unittest object_physics_latent.test_dataset -v
python -m unittest object_physics_latent.test_models -v
```
