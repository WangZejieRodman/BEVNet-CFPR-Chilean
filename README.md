# BEVNet-CFPR-Chilean

基于BEVNet和CFPR的点云位置识别系统，专门针对智利地下矿井数据集（Chilean Underground Mine Dataset）进行优化。

## 项目简介

本项目结合了BEVNet的鸟瞰图（BEV）特征提取方法和CFPR的全局描述符生成技术，实现了高精度的点云位置识别（Place Recognition）。系统采用两阶段训练策略，特别适用于地下矿井等挑战性环境。

### 主要特性

- **两阶段训练**: 先训练Backbone和重叠度预测，再训练全局描述符生成器
- **稀疏卷积**: 使用SPConv实现高效的3D点云处理
- **注意力机制**: 采用自注意力增强特征表达能力
- **NetVLAD聚合**: 生成紧凑的全局描述符用于场景识别

## 系统架构

### 网络结构

1. **Backbone** (modules/net.py:175-219)
   - 基于稀疏2D卷积的下采样网络
   - 输入: BEV体素化表示 `[B, 32, 256, 256]`
   - 输出: 高维特征 `[B, 512, 32, 32]`

2. **OverlapHead** (modules/net.py:263-307)
   - 使用交叉注意力机制预测两个点云的重叠度
   - 输出: 0-1之间的重叠度分数

3. **AttnVLADHead** (modules/net.py:232-261)
   - 自注意力增强 + NetVLAD聚合
   - 输出: 1024维全局描述符

### 训练流程

```
Stage 1: 训练Backbone + OverlapHead
├── 数据: 点云对 (序列100-159)
├── 损失:
│   ├── Pair Loss (点级匹配)
│   ├── Overlap Loss (重叠区域预测)
│   └── Dist Loss (特征距离)
└── 输出: backbone.ckpt, overlap.ckpt

Stage 2: 训练AttnVLADHead
├── 数据: 使用Stage1的backbone提取BEV特征
├── 损失: Triplet Loss (场景级三元组)
└── 输出: attnvlad.ckpt
```

## 数据集

### Chilean Underground Mine Dataset

- **总序列**: 100-209 (110个序列)
- **训练集**: 100-159 (60个序列)
- **测试集**:
  - Database (地图): 160-189 (30个序列)
  - Query (查询): 190-209 (20个序列)

### 数据格式

每个序列包含:
```
{seq}/
├── pointcloud_20m_10overlap/     # 点云文件 (.bin)
├── pointcloud_pos_ori_20m_10overlap.csv  # GPS位置 (x, y, z)
└── BEV_FEA/                      # BEV特征 (由Stage1生成, .npy)
```

## 安装

### 环境要求

- Python >= 3.7
- PyTorch >= 1.10
- CUDA >= 11.0 (GPU训练)

### 依赖安装

```bash
# 创建虚拟环境
conda create -n bevnet python=3.8
conda activate bevnet

# 安装PyTorch (根据CUDA版本选择)
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# 安装SPConv
pip install spconv-cu113

# 安装其他依赖
pip install -r requirements.txt
```

主要依赖:
- torch
- spconv-cu113 (或对应的CUDA版本)
- numpy
- pandas
- pyyaml
- loguru
- tqdm
- tensorboard

## 配置

编辑 `config/config.yml` 文件:

```yaml
# 数据路径
data_root:
  data_root_folder: "/path/to/Chilean_Underground_Mine_Dataset"

# Stage1训练参数
stage1_training_config:
  training_seqs: ["100", "101", ..., "159"]
  out_folder: "/path/to/outputs/stage1_chilean"
  num_iter: 50000
  learning_rate: 1.0e-4
  batch_size: 1
  # 点云范围 (Chilean数据集)
  coords_range_xyz: [-10., -10, -4, 10, 10, 8]
  div_n: [256, 256, 32]  # 体素分辨率

# Stage2训练参数
stage2_training_config:
  pretrained_backbone_model: "/path/to/outputs/stage1_chilean/backbone_final.ckpt"
  out_folder: "/path/to/outputs/stage2_chilean"
  epoch: 20
  learning_rate: 1.0e-5
  batch_size: 2

# 评估参数
evaluate_config:
  database_seqs: ["160", "161", ..., "189"]
  query_seqs: ["190", "191", ..., "209"]
  pos_threshold: 10.0  # GPS距离阈值(米)
  top_k: [1, 5, 10, 25]
```

## 使用方法

### 1. Stage1训练 (Backbone + OverlapHead)

```bash
python train/train_stage1_chilean.py
```

训练参数:
- **迭代次数**: 50,000次
- **学习率**: 1e-4, 每1000次衰减0.99
- **批大小**: 1 (两个点云拼接)
- **数据增强**: 随机旋转、随机遮挡

输出文件:
- `outputs/stage1_chilean/bevnet_final.ckpt` - 完整BEVNet模型
- `outputs/stage1_chilean/backbone_final.ckpt` - Backbone权重 (用于Stage2)
- `outputs/stage1_chilean/overlap_final.ckpt` - OverlapHead权重 (用于评估)
- `outputs/stage1_chilean/tensorboard/` - TensorBoard日志

### 2. Stage2训练 (AttnVLADHead)

```bash
python train/train_stage2_chilean.py
```

**注意**: Stage2会自动检测BEV特征是否存在，如不存在会使用Stage1的backbone自动提取。

训练参数:
- **训练轮数**: 20 epochs
- **学习率**: 1e-5
- **批大小**: 2
- **损失函数**: Triplet Loss (margin=0.5)

输出文件:
- `outputs/stage2_chilean/attnvlad_final.ckpt` - AttnVLAD模型

### 3. 评估

```bash
python evaluate/evaluate_chilean.py
```

评估指标:
- **Recall@K**: K=1, 5, 10, 25
- **平均召回率** (Average Recall)
- **平均最近邻距离** (Average 1% Recall)

评估使用:
- Database: 序列160-189 (构建地图)
- Query: 序列190-209 (查询位置)
- Ground Truth: GPS距离 < 10米

### 4. 特征提取 (可选)

如需单独提取BEV特征:

```bash
python tools/utils/gen_bev_features_chilean.py
```

## 项目结构

```
BEVNet-CFPR-Chilean/
├── config/
│   └── config.yml              # 配置文件
├── modules/
│   ├── net.py                  # 网络架构 (Backbone, BEVNet, AttnVLADHead, OverlapHead)
│   ├── netvlad.py              # NetVLAD实现
│   ├── loss.py                 # Stage2损失函数 (Triplet Loss)
│   └── loss_stage1.py          # Stage1损失函数 (Pair, Overlap, Dist Loss)
├── train/
│   ├── train_stage1_chilean.py # Stage1训练脚本
│   └── train_stage2_chilean.py # Stage2训练脚本
├── evaluate/
│   └── evaluate_chilean.py     # 评估脚本
├── tools/
│   ├── database_chilean.py     # 数据加载器
│   └── utils/
│       ├── gen_bev_features_chilean.py     # BEV特征提取
│       └── utils.py            # 工具函数
└── README.md
```

## 技术细节

### 点云体素化

Chilean数据集的点云范围:
- X: [-10m, 10m]
- Y: [-10m, 10m]
- Z: [-4m, 8m]

体素化分辨率: `256 × 256 × 32`

每个体素大小:
- XY平面: ~0.078m
- Z方向: ~0.375m

### 损失函数

**Stage1**:
- Pair Loss: 点级描述符匹配 + 关键点检测 + 高度回归
- Overlap Loss: 重叠区域二值分类 (BCE Loss)
- Dist Loss: 特征层级的距离损失

**Stage2**:
- Triplet Loss: `max(d(a,p) - d(a,n) + margin, 0)`
  - a: anchor (锚点)
  - p: positive (正样本, GPS距离 < 7m)
  - n: negative (负样本, GPS距离 > 30m)

### 数据增强

- 随机旋转: Z轴旋转 [-π, π]
- 随机遮挡: 随机移除部分点云

## 性能预期

在Chilean Underground Mine Dataset测试集上的典型性能:

- **Recall@1**: ~80-90%
- **Recall@5**: ~90-95%
- **Recall@10**: ~95-98%
- **Recall@25**: ~98-99%

注: 实际性能取决于训练参数、数据质量等因素

## 常见问题

### 1. CUDA Out of Memory

**解决方案**:
- 减小batch_size (Stage1设为1, Stage2设为1)
- 减小体素分辨率 (如改为128×128×16)
- 使用梯度累积

### 2. BEV特征未找到

**解决方案**:
- Stage2会自动提取,确保Stage1的backbone_final.ckpt存在
- 或手动运行: `python tools/utils/gen_bev_features_chilean.py`

### 3. 训练时间过长

**建议**:
- Stage1: ~8-12小时 (单GPU, 50k迭代)
- Stage2: ~2-4小时 (单GPU, 20 epochs)
- 使用更强的GPU或减少迭代次数

## 引用

如果本项目对你的研究有帮助，请引用相关论文:

```bibtex
@article{bevnet,
  title={BEVNet: 3D Place Recognition using Bird's Eye View Representation},
  author={...},
  journal={...},
  year={...}
}

@article{cfpr,
  title={CFPR: Context-aware Feature Point Retrieval for 3D Point Cloud Place Recognition},
  author={...},
  journal={...},
  year={...}
}
```

## 许可证

本项目采用 MIT 许可证。

## 联系方式

如有问题或建议，请提交Issue或联系维护者。

## 致谢

- BEVNet: 提供了BEV表示的启发
- CFPR: 提供了全局描述符生成的框架
- Chilean Underground Mine Dataset: 提供了地下矿井场景数据
