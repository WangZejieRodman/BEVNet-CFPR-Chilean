# tools/utils/gen_bev_features_chilean_rot.py
"""
提取Chilean数据集的旋转增强BEV特征
使用Stage1训练好的Backbone
对原始点云进行随机Z轴旋转后提取特征
"""
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
import yaml
from tqdm import tqdm
import torch
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from modules.net import Backbone
from tools.utils import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def rotate_point_cloud(pc):
    """
    随机旋转点云（绕Z轴）
    与database_chilean.py中的数据增强保持一致

    Args:
        pc: 点云 (N, 3) numpy array

    Returns:
        rotated_pc: 旋转后的点云 (N, 3)
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, -sinval, 0],
                                [sinval, cosval, 0],
                                [0, 0, 1]])
    rotated_pc = np.dot(pc, rotation_matrix)
    return rotated_pc


def load_and_rotate_pc(filename):
    """
    加载原始点云并应用随机Z轴旋转

    Args:
        filename: 点云文件路径 (.bin)

    Returns:
        rotated_pc: 旋转后的点云 (N, 3)
    """
    # 加载原始点云 (与utils.py中的load_pc_file保持一致)
    raw_data = np.fromfile(filename, dtype=np.float64)

    # 判断数据维度
    if raw_data.shape[0] % 3 == 0:
        # Chilean格式: float64, 3维 [x, y, z]
        pc = raw_data.reshape(-1, 3).astype(np.float32)
    elif raw_data.shape[0] % 4 == 0:
        # 可能是KITTI格式
        raw_data_f32 = np.fromfile(filename, dtype=np.float32)
        if raw_data_f32.shape[0] % 4 == 0:
            pc = raw_data_f32.reshape(-1, 4)[:, :3]
        else:
            pc = raw_data[:raw_data.shape[0] // 3 * 3].reshape(-1, 3).astype(np.float32)
    else:
        raise ValueError(f"Cannot parse point cloud file {filename}")

    # 应用随机Z轴旋转
    rotated_pc = rotate_point_cloud(pc)

    return rotated_pc


def voxelize_point_cloud(pc, coords_range_xyz, div_n):
    """
    将点云体素化（与utils.py中的逻辑一致）

    Args:
        pc: 点云 (N, 3) numpy array
        coords_range_xyz: 体素化范围
        div_n: 体素分辨率

    Returns:
        voxel_out: 体素化后的occupancy grid
    """
    pc = torch.from_numpy(pc).to(device)

    # 计算体素索引
    ids = utils.load_voxel(pc, coords_range_xyz=coords_range_xyz, div_n=div_n)

    # 生成occupancy grid
    voxel_out = torch.zeros(div_n)
    voxel_out[ids[:, 0], ids[:, 1], ids[:, 2]] = 1

    return voxel_out


def extract_chilean_rot(net, scan_folder, dst_folder, batch_num=1,
                        coords_range_xyz=[-10, -10, -4, 10, 10, 8],
                        div_n=[256, 256, 32]):
    """
    提取旋转增强的Chilean数据集BEV特征

    Args:
        net: Backbone网络
        scan_folder: 点云文件夹路径
        dst_folder: 特征保存路径
        batch_num: batch大小
        coords_range_xyz: 体素化范围
        div_n: 体素分辨率
    """
    files = sorted(os.listdir(scan_folder))
    files = [os.path.join(scan_folder, v) for v in files if v.endswith('.bin')]
    length = len(files)

    if length == 0:
        print(f"Warning: No .bin files found in {scan_folder}")
        return

    net.eval()

    for q_index in tqdm(range(length // batch_num + (1 if length % batch_num > 0 else 0)),
                        total=(length // batch_num + (1 if length % batch_num > 0 else 0)),
                        desc="Extracting rotated features"):
        batch_files = files[q_index * batch_num:min((q_index + 1) * batch_num, length)]

        # 加载并旋转每个点云
        batch_voxels = []
        for file in batch_files:
            # 加载原始点云并随机旋转
            rotated_pc = load_and_rotate_pc(file)

            # 体素化
            voxel = voxelize_point_cloud(rotated_pc, coords_range_xyz, div_n)
            batch_voxels.append(voxel)

        # 转换为batch tensor
        batch_voxels = torch.stack(batch_voxels).to(device)

        # 提取BEV特征
        with torch.no_grad():
            fea_out = net(batch_voxels).cpu().numpy()

        # 保存特征
        for i in range(len(batch_files)):
            fea_file = os.path.join(dst_folder, os.path.basename(batch_files[i]).replace('.bin', '.npy'))
            np.save(fea_file, fea_out[i])


if __name__ == "__main__":
    # ==================== 参数配置 ====================
    # 数据根目录
    data_root = "/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times/chilean_NoRot_NoScale"

    # Backbone模型路径
    backbone_ckpt = "/home/wzj/pan1/BEVNet-CFPR/outputs/stage1_chilean/backbone_final.ckpt"

    # 要处理的序列（评估用的Database和Query）
    # Database: 160-189
    database_seqs = [str(i) for i in range(160, 190)]
    # Query: 190-209
    query_seqs = [str(i) for i in range(190, 210)]
    # 所有评估序列
    seqs = database_seqs + query_seqs

    # 体素化参数
    coords_range_xyz = [-10., -10, -4, 10, 10, 8]
    div_n = [256, 256, 32]

    # Batch大小
    batch_num = 16
    # ==================================================

    print("=" * 60)
    print("Extracting Rotated BEV Features for Chilean Dataset")
    print("=" * 60)
    print(f"Data root: {data_root}")
    print(f"Backbone checkpoint: {backbone_ckpt}")
    print(f"Sequences to process: {len(seqs)}")
    print(f"  Database (160-189): {len(database_seqs)} sequences")
    print(f"  Query (190-209): {len(query_seqs)} sequences")
    print(f"Coords range: {coords_range_xyz}")
    print(f"Voxel resolution: {div_n}")
    print(f"Random rotation: Z-axis (0-360°)")
    print("=" * 60)

    # 加载Backbone模型
    print(f"\nLoading backbone from {backbone_ckpt}")
    net = Backbone(32).to(device)
    checkpoint = torch.load(backbone_ckpt)
    state_dict = checkpoint['state_dict']

    # 转换权重形状（如果需要）
    for key in list(state_dict.keys()):
        if 'conv.3.weight' in key:
            weight = state_dict[key]
            if len(weight.shape) == 4 and weight.shape[0] < weight.shape[2]:
                state_dict[key] = weight.permute(3, 0, 1, 2).contiguous()

    net.load_state_dict(state_dict)
    print("✓ Backbone loaded successfully")

    # 处理每个序列
    for seq in seqs:
        print(f"\n{'=' * 60}")
        print(f"Processing sequence {seq}")
        print(f"{'=' * 60}")

        scan_folder = os.path.join(data_root, seq, "pointcloud_20m_10overlap")
        dst_folder = os.path.join(data_root, seq, "BEV_FEA_ROT")

        if not os.path.exists(scan_folder):
            print(f"⚠ Warning: Scan folder not found: {scan_folder}")
            continue

        # 检查是否已存在旋转特征
        if os.path.exists(dst_folder) and len(os.listdir(dst_folder)) > 0:
            print(f"✓ Rotated features already exist, skipping sequence {seq}")
            continue

        # 创建输出目录
        os.makedirs(dst_folder, exist_ok=True)

        # 提取旋转特征
        extract_chilean_rot(net, scan_folder, dst_folder, batch_num,
                            coords_range_xyz, div_n)

        print(f"✓ Sequence {seq} completed")

    print(f"\n{'=' * 60}")
    print("All rotated BEV features extracted successfully!")
    print(f"{'=' * 60}")
    print("\nOutput structure:")
    print("  <seq>/BEV_FEA_ROT/*.npy - Rotated BEV features")
    print("\nNext steps:")
    print("  1. Run evaluate_chilean.py with rotated features")
    print("  2. Compare performance with non-rotated features")