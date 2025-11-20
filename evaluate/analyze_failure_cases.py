# evaluate/analyze_failure_cases.py
"""
分析Stage1 BEV特征的失败case
找出正样本对中特征距离最大的100个case
输出点云路径和分层BEV可视化
"""
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from loguru import logger
import cv2


def load_bev_features_and_info(root, seqs):
    """
    加载BEV特征、几何位置、文件路径

    Returns:
        features: BEV特征列表 [(512, 32, 32), ...]
        poses: 几何位置列表 [(x, y), ...]
        file_paths: 点云文件路径列表
        seq_ids: 序列ID列表
    """
    features = []
    poses = []
    file_paths = []
    seq_ids = []

    for seq in tqdm(seqs, desc="加载数据"):
        fea_folder = os.path.join(root, seq, "BEV_FEA")
        pc_folder = os.path.join(root, seq, "pointcloud_20m_10overlap")
        pose_file = os.path.join(root, seq, "pointcloud_pos_ori_20m_10overlap.csv")

        if not os.path.exists(fea_folder) or not os.path.exists(pose_file):
            continue

        # 读取位置数据
        df = pd.read_csv(pose_file)
        pose_dict = {}
        for _, row in df.iterrows():
            timestamp = str(int(row['timestamp']))
            x, y = row['x'], row['y']
            pose_dict[timestamp] = (x, y)

        # 读取特征
        fea_files = sorted([f for f in os.listdir(fea_folder) if f.endswith('.npy')])

        for fea_file in fea_files:
            timestamp = fea_file.replace('.npy', '')

            if timestamp not in pose_dict:
                continue

            # 特征路径
            fea_path = os.path.join(fea_folder, fea_file)
            fea = np.load(fea_path)  # (512, 32, 32)

            # 点云路径
            pc_path = os.path.join(pc_folder, f"{timestamp}.bin")

            features.append(fea)
            poses.append(pose_dict[timestamp])
            file_paths.append(pc_path)
            seq_ids.append(seq)

    return features, poses, file_paths, seq_ids


def pooling_features(features, method='avg'):
    """对BEV特征进行池化"""
    descriptors = []

    for fea in features:
        if method == 'avg':
            desc = fea.mean(axis=(1, 2))  # (512,)
        elif method == 'max':
            desc = fea.max(axis=(1, 2))
        else:
            raise ValueError(f"Unknown pooling: {method}")

        # L2归一化
        desc = desc / (np.linalg.norm(desc) + 1e-8)
        descriptors.append(desc)

    return np.array(descriptors)


def compute_feature_distance(desc1, desc2):
    """计算两个描述符之间的欧氏距离"""
    return np.linalg.norm(desc1 - desc2)


def find_failure_cases(db_features, db_poses, db_paths, db_seq_ids,
                       q_features, q_poses, q_paths, q_seq_ids,
                       geo_threshold=7.0, top_k=100, pooling='avg'):
    """
    找出最差的正样本对

    Args:
        geo_threshold: 几何距离阈值（米）
        top_k: 返回前K个最差case
        pooling: 池化方式

    Returns:
        failure_cases: list of dict，每个dict包含case信息
    """
    print(f"\n查找失败case（几何距离<{geo_threshold}m，特征距离最大的{top_k}个）...")

    # 池化
    db_desc = pooling_features(db_features, pooling)
    q_desc = pooling_features(q_features, pooling)

    # 遍历所有query-database对
    positive_pairs = []

    for i in tqdm(range(len(q_poses)), desc="查找正样本对"):
        q_pos = np.array(q_poses[i])

        for j in range(len(db_poses)):
            db_pos = np.array(db_poses[j])

            # 计算几何距离
            geo_dist = np.linalg.norm(q_pos - db_pos)

            # 如果是正样本对
            if geo_dist < geo_threshold:
                # 计算特征距离
                fea_dist = compute_feature_distance(q_desc[i], db_desc[j])

                positive_pairs.append({
                    'query_idx': i,
                    'db_idx': j,
                    'query_path': q_paths[i],
                    'db_path': db_paths[j],
                    'query_seq': q_seq_ids[i],
                    'db_seq': db_seq_ids[j],
                    'geo_distance': geo_dist,
                    'feature_distance': fea_dist,
                    'query_feature': q_features[i],  # (512, 32, 32)
                    'db_feature': db_features[j]
                })

    print(f"找到 {len(positive_pairs)} 个正样本对")

    # 按特征距离降序排序
    positive_pairs.sort(key=lambda x: x['feature_distance'], reverse=True)

    # 取前top_k个
    failure_cases = positive_pairs[:top_k]

    print(f"最差的{top_k}个正样本对:")
    print(f"  特征距离范围: {failure_cases[-1]['feature_distance']:.4f} ~ {failure_cases[0]['feature_distance']:.4f}")
    print(
        f"  几何距离范围: {min(c['geo_distance'] for c in failure_cases):.2f}m ~ {max(c['geo_distance'] for c in failure_cases):.2f}m")

    return failure_cases


def visualize_bev_layers(bev_feature, output_folder):
    """
    可视化BEV特征的32层

    Args:
        bev_feature: (512, 32, 32) BEV特征
        output_folder: 输出文件夹
    """
    os.makedirs(output_folder, exist_ok=True)

    # bev_feature是(512, 32, 32)，但我们需要的是原始的occupancy grid
    # 这里的bev_feature已经是backbone提取的特征，不是原始的二值occupancy
    # 我们需要重新加载原始点云并体素化
    pass  # 这个函数需要重新设计


def load_and_voxelize_pointcloud(pc_path, coords_range_xyz=[-10., -10, -4, 10, 10, 8],
                                 div_n=[256, 256, 32]):
    """
    加载点云并体素化为occupancy grid

    Args:
        pc_path: 点云文件路径 (.bin)
        coords_range_xyz: 体素化范围
        div_n: 体素分辨率

    Returns:
        voxel_grid: (256, 256, 32) 二值occupancy grid
    """
    # 读取点云
    raw_data = np.fromfile(pc_path, dtype=np.float64)

    if raw_data.shape[0] % 3 != 0:
        raise ValueError(f"Invalid point cloud file {pc_path}")

    points = raw_data.reshape(-1, 3).astype(np.float32)

    # 计算体素索引
    div = [(coords_range_xyz[3] - coords_range_xyz[0]) / div_n[0],
           (coords_range_xyz[4] - coords_range_xyz[1]) / div_n[1],
           (coords_range_xyz[5] - coords_range_xyz[2]) / div_n[2]]

    id_x = (points[:, 0] - coords_range_xyz[0]) / div[0]
    id_y = (points[:, 1] - coords_range_xyz[1]) / div[1]
    id_z = (points[:, 2] - coords_range_xyz[2]) / div[2]

    all_id = np.stack([id_x, id_y, id_z], axis=1).astype(np.int32)

    # 过滤超出范围的点
    mask = (all_id[:, 0] >= 0) & (all_id[:, 1] >= 0) & (all_id[:, 2] >= 0) & \
           (all_id[:, 0] < div_n[0]) & (all_id[:, 1] < div_n[1]) & (all_id[:, 2] < div_n[2])

    all_id = all_id[mask]

    # 构建occupancy grid
    voxel_grid = np.zeros(div_n, dtype=np.uint8)
    voxel_grid[all_id[:, 0], all_id[:, 1], all_id[:, 2]] = 1

    return voxel_grid


def save_bev_layers_as_images(voxel_grid, output_folder):
    """
    保存occupancy grid的32层为png图像

    Args:
        voxel_grid: (256, 256, 32) 二值occupancy grid
        output_folder: 输出文件夹
    """
    os.makedirs(output_folder, exist_ok=True)

    # 遍历32层
    for z in range(voxel_grid.shape[2]):
        # 提取第z层 (256, 256)
        layer = voxel_grid[:, :, z]

        # 转换为0-255图像（0=黑色，255=白色）
        img = (layer * 255).astype(np.uint8)

        # 保存（z从下到上，层0是最底层-4m附近，层31是最顶层8m附近）
        img_path = os.path.join(output_folder, f"layer_{z:02d}.png")
        cv2.imwrite(img_path, img)


def analyze_failure_cases(config):
    """主函数：分析失败case"""
    root = config["data_root"]["data_root_folder"]

    # Database和Query序列
    database_seqs = [str(i) for i in range(160, 190)]
    query_seqs = [str(i) for i in range(190, 210)]

    # 参数
    geo_threshold = 7.0  # 几何距离阈值
    top_k = 100  # 找出前100个最差case
    pooling = 'avg'

    logger.info("=" * 80)
    logger.info("Stage1 BEV特征失败case分析")
    logger.info("=" * 80)
    logger.info(f"几何距离阈值: {geo_threshold}m")
    logger.info(f"分析前{top_k}个最差case")
    logger.info(f"池化方式: {pooling}")
    logger.info("=" * 80)

    # 加载Database数据
    logger.info("\n[步骤1] 加载Database数据...")
    db_features, db_poses, db_paths, db_seq_ids = load_bev_features_and_info(root, database_seqs)
    logger.info(f"Database: {len(db_features)} 个点云")

    # 加载Query数据
    logger.info("\n[步骤2] 加载Query数据...")
    q_features, q_poses, q_paths, q_seq_ids = load_bev_features_and_info(root, query_seqs)
    logger.info(f"Query: {len(q_features)} 个点云")

    # 查找失败case
    logger.info("\n[步骤3] 查找失败case...")
    failure_cases = find_failure_cases(
        db_features, db_poses, db_paths, db_seq_ids,
        q_features, q_poses, q_paths, q_seq_ids,
        geo_threshold=geo_threshold,
        top_k=top_k,
        pooling=pooling
    )

    # 创建输出目录
    output_root = "/home/wzj/pan1/BEVNet-CFPR/outputs/failure_cases"
    os.makedirs(output_root, exist_ok=True)

    # 生成报告并可视化
    logger.info(f"\n[步骤4] 生成报告和BEV分层可视化...")

    # 创建CSV报告
    report_data = []

    for rank, case in enumerate(tqdm(failure_cases, desc="处理失败case")):
        case_id = f"case_{rank:03d}"

        # 创建该case的文件夹
        case_folder = os.path.join(output_root, case_id)
        os.makedirs(case_folder, exist_ok=True)

        # Query BEV分层可视化
        query_bev_folder = os.path.join(case_folder, "query_bev_layers")
        logger.info(f"  [{rank + 1}/{top_k}] 处理 {case_id}...")

        try:
            # 加载并体素化Query点云
            q_voxel = load_and_voxelize_pointcloud(case['query_path'])
            save_bev_layers_as_images(q_voxel, query_bev_folder)

            # Database BEV分层可视化
            db_bev_folder = os.path.join(case_folder, "db_bev_layers")
            db_voxel = load_and_voxelize_pointcloud(case['db_path'])
            save_bev_layers_as_images(db_voxel, db_bev_folder)

            # 记录到报告
            report_data.append({
                'rank': rank + 1,
                'case_id': case_id,
                'query_path': case['query_path'],
                'query_seq': case['query_seq'],
                'query_bev_folder': query_bev_folder,
                'db_path': case['db_path'],
                'db_seq': case['db_seq'],
                'db_bev_folder': db_bev_folder,
                'geo_distance': f"{case['geo_distance']:.3f}",
                'feature_distance': f"{case['feature_distance']:.4f}"
            })

        except Exception as e:
            logger.error(f"  处理 {case_id} 失败: {e}")
            continue

    # 保存CSV报告
    report_df = pd.DataFrame(report_data)
    report_csv = os.path.join(output_root, "failure_cases_report.csv")
    report_df.to_csv(report_csv, index=False)

    logger.info(f"\n报告已保存: {report_csv}")
    logger.info(f"BEV分层图已保存在: {output_root}/case_XXX/")

    # 打印前10个case
    logger.info("\n" + "=" * 80)
    logger.info("前10个最差case:")
    logger.info("=" * 80)

    for i in range(min(10, len(report_data))):
        case = report_data[i]
        logger.info(f"\n[Case {i + 1}]")
        logger.info(f"  Query:  {case['query_path']}")
        logger.info(f"  DB:     {case['db_path']}")
        logger.info(f"  几何距离: {case['geo_distance']}m")
        logger.info(f"  特征距离: {case['feature_distance']}")
        logger.info(f"  BEV分层: {case['query_bev_folder']}")
        logger.info(f"          {case['db_bev_folder']}")

    logger.info("\n" + "=" * 80)
    logger.info("分析完成！")
    logger.info("=" * 80)


if __name__ == '__main__':
    # 配置路径
    config_path = '/home/wzj/pan1/BEVNet-CFPR/config/config.yml'

    # 加载配置
    config = yaml.safe_load(open(config_path))

    # 设置日志
    logger.add("/home/wzj/pan1/BEVNet-CFPR/outputs/failure_cases_analysis.log",
               format='{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}',
               encoding='utf-8')

    # 运行分析
    analyze_failure_cases(config)
