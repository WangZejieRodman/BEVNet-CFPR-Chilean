# tools/utils/gen_bev_features_chilean.py
"""
提取Chilean数据集的BEV特征
使用Stage1训练好的Backbone
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


def extract_chilean(net, scan_folder, dst_folder, batch_num=1, coords_range_xyz=[-10, -10, -4, 10, 10, 8]):
    """
    提取Chilean数据集的BEV特征

    Args:
        net: Backbone网络
        scan_folder: 点云文件夹路径
        dst_folder: 特征保存路径
        batch_num: batch大小
        coords_range_xyz: 体素化范围
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
                       desc="Extracting features"):
        batch_files = files[q_index * batch_num:min((q_index + 1) * batch_num, length)]

        with torch.no_grad():
            queries = utils.load_pc_files(batch_files, coords_range_xyz=coords_range_xyz).to(device)
            fea_out = net(queries).cpu().numpy()

        for i in range(len(batch_files)):
            fea_file = os.path.join(dst_folder, os.path.basename(batch_files[i]).replace('.bin', '.npy'))
            np.save(fea_file, fea_out[i])


if __name__ == "__main__":
    config = yaml.safe_load(open('/home/wzj/pan1/BEVNet-CFPR/config/config.yml'))

    root = config["data_root"]["data_root_folder"]
    ckpt = config["stage1_training_config"]["out_folder"] + "/backbone_final.ckpt"
    seqs = config["extractor_config"]["seqs"]
    batch_num = config["extractor_config"]["batch_num"]
    coords_range_xyz = config["stage1_training_config"]["coords_range_xyz"]

    print(f"Loading backbone from {ckpt}")
    net = Backbone(32).to(device)
    checkpoint = torch.load(ckpt)
    state_dict = checkpoint['state_dict']

    # 转换权重形状（如果需要）
    for key in list(state_dict.keys()):
        if 'conv.3.weight' in key:  # SubMConv2d weights
            weight = state_dict[key]
            if len(weight.shape) == 4 and weight.shape[0] < weight.shape[2]:
                # 从 (k, k, in, out) 转换为 (out, k, k, in)
                state_dict[key] = weight.permute(3, 0, 1, 2).contiguous()

    net.load_state_dict(state_dict)
    print("Backbone loaded successfully")

    for seq in seqs:
        print(f"\n{'='*60}")
        print(f"Extracting BEV features for sequence {seq}")
        print(f"{'='*60}")

        scan_folder = os.path.join(root, seq, "pointcloud_20m_10overlap")
        dst_folder = os.path.join(root, seq, "BEV_FEA")

        if not os.path.exists(scan_folder):
            print(f"Warning: Scan folder not found: {scan_folder}")
            continue

        os.makedirs(dst_folder, exist_ok=True)

        extract_chilean(net, scan_folder, dst_folder, batch_num, coords_range_xyz)
        print(f"✓ Sequence {seq} completed")

    print(f"\n{'='*60}")
    print("All BEV features extracted successfully!")
    print(f"{'='*60}")