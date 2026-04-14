#!/usr/bin/env python3
"""
从 Quest JSON 文件加载位姿数据并转换坐标系。

功能：
1. 从 quest_poses_*.json 文件加载 Quest 手柄位姿数据
2. 将 Unity 左手坐标系转换为右手坐标系
3. 输出转换后的轨迹 CSV 文件

使用方法：
    python convert_quest_poses.py --input /path/to/all_trajectory --output /path/to/output.csv --hand left
    python convert_quest_poses.py --input /path/to/quest_poses.json --output /path/to/output.csv --hand right
"""
import sys
import argparse
import json
import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def compute_rel_transform(pose: np.ndarray) -> tuple:
    """
    将 Unity 坐标系的位姿转换为右手坐标系。
    
    Unity 使用左手坐标系：
    - X: 右
    - Y: 上  
    - Z: 前
    
    转换后的右手坐标系：
    - X: 右
    - Y: 前 (原 Unity Z)
    - Z: 上 (原 Unity Y)
    
    Args:
        pose: 7维数组 [x, y, z, q_x, q_y, q_z, q_w]，Unity坐标系下的位姿
        
    Returns:
        rel_pos: 3维数组，转换后的位置 [x, y, z]
        rel_quat: 4维数组，转换后的四元数 [q_x, q_y, q_z, q_w]
    """
    world_frame = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float)
    pose = pose.copy()

    # Unity [x, y, z] -> right-handed [x, z, y]
    # 交换y和z轴：Unity的y（上）变成右手系的z（上），Unity的z（前）变成右手系的y（前）
    world_frame[:3] = np.array([world_frame[0], world_frame[2], world_frame[1]])
    pose[:3] = np.array([pose[0], pose[2], pose[1]])

    # 旋转矩阵Q用于交换y和z轴
    # Q将Unity坐标系中的旋转矩阵转换到右手系
    Q = np.array([[1, 0, 0],
                  [0, 0, 1],
                  [0, 1, 0.]])

    rot_base = Rotation.from_quat(world_frame[3:]).as_matrix()
    rot = Rotation.from_quat(pose[3:]).as_matrix()
    # 将旋转矩阵从Unity坐标系转换到右手系：Q @ rot @ Q.T
    rel_rot = Rotation.from_matrix(Q @ (rot_base.T @ rot) @ Q.T)
    # 将位置向量从Unity坐标系转换到右手系
    rel_pos = Rotation.from_matrix(Q @ rot_base.T @ Q.T).apply(pose[:3] - world_frame[:3])
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    return rel_pos, rel_rot.as_quat()


def load_quest_poses_from_json(json_path: Path) -> list:
    """
    从单个 JSON 文件加载 Quest 位姿数据。
    
    Args:
        json_path: JSON 文件路径
        
    Returns:
        位姿数据列表
    """
    with open(json_path, "r") as f:
        pose_list = json.load(f)
    return pose_list


def load_quest_poses_from_directory(dir_path: Path) -> list:
    """
    从目录中的所有 quest_poses_*.json 文件加载位姿数据。
    
    Args:
        dir_path: 包含 JSON 文件的目录路径
        
    Returns:
        合并后的位姿数据列表
    """
    json_files = sorted(dir_path.glob("quest_poses_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No quest_poses_*.json found in {dir_path}")
    
    print(f"[INFO] Found {len(json_files)} JSON files")
    
    all_entries = []
    for json_path in json_files:
        try:
            pose_list = load_quest_poses_from_json(json_path)
            all_entries.extend(pose_list)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON file {json_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error reading JSON file {json_path}: {e}")
    
    return all_entries


def extract_wrist_poses(entries: list, hand: str) -> tuple:
    """
    从 Quest 位姿数据中提取手腕位姿并转换坐标系。
    
    注意：由于左手设备使用右quest追踪，右手设备用左quest追踪，
    所以需要互换：left_hand -> right_wrist, right_hand -> left_wrist
    
    Args:
        entries: Quest 位姿数据列表
        hand: 'left' 或 'right'，指定要提取的手
        
    Returns:
        timestamps: 时间戳列表
        poses: 转换后的位姿列表，每个元素为 [x, y, z, q_x, q_y, q_z, q_w]
    """
    # 互换逻辑：left_hand使用right_wrist数据，right_hand使用left_wrist数据
    if hand == 'left':
        quest_wrist_key = 'right_wrist'  # 左手设备用右quest追踪
    elif hand == 'right':
        quest_wrist_key = 'left_wrist'   # 右手设备用左quest追踪
    else:
        raise ValueError(f"Unknown hand: {hand}, expected 'left' or 'right'")
    
    print(f"[INFO] Extracting {hand} hand poses (using {quest_wrist_key} data)...")
    
    timestamps = []
    poses = []
    
    for entry in entries:
        if quest_wrist_key not in entry:
            continue
        
        wrist = entry[quest_wrist_key]
        pos = wrist.get("position", {})
        rot = wrist.get("rotation", {})
        
        # 提取位置
        if isinstance(pos, dict):
            x = float(pos.get("x", 0.0))
            y = float(pos.get("y", 0.0))
            z = float(pos.get("z", 0.0))
        else:
            try:
                x, y, z = map(float, pos[:3])
            except Exception:
                x = y = z = 0.0
        
        # 提取旋转
        if isinstance(rot, dict):
            q_x = float(rot.get("x", 0.0))
            q_y = float(rot.get("y", 0.0))
            q_z = float(rot.get("z", 0.0))
            q_w = float(rot.get("w", 1.0))
        else:
            try:
                q_x, q_y, q_z, q_w = map(float, rot[:4])
            except Exception:
                q_x = q_y = q_z = 0.0
                q_w = 1.0
        
        # 转换坐标系
        rel_pos, rel_quat = compute_rel_transform(
            np.array([x, y, z, q_x, q_y, q_z, q_w], dtype=float)
        )
        
        ts = entry.get("timestamp_unix", entry.get("timestamp", 0.0))
        timestamps.append(ts)
        poses.append([rel_pos[0], rel_pos[1], rel_pos[2], 
                      rel_quat[0], rel_quat[1], rel_quat[2], rel_quat[3]])
    
    if not timestamps:
        raise RuntimeError(f"No valid pose data for hand '{hand}' (using {quest_wrist_key})")
    
    return timestamps, poses


def save_trajectory_csv(output_path: Path, timestamps: list, poses: list):
    """
    将轨迹数据保存为 CSV 文件。
    
    Args:
        output_path: 输出 CSV 文件路径
        timestamps: 时间戳列表
        poses: 位姿列表，每个元素为 [x, y, z, q_x, q_y, q_z, q_w]
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data_array = np.column_stack([timestamps, poses])
    header = "timestamp,x,y,z,q_x,q_y,q_z,q_w"
    
    np.savetxt(
        output_path,
        data_array,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.9f"
    )
    
    print(f"[INFO] Saved trajectory CSV: {output_path}")
    print(f"[INFO] Total rows: {len(timestamps)}")


def convert_quest_poses(input_path: Path, output_path: Path, hand: str, no_swap: bool = False):
    """
    主函数：从 JSON 加载 Quest 位姿数据，转换坐标系，保存为 CSV。
    
    Args:
        input_path: 输入路径（JSON 文件或包含 JSON 文件的目录）
        output_path: 输出 CSV 文件路径
        hand: 'left' 或 'right'
        no_swap: 如果为 True，不进行左右手互换（直接使用指定的 hand）
    """
    # 加载数据
    if input_path.is_file():
        print(f"[INFO] Loading from single file: {input_path}")
        entries = load_quest_poses_from_json(input_path)
    elif input_path.is_dir():
        print(f"[INFO] Loading from directory: {input_path}")
        entries = load_quest_poses_from_directory(input_path)
    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")
    
    print(f"[INFO] Loaded {len(entries)} pose entries")
    
    # 如果不互换，临时修改 hand 逻辑
    if no_swap:
        # 直接使用指定的 wrist key
        quest_wrist_key = f'{hand}_wrist'
        print(f"[INFO] No swap mode: directly using {quest_wrist_key}")
        
        timestamps = []
        poses = []
        
        for entry in entries:
            if quest_wrist_key not in entry:
                continue
            
            wrist = entry[quest_wrist_key]
            pos = wrist.get("position", {})
            rot = wrist.get("rotation", {})
            
            if isinstance(pos, dict):
                x = float(pos.get("x", 0.0))
                y = float(pos.get("y", 0.0))
                z = float(pos.get("z", 0.0))
            else:
                x, y, z = map(float, pos[:3]) if len(pos) >= 3 else (0.0, 0.0, 0.0)
            
            if isinstance(rot, dict):
                q_x = float(rot.get("x", 0.0))
                q_y = float(rot.get("y", 0.0))
                q_z = float(rot.get("z", 0.0))
                q_w = float(rot.get("w", 1.0))
            else:
                q_x, q_y, q_z, q_w = map(float, rot[:4]) if len(rot) >= 4 else (0.0, 0.0, 0.0, 1.0)
            
            rel_pos, rel_quat = compute_rel_transform(
                np.array([x, y, z, q_x, q_y, q_z, q_w], dtype=float)
            )
            
            ts = entry.get("timestamp_unix", entry.get("timestamp", 0.0))
            timestamps.append(ts)
            poses.append([rel_pos[0], rel_pos[1], rel_pos[2], 
                          rel_quat[0], rel_quat[1], rel_quat[2], rel_quat[3]])
        
        if not timestamps:
            raise RuntimeError(f"No valid pose data for {quest_wrist_key}")
    else:
        # 使用默认的互换逻辑
        timestamps, poses = extract_wrist_poses(entries, hand)
    
    # 保存结果
    save_trajectory_csv(output_path, timestamps, poses)
    
    return timestamps, poses


def main():
    parser = argparse.ArgumentParser(
        description="从 Quest JSON 文件加载位姿数据并转换坐标系",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 从目录加载并转换左手数据
  python convert_quest_poses.py --input /path/to/all_trajectory --output left_trajectory.csv --hand left
  
  # 从单个 JSON 文件加载
  python convert_quest_poses.py --input quest_poses_001.json --output trajectory.csv --hand right
  
  # 不进行左右手互换
  python convert_quest_poses.py --input /path/to/data --output output.csv --hand left --no-swap
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入路径：JSON 文件或包含 quest_poses_*.json 的目录'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='输出 CSV 文件路径'
    )
    
    parser.add_argument(
        '--hand',
        type=str,
        choices=['left', 'right'],
        required=True,
        help='要提取的手：left 或 right'
    )
    
    parser.add_argument(
        '--no-swap',
        action='store_true',
        help='不进行左右手互换（默认会互换：left_hand 用 right_wrist 数据）'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    try:
        convert_quest_poses(input_path, output_path, args.hand, args.no_swap)
        print("\n[SUCCESS] 转换完成！")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

