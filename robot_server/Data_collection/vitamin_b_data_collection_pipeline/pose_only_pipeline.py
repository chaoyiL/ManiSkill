#!/usr/bin/env python3
"""
只处理位姿数据的简化 Pipeline（支持单手/双手模式）。

从 Quest JSON 文件加载轨迹数据，转换坐标系，直接生成只包含 pose 的 zarr.zip。
不处理图像等其他 observation 数据。

使用方法：
    # 双手模式（默认）
    python pose_only_pipeline.py --input /home/rvsa/codehub/ViTaMIn-B/data/_traj_1 /home/rvsa/codehub/ViTaMIn-B/data/_traj_2 /home/rvsa/codehub/ViTaMIn-B/data/_traj_3 --output pose_data.zarr.zip
    
    # 单手模式
    python pose_only_pipeline.py --input /home/rvsa/codehub/ViTaMIn-B/data/_traj_1 --output pose_data.zarr.zip --single-hand left
    
    # 指定最小长度过滤
    python pose_only_pipeline.py --input /path/to/traj1 --output pose_data.zarr.zip --min-length 10

数据结构：
    输入目录结构 (每个 traj 目录):
        traj1/
        ├── quest_poses_*.json  (Quest 位姿数据)
        └── ...
    
    输出 zarr.zip 包含 (双手模式):
        - robot0_eef_pos: (N, 3) 左手末端执行器位置
        - robot0_eef_rot_axis_angle: (N, 3) 左手末端执行器旋转(轴角表示)
        - robot0_demo_start_pose: (N, 6) 左手起始位姿
        - robot0_demo_end_pose: (N, 6) 左手结束位姿
        - robot1_eef_pos: (N, 3) 右手末端执行器位置
        - robot1_eef_rot_axis_angle: (N, 3) 右手末端执行器旋转(轴角表示)
        - robot1_demo_start_pose: (N, 6) 右手起始位姿
        - robot1_demo_end_pose: (N, 6) 右手结束位姿
        - meta/episode_ends: 每个episode的结束索引
"""
import sys
import os
import argparse
import json
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import zarr

# Get project root
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.pose_util import mat_to_pose
from utils.replay_buffer import ReplayBuffer


def compute_rel_transform(pose: np.ndarray) -> tuple:
    """
    将 Unity 坐标系的位姿转换为右手坐标系。
    
    Args:
        pose: 7维数组 [x, y, z, q_x, q_y, q_z, q_w]
        
    Returns:
        rel_pos: 转换后的位置
        rel_quat: 转换后的四元数
    """
    world_frame = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float)
    pose = pose.copy()

    # Unity [x, y, z] -> right-handed [x, z, y]
    world_frame[:3] = np.array([world_frame[0], world_frame[2], world_frame[1]])
    pose[:3] = np.array([pose[0], pose[2], pose[1]])

    Q = np.array([[1, 0, 0],
                  [0, 0, 1],
                  [0, 1, 0.]])

    rot_base = Rotation.from_quat(world_frame[3:]).as_matrix()
    rot = Rotation.from_quat(pose[3:]).as_matrix()
    rel_rot = Rotation.from_matrix(Q @ (rot_base.T @ rot) @ Q.T)
    rel_pos = Rotation.from_matrix(Q @ rot_base.T @ Q.T).apply(pose[:3] - world_frame[:3])
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    return rel_pos, rel_rot.as_quat()


def load_quest_poses_from_directory(dir_path: Path, pattern: str = "quest_poses_*.json") -> list:
    """
    从目录中加载 Quest 位姿数据。
    
    支持两种目录结构:
    1. 直接包含 JSON 文件: dir_path/quest_poses_*.json
    2. 包含 all_trajectory 子目录: dir_path/all_trajectory/quest_poses_*.json
    
    Args:
        dir_path: 包含 JSON 文件的目录
        pattern: JSON 文件匹配模式
        
    Returns:
        合并后的位姿数据列表
    """
    # 首先尝试直接在目录下查找
    json_files = sorted(dir_path.glob(pattern))
    search_dir = dir_path
    
    # 如果没找到，尝试在 all_trajectory 子目录下查找
    if not json_files:
        all_traj_dir = dir_path / "all_trajectory"
        if all_traj_dir.exists():
            json_files = sorted(all_traj_dir.glob(pattern))
            search_dir = all_traj_dir
    
    if not json_files:
        raise FileNotFoundError(f"No {pattern} found in {dir_path} or {dir_path}/all_trajectory")
    
    print(f"  [INFO] Found {len(json_files)} JSON files in {search_dir}")
    
    all_entries = []
    for json_path in json_files:
        try:
            with open(json_path, "r") as f:
                pose_list = json.load(f)
                all_entries.extend(pose_list)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON file {json_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Error reading JSON file {json_path}: {e}")
    
    return all_entries


def extract_trajectory(entries: list, hand: str, no_swap: bool = False) -> dict:
    """
    从 Quest 位姿数据中提取轨迹。
    
    Args:
        entries: Quest 位姿数据列表
        hand: 'left' 或 'right'
        no_swap: 如果为 True，不进行左右手互换
        
    Returns:
        包含位姿数据的字典
    """
    # 确定使用哪个 wrist key
    if no_swap:
        quest_wrist_key = f'{hand}_wrist'
    else:
        # 互换逻辑：left_hand使用right_wrist数据，right_hand使用left_wrist数据
        if hand == 'left':
            quest_wrist_key = 'right_wrist'
        elif hand == 'right':
            quest_wrist_key = 'left_wrist'
        else:
            raise ValueError(f"Unknown hand: {hand}")
    
    timestamps = []
    positions = []
    quaternions = []
    
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
        positions.append(rel_pos)
        quaternions.append(rel_quat)
    
    if not timestamps:
        raise RuntimeError(f"No valid pose data (using {quest_wrist_key})")
    
    # 转换为 numpy 数组
    timestamps = np.array(timestamps, dtype=np.float64)
    positions = np.array(positions, dtype=np.float64)
    quaternions = np.array(quaternions, dtype=np.float64)
    
    # 按时间排序
    order = np.argsort(timestamps)
    timestamps = timestamps[order]
    positions = positions[order]
    quaternions = quaternions[order]
    
    # 去除重复时间戳
    timestamps, unique_idx = np.unique(timestamps, return_index=True)
    positions = positions[unique_idx]
    quaternions = quaternions[unique_idx]
    
    return {
        'timestamps': timestamps,
        'positions': positions,
        'quaternions': quaternions
    }


def compute_pose_from_trajectory(positions: np.ndarray, quaternions: np.ndarray) -> np.ndarray:
    """
    从位置和四元数计算 pose (位置 + 轴角)。
    
    Args:
        positions: (N, 3) 位置数组
        quaternions: (N, 4) 四元数数组
        
    Returns:
        (N, 6) pose 数组 [x, y, z, rx, ry, rz]
    """
    n_frames = len(positions)
    
    # 将四元数转换为旋转矩阵
    rot_mats = Rotation.from_quat(quaternions).as_matrix()
    
    # 构建 4x4 变换矩阵
    pose_mat = np.zeros((n_frames, 4, 4), dtype=np.float32)
    pose_mat[:, 3, 3] = 1
    pose_mat[:, :3, 3] = positions
    pose_mat[:, :3, :3] = rot_mats
    
    # 转换为 pose (位置 + 轴角)
    quest_pose = mat_to_pose(pose_mat)
    
    return quest_pose


def process_single_hand(entries: list, hand: str, no_swap: bool = False) -> Optional[dict]:
    """
    处理单只手的轨迹数据。
    
    Args:
        entries: Quest 位姿数据列表
        hand: 'left' 或 'right'
        no_swap: 是否禁用左右手互换
        
    Returns:
        处理后的数据字典
    """
    # 提取轨迹
    traj_data = extract_trajectory(entries, hand, no_swap)
    
    # 计算 pose
    quest_pose = compute_pose_from_trajectory(
        traj_data['positions'], 
        traj_data['quaternions']
    )
    
    # 提取位置和旋转
    eef_pos = quest_pose[..., :3].astype(np.float32)
    eef_rot = quest_pose[..., 3:].astype(np.float32)
    
    # 计算起始和结束位姿
    demo_start_pose = np.empty_like(quest_pose)
    demo_start_pose[:] = quest_pose[0]
    demo_end_pose = np.empty_like(quest_pose)
    demo_end_pose[:] = quest_pose[-1]
    
    return {
        'eef_pos': eef_pos,
        'eef_rot': eef_rot,
        'demo_start_pose': demo_start_pose.astype(np.float32),
        'demo_end_pose': demo_end_pose.astype(np.float32),
        'timestamps': traj_data['timestamps']
    }


def process_trajectory_dir(traj_dir: Path, bimanual: bool = True, single_hand: str = None,
                          no_swap: bool = False, min_length: int = 1) -> Optional[dict]:
    """
    处理单个轨迹目录（支持单手/双手模式）。
    
    Args:
        traj_dir: 轨迹目录路径
        bimanual: 是否为双手模式
        single_hand: 单手模式时指定的手 ('left' 或 'right')
        no_swap: 是否禁用左右手互换
        min_length: 最小帧数
        
    Returns:
        处理后的数据字典，如果失败则返回 None
    """
    try:
        print(f"\n[Processing] {traj_dir.name}")
        
        # 加载位姿数据
        entries = load_quest_poses_from_directory(traj_dir)
        print(f"  [INFO] Loaded {len(entries)} pose entries")
        
        if bimanual:
            # 双手模式：同时处理 left 和 right
            hands_data = {}
            min_frames = None
            
            for hand in ['left', 'right']:
                try:
                    hand_data = process_single_hand(entries, hand, no_swap)
                    hands_data[hand] = hand_data
                    n_frames = len(hand_data['timestamps'])
                    print(f"  [INFO] {hand} hand: {n_frames} frames")
                    
                    if min_frames is None or n_frames < min_frames:
                        min_frames = n_frames
                except Exception as e:
                    print(f"  [ERROR] Failed to process {hand} hand: {e}")
                    return None
            
            if min_frames is None or min_frames < min_length:
                print(f"  [SKIP] Too short ({min_frames} < {min_length})")
                return None
            
            # 对齐两只手的帧数（取较短的）
            n_frames = min_frames
            for hand in ['left', 'right']:
                for key in ['eef_pos', 'eef_rot', 'demo_start_pose', 'demo_end_pose']:
                    hands_data[hand][key] = hands_data[hand][key][:n_frames]
            
            # 计算 FPS（使用左手的时间戳）
            timestamps = hands_data['left']['timestamps'][:n_frames]
            if n_frames > 1:
                duration = timestamps[-1] - timestamps[0]
                fps = (n_frames - 1) / duration if duration > 0 else 25.0
            else:
                fps = 25.0
            
            print(f"  [OK] Bimanual: {n_frames} frames, {fps:.1f} fps")
            
            return {
                'left': hands_data['left'],
                'right': hands_data['right'],
                'n_frames': n_frames,
                'fps': fps,
                'traj_name': traj_dir.name,
                'bimanual': True
            }
        else:
            # 单手模式
            hand = single_hand or 'left'
            hand_data = process_single_hand(entries, hand, no_swap)
            n_frames = len(hand_data['timestamps'])
            
            if n_frames < min_length:
                print(f"  [SKIP] Too short ({n_frames} < {min_length})")
                return None
            
            # 计算 FPS
            timestamps = hand_data['timestamps']
            if n_frames > 1:
                duration = timestamps[-1] - timestamps[0]
                fps = (n_frames - 1) / duration if duration > 0 else 25.0
            else:
                fps = 25.0
            
            print(f"  [OK] Single hand ({hand}): {n_frames} frames, {fps:.1f} fps")
            
            return {
                'hand': hand,
                'data': hand_data,
                'n_frames': n_frames,
                'fps': fps,
                'traj_name': traj_dir.name,
                'bimanual': False
            }
        
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_pose_only_zarr(
    input_paths: List[Path],
    output_path: Path,
    bimanual: bool = True,
    single_hand: str = None,
    no_swap: bool = False,
    min_length: int = 1,
    use_ee_pose: bool = False,
    tx_quest_2_ee_left_path: Optional[str] = None,
    tx_quest_2_ee_right_path: Optional[str] = None
):
    """
    主函数：处理所有轨迹目录并生成 zarr.zip。
    
    Args:
        input_paths: 轨迹目录列表
        output_path: 输出 zarr.zip 路径
        bimanual: 是否为双手模式
        single_hand: 单手模式时指定的手
        no_swap: 是否禁用左右手互换
        min_length: 最小帧数
        use_ee_pose: 是否使用末端执行器位姿转换
        tx_quest_2_ee_left_path: 左手 Quest 到 EE 的转换矩阵路径
        tx_quest_2_ee_right_path: 右手 Quest 到 EE 的转换矩阵路径
    """
    print("=" * 60)
    print("Pose-Only Pipeline")
    print("=" * 60)
    print(f"Input directories: {len(input_paths)}")
    print(f"Output: {output_path}")
    print(f"Mode: {'Bimanual' if bimanual else f'Single hand ({single_hand})'}")
    print(f"No swap: {no_swap}")
    print(f"Min length: {min_length}")
    print(f"Use EE pose: {use_ee_pose}")
    
    # 检查输出文件是否存在
    if output_path.exists():
        print(f"\n[WARNING] Output file exists, will overwrite: {output_path}")
        output_path.unlink()
    
    # 创建空的 ReplayBuffer
    out_replay_buffer = ReplayBuffer.create_empty_zarr(
        storage=zarr.MemoryStore()
    )
    
    # 加载转换矩阵（如果需要）
    tx_quest_2_ee = {'left': None, 'right': None}
    if use_ee_pose:
        if tx_quest_2_ee_left_path and os.path.exists(tx_quest_2_ee_left_path):
            tx_quest_2_ee['left'] = np.load(tx_quest_2_ee_left_path)
            print(f"Loaded left transformation matrix: {tx_quest_2_ee_left_path}")
        if tx_quest_2_ee_right_path and os.path.exists(tx_quest_2_ee_right_path):
            tx_quest_2_ee['right'] = np.load(tx_quest_2_ee_right_path)
            print(f"Loaded right transformation matrix: {tx_quest_2_ee_right_path}")
    
    # 统计信息
    stats = {
        'total': len(input_paths),
        'processed': 0,
        'skipped': 0,
        'frames': 0,
        'duration': 0.0
    }
    
    # 处理每个轨迹目录
    for traj_path in input_paths:
        traj_dir = Path(traj_path)
        if not traj_dir.is_dir():
            print(f"\n[SKIP] Not a directory: {traj_path}")
            stats['skipped'] += 1
            continue
        
        # 处理轨迹
        result = process_trajectory_dir(
            traj_dir, 
            bimanual=bimanual, 
            single_hand=single_hand,
            no_swap=no_swap, 
            min_length=min_length
        )
        
        if result is None:
            stats['skipped'] += 1
            continue
        
        episode_data = {}
        
        if result['bimanual']:
            # 双手模式：处理 robot0 (left) 和 robot1 (right)
            hand_to_robot = {'left': 0, 'right': 1}
            
            for hand in ['left', 'right']:
                robot_id = hand_to_robot[hand]
                hand_data = result[hand]
                
                eef_pos = hand_data['eef_pos']
                eef_rot = hand_data['eef_rot']
                
                # 应用 EE 位姿转换（如果需要）
                if use_ee_pose and tx_quest_2_ee[hand] is not None:
                    from utils.pose_util import pose_to_mat
                    quest_pose = np.concatenate([eef_pos, eef_rot], axis=-1)
                    eef_pose = mat_to_pose(pose_to_mat(quest_pose) @ np.linalg.inv(tx_quest_2_ee[hand]))
                    eef_pos = eef_pose[..., :3].astype(np.float32)
                    eef_rot = eef_pose[..., 3:].astype(np.float32)
                
                robot_name = f'robot{robot_id}'
                episode_data[f'{robot_name}_eef_pos'] = eef_pos
                episode_data[f'{robot_name}_eef_rot_axis_angle'] = eef_rot
                episode_data[f'{robot_name}_demo_start_pose'] = hand_data['demo_start_pose']
                episode_data[f'{robot_name}_demo_end_pose'] = hand_data['demo_end_pose']
        else:
            # 单手模式：只处理 robot0
            hand = result['hand']
            hand_data = result['data']
            
            eef_pos = hand_data['eef_pos']
            eef_rot = hand_data['eef_rot']
            
            # 应用 EE 位姿转换（如果需要）
            if use_ee_pose and tx_quest_2_ee[hand] is not None:
                from utils.pose_util import pose_to_mat
                quest_pose = np.concatenate([eef_pos, eef_rot], axis=-1)
                eef_pose = mat_to_pose(pose_to_mat(quest_pose) @ np.linalg.inv(tx_quest_2_ee[hand]))
                eef_pos = eef_pose[..., :3].astype(np.float32)
                eef_rot = eef_pose[..., 3:].astype(np.float32)
            
            episode_data['robot0_eef_pos'] = eef_pos
            episode_data['robot0_eef_rot_axis_angle'] = eef_rot
            episode_data['robot0_demo_start_pose'] = hand_data['demo_start_pose']
            episode_data['robot0_demo_end_pose'] = hand_data['demo_end_pose']
        
        out_replay_buffer.add_episode(data=episode_data, compressors=None)
        
        stats['processed'] += 1
        stats['frames'] += result['n_frames']
        stats['duration'] += result['n_frames'] / result['fps']
    
    # 检查是否有有效数据
    if stats['processed'] == 0:
        print("\n[ERROR] No valid trajectories found!")
        return False
    
    # 保存到 zarr.zip
    print(f"\n[INFO] Saving to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with zarr.ZipStore(str(output_path), mode='w') as zip_store:
        out_replay_buffer.save_to_store(store=zip_store)
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Mode: {'Bimanual' if bimanual else 'Single hand'}")
    print(f"Total directories: {stats['total']}")
    print(f"Processed: {stats['processed']}")
    print(f"Skipped: {stats['skipped']}")
    print(f"Total frames: {stats['frames']:,}")
    print(f"Total duration: {stats['duration']:.1f}s ({stats['duration']/60:.1f}min)")
    print(f"Success rate: {stats['processed']/stats['total']*100:.1f}%")
    
    # 打印数据集信息
    print(f"\nDataset Info:")
    print(f"  Episodes: {out_replay_buffer.n_episodes}")
    print(f"  Total steps: {out_replay_buffer.n_steps}")
    for key in sorted(out_replay_buffer.data.keys()):
        data = out_replay_buffer.data[key]
        print(f"  {key}: {data.shape} ({data.dtype})")
    
    # 打印文件大小
    if output_path.exists():
        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"\nOutput file: {output_path}")
        print(f"File size: {file_size:.2f} MB")
    
    print("\n[SUCCESS] Done!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="只处理位姿数据的简化 Pipeline（支持单手/双手模式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 双手模式（默认）- 处理多个轨迹目录
  python pose_only_pipeline.py --input /path/to/traj1 /path/to/traj2 /path/to/traj3 --output pose_data.zarr.zip
  
  # 单手模式
  python pose_only_pipeline.py --input /path/to/traj1 --output pose_data.zarr.zip --single-hand left
  
  # 不进行左右手互换
  python pose_only_pipeline.py --input /path/to/traj1 --output pose_data.zarr.zip --no-swap
  
  # 指定最小长度
  python pose_only_pipeline.py --input /path/to/traj1 --output pose_data.zarr.zip --min-length 10
  
  # 使用 EE 位姿转换
  python pose_only_pipeline.py --input /path/to/traj1 --output pose_data.zarr.zip --use-ee-pose --tx-left /path/to/tx_left.npy --tx-right /path/to/tx_right.npy
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        nargs='+',
        required=True,
        help='输入轨迹目录路径（可以指定多个）'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='输出 zarr.zip 文件路径'
    )
    
    parser.add_argument(
        '--single-hand',
        type=str,
        choices=['left', 'right'],
        default=None,
        help='单手模式：指定要提取的手（left 或 right）。不指定则为双手模式'
    )
    
    parser.add_argument(
        '--no-swap',
        action='store_true',
        help='不进行左右手互换（默认会互换：left_hand 用 right_wrist 数据）'
    )
    
    parser.add_argument(
        '--min-length',
        type=int,
        default=1,
        help='最小帧数，少于此数量的轨迹将被跳过（默认: 1）'
    )
    
    parser.add_argument(
        '--use-ee-pose',
        action='store_true',
        help='使用末端执行器位姿转换'
    )
    
    parser.add_argument(
        '--tx-left',
        type=str,
        default=None,
        help='左手 Quest 到 EE 的转换矩阵路径（.npy 文件）'
    )
    
    parser.add_argument(
        '--tx-right',
        type=str,
        default=None,
        help='右手 Quest 到 EE 的转换矩阵路径（.npy 文件）'
    )
    
    args = parser.parse_args()
    
    # 转换路径
    input_paths = [Path(p) for p in args.input]
    output_path = Path(args.output)
    
    # 确保输出路径以 .zarr.zip 结尾
    if not str(output_path).endswith('.zarr.zip'):
        output_path = Path(str(output_path) + '.zarr.zip')
    
    # 确定是否为双手模式
    bimanual = args.single_hand is None
    
    try:
        success = generate_pose_only_zarr(
            input_paths=input_paths,
            output_path=output_path,
            bimanual=bimanual,
            single_hand=args.single_hand,
            no_swap=args.no_swap,
            min_length=args.min_length,
            use_ee_pose=args.use_ee_pose,
            tx_quest_2_ee_left_path=args.tx_left,
            tx_quest_2_ee_right_path=args.tx_right
        )
        
        if not success:
            sys.exit(1)
            
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

