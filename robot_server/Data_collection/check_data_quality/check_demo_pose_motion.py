#!/usr/bin/env python3
"""
检查每个demo内的pose_data是否运动
"""
import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

def check_trajectory_motion(traj_file):
    """检查轨迹文件是否有运动"""
    try:
        data = np.genfromtxt(traj_file, delimiter=",", names=True)
        if data.size == 0:
            return None, "Empty file"
        
        # 提取位置和旋转
        positions = np.column_stack([
            np.asarray(data["x"], dtype=np.float64),
            np.asarray(data["y"], dtype=np.float64),
            np.asarray(data["z"], dtype=np.float64),
        ])
        
        quat = np.column_stack([
            np.asarray(data["q_x"], dtype=np.float64),
            np.asarray(data["q_y"], dtype=np.float64),
            np.asarray(data["q_z"], dtype=np.float64),
            np.asarray(data["q_w"], dtype=np.float64),
        ])
        
        # 计算位置变化
        pos_diff = np.diff(positions, axis=0)
        pos_magnitude = np.linalg.norm(pos_diff, axis=1)
        max_pos_change = np.max(pos_magnitude)
        mean_pos_change = np.mean(pos_magnitude)
        
        # 计算旋转变化（使用四元数的角度差）
        from scipy.spatial.transform import Rotation
        rot = Rotation.from_quat(quat)
        rot_diff = rot[:-1].inv() * rot[1:]
        rot_angles = rot_diff.magnitude()
        max_rot_change = np.max(rot_angles)
        mean_rot_change = np.mean(rot_angles)
        
        # 位置范围
        pos_range = {
            'x': (positions[:, 0].min(), positions[:, 0].max(), positions[:, 0].max() - positions[:, 0].min()),
            'y': (positions[:, 1].min(), positions[:, 1].max(), positions[:, 1].max() - positions[:, 1].min()),
            'z': (positions[:, 2].min(), positions[:, 2].max(), positions[:, 2].max() - positions[:, 2].min()),
        }
        
        return {
            'n_frames': len(positions),
            'max_pos_change': max_pos_change,
            'mean_pos_change': mean_pos_change,
            'max_rot_change': max_rot_change,
            'mean_rot_change': mean_rot_change,
            'pos_range': pos_range,
            'is_moving': max_pos_change > 0.001 or max_rot_change > 0.01,  # 阈值：1mm位置变化或0.01rad旋转变化
        }, None
    except Exception as e:
        return None, str(e)

def check_all_demos():
    """检查所有demo的pose_data"""
    demos_dir = Path("/mnt/disk_1_4T/Chuanyu/codehub/VB-vla/data/_123_1/demos")
    
    if not demos_dir.exists():
        print(f"❌ Demos目录不存在: {demos_dir}")
        return
    
    demo_dirs = sorted([d for d in demos_dir.glob('demo_*') if d.is_dir()])
    
    if not demo_dirs:
        print(f"❌ 未找到demo目录")
        return
    
    print(f"🔍 检查 {len(demo_dirs)} 个demo的pose_data运动情况\n")
    print("=" * 100)
    
    for demo_dir in demo_dirs:
        print(f"\n📂 Demo: {demo_dir.name}")
        print("-" * 100)
        
        pose_data_dir = demo_dir / 'pose_data'
        if not pose_data_dir.exists():
            print(f"  ⚠️  pose_data目录不存在")
            continue
        
        # 查找所有轨迹文件
        traj_files = sorted(pose_data_dir.glob('*_hand_trajectory.csv'))
        
        if not traj_files:
            print(f"  ⚠️  未找到轨迹文件")
            continue
        
        for traj_file in traj_files:
            hand_name = traj_file.stem.replace('_hand_trajectory', '')
            print(f"\n  🤖 {hand_name.upper()} Hand: {traj_file.name}")
            
            result, error = check_trajectory_motion(traj_file)
            
            if error:
                print(f"    ❌ 错误: {error}")
                continue
            
            if result is None:
                print(f"    ❌ 无法读取数据")
                continue
            
            print(f"    帧数: {result['n_frames']}")
            print(f"    位置范围:")
            print(f"      X: [{result['pos_range']['x'][0]:8.6f}, {result['pos_range']['x'][1]:8.6f}] (变化: {result['pos_range']['x'][2]:.6f} m)")
            print(f"      Y: [{result['pos_range']['y'][0]:8.6f}, {result['pos_range']['y'][1]:8.6f}] (变化: {result['pos_range']['y'][2]:.6f} m)")
            print(f"      Z: [{result['pos_range']['z'][0]:8.6f}, {result['pos_range']['z'][1]:8.6f}] (变化: {result['pos_range']['z'][2]:.6f} m)")
            print(f"    位置变化:")
            print(f"      最大变化: {result['max_pos_change']:.6f} m")
            print(f"      平均变化: {result['mean_pos_change']:.6f} m")
            print(f"    旋转变化:")
            print(f"      最大变化: {result['max_rot_change']:.6f} rad ({np.degrees(result['max_rot_change']):.3f}°)")
            print(f"      平均变化: {result['mean_rot_change']:.6f} rad ({np.degrees(result['mean_rot_change']):.3f}°)")
            
            if result['is_moving']:
                print(f"    ✅ 有运动")
            else:
                print(f"    ⚠️  基本静止（位置变化 < 1mm 且旋转变化 < 0.01 rad）")
    
    print("\n" + "=" * 100)

if __name__ == "__main__":
    check_all_demos()

