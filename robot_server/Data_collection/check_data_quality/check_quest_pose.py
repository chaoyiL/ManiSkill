import sys
import os
import pickle
import numpy as np
from pathlib import Path

# Add project root (VB-vla) to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.pose_util import mat_to_pose, pose_to_mat

def check_quest_pose_motion(plan_path):
    """检查dataset_plan.pkl中的quest_pose是否静止"""
    print(f"📂 加载 dataset_plan: {plan_path}")
    
    with open(plan_path, 'rb') as f:
        plan = pickle.load(f)
    
    print(f"✅ 加载了 {len(plan)} 个episodes\n")
    print("=" * 100)
    
    for episode_idx, episode in enumerate(plan):
        demo_name = episode.get('demo_name', f'episode_{episode_idx}')
        grippers = episode['grippers']
        n_frames = episode.get('n_frames', 0)
        
        print(f"\n📊 Episode {episode_idx}: {demo_name}")
        print(f"   帧数: {n_frames}")
        print(f"   手数量: {len(grippers)}")
        print("-" * 100)
        
        for gripper_id, gripper in enumerate(grippers):
            hand_name = ['left', 'right'][gripper_id] if len(grippers) > 1 else 'left'
            quest_pose = gripper['quest_pose']
            
            # 提取位置和旋转
            positions = quest_pose[:, :3]  # (n_frames, 3)
            rotations = quest_pose[:, 3:]  # (n_frames, 3) - axis-angle
            
            # 计算位置变化
            pos_diff = np.diff(positions, axis=0)  # (n_frames-1, 3)
            pos_magnitude = np.linalg.norm(pos_diff, axis=1)  # (n_frames-1,)
            max_pos_change = np.max(pos_magnitude)
            mean_pos_change = np.mean(pos_magnitude)
            
            # 计算旋转变化
            rot_diff = np.diff(rotations, axis=0)  # (n_frames-1, 3)
            rot_magnitude = np.linalg.norm(rot_diff, axis=1)  # (n_frames-1,)
            max_rot_change = np.max(rot_magnitude)
            mean_rot_change = np.mean(rot_magnitude)
            
            # 检查是否静止（阈值：位置变化 < 1mm，旋转变化 < 0.01 rad）
            pos_static_threshold = 0.001  # 1mm
            rot_static_threshold = 0.01   # ~0.57度
            
            is_pos_static = max_pos_change < pos_static_threshold
            is_rot_static = max_rot_change < rot_static_threshold
            is_static = is_pos_static and is_rot_static
            
            print(f"\n  🤖 {hand_name.upper()} Hand (Gripper {gripper_id}):")
            print(f"     Quest Pose Shape: {quest_pose.shape}")
            print(f"     位置范围:")
            print(f"       X: [{positions[:, 0].min():8.6f}, {positions[:, 0].max():8.6f}] (变化: {positions[:, 0].max() - positions[:, 0].min():.6f})")
            print(f"       Y: [{positions[:, 1].min():8.6f}, {positions[:, 1].max():8.6f}] (变化: {positions[:, 1].max() - positions[:, 1].min():.6f})")
            print(f"       Z: [{positions[:, 2].min():8.6f}, {positions[:, 2].max():8.6f}] (变化: {positions[:, 2].max() - positions[:, 2].min():.6f})")
            print(f"     位置变化统计:")
            print(f"       最大变化: {max_pos_change:.6f} m")
            print(f"       平均变化: {mean_pos_change:.6f} m")
            print(f"       标准差: {np.std(pos_magnitude):.6f} m")
            
            print(f"     旋转范围:")
            print(f"       X轴: [{rotations[:, 0].min():8.6f}, {rotations[:, 0].max():8.6f}] (变化: {rotations[:, 0].max() - rotations[:, 0].min():.6f})")
            print(f"       Y轴: [{rotations[:, 1].min():8.6f}, {rotations[:, 1].max():8.6f}] (变化: {rotations[:, 1].max() - rotations[:, 1].min():.6f})")
            print(f"       Z轴: [{rotations[:, 2].min():8.6f}, {rotations[:, 2].max():8.6f}] (变化: {rotations[:, 2].max() - rotations[:, 2].min():.6f})")
            print(f"     旋转变化统计:")
            print(f"       最大变化: {max_rot_change:.6f} rad ({np.degrees(max_rot_change):.3f}°)")
            print(f"       平均变化: {mean_rot_change:.6f} rad ({np.degrees(mean_rot_change):.3f}°)")
            print(f"       标准差: {np.std(rot_magnitude):.6f} rad")
            
            # 显示前几帧和后几帧的pose
            print(f"\n     前3帧 Quest Pose:")
            for i in range(min(3, len(quest_pose))):
                print(f"       帧 {i}: Pos=[{positions[i, 0]:8.6f}, {positions[i, 1]:8.6f}, {positions[i, 2]:8.6f}], "
                      f"Rot=[{rotations[i, 0]:8.6f}, {rotations[i, 1]:8.6f}, {rotations[i, 2]:8.6f}]")
            
            if len(quest_pose) > 3:
                print(f"     后3帧 Quest Pose:")
                for i in range(max(0, len(quest_pose)-3), len(quest_pose)):
                    print(f"       帧 {i}: Pos=[{positions[i, 0]:8.6f}, {positions[i, 1]:8.6f}, {positions[i, 2]:8.6f}], "
                          f"Rot=[{rotations[i, 0]:8.6f}, {rotations[i, 1]:8.6f}, {rotations[i, 2]:8.6f}]")
            
            # 判断结果
            print(f"\n     📌 判断结果:")
            if is_static:
                print(f"       ⚠️  QUEST POSE 基本静止！")
                print(f"         位置最大变化: {max_pos_change:.6f} m (< {pos_static_threshold} m)")
                print(f"         旋转最大变化: {max_rot_change:.6f} rad (< {rot_static_threshold} rad)")
            else:
                if not is_pos_static:
                    print(f"       ✅ 位置有变化: 最大 {max_pos_change:.6f} m")
                else:
                    print(f"       ⚠️  位置静止: 最大变化 {max_pos_change:.6f} m")
                
                if not is_rot_static:
                    print(f"       ✅ 旋转有变化: 最大 {max_rot_change:.6f} rad ({np.degrees(max_rot_change):.3f}°)")
                else:
                    print(f"       ⚠️  旋转静止: 最大变化 {max_rot_change:.6f} rad")
        
        print()
    
    print("=" * 100)
    print("\n📈 总结:")
    
    # 统计所有episode的情况
    total_static_left = 0
    total_static_right = 0
    total_episodes = len(plan)
    
    for episode in plan:
        grippers = episode['grippers']
        for gripper_id, gripper in enumerate(grippers):
            quest_pose = gripper['quest_pose']
            positions = quest_pose[:, :3]
            rotations = quest_pose[:, 3:]
            
            pos_diff = np.diff(positions, axis=0)
            rot_diff = np.diff(rotations, axis=0)
            max_pos_change = np.max(np.linalg.norm(pos_diff, axis=1))
            max_rot_change = np.max(np.linalg.norm(rot_diff, axis=1))
            
            is_static = (max_pos_change < 0.001) and (max_rot_change < 0.01)
            
            if gripper_id == 0:
                if is_static:
                    total_static_left += 1
            else:
                if is_static:
                    total_static_right += 1
    
    print(f"   左手静止的episodes: {total_static_left}/{total_episodes}")
    if total_static_right > 0:
        print(f"   右手静止的episodes: {total_static_right}/{total_episodes}")
    
    if total_static_left == total_episodes:
        print(f"\n   ⚠️  警告: 所有episode的左手quest pose都是静止的！")
    elif total_static_left > 0:
        print(f"\n   ⚠️  警告: {total_static_left}个episode的左手quest pose是静止的！")
    else:
        print(f"\n   ✅ 所有episode的左手quest pose都有运动")


if __name__ == "__main__":
    plan_path = Path("/mnt/disk_1_4T/Chuanyu/codehub/VB-vla/data/_123_1/dataset_plan.pkl")
    
    if not plan_path.exists():
        print(f"❌ 文件不存在: {plan_path}")
        sys.exit(1)
    
    check_quest_pose_motion(plan_path)

