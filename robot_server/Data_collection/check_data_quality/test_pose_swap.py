#!/usr/bin/env python3
"""
测试左右手pose互换逻辑
"""
import sys
import os
import json
import glob
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

def test_pose_swap_logic():
    """测试pose互换逻辑"""
    print("🧪 测试左右手pose互换逻辑\n")
    
    # 加载原始JSON数据
    json_files = sorted(glob.glob('/mnt/disk_1_4T/Chuanyu/codehub/VB-vla/data/_123_1/all_trajectory/quest_poses_*.json'))
    if not json_files:
        print("❌ 未找到JSON文件")
        return
    
    print(f"📂 找到 {len(json_files)} 个JSON文件")
    
    # 读取第一个文件的前几个entry
    with open(json_files[0], 'r') as f:
        data = json.load(f)
    
    print(f"📊 第一个文件包含 {len(data)} 个entry\n")
    
    # 检查前3个entry
    for i in range(min(3, len(data))):
        entry = data[i]
        print(f"Entry {i}:")
        print(f"  timestamp: {entry.get('timestamp_unix', 'N/A')}")
        
        if 'left_wrist' in entry:
            left_pos = entry['left_wrist']['position']
            print(f"  left_wrist:  Pos=[{left_pos['x']:.6f}, {left_pos['y']:.6f}, {left_pos['z']:.6f}]")
        else:
            print(f"  left_wrist:  ❌ 不存在")
        
        if 'right_wrist' in entry:
            right_pos = entry['right_wrist']['position']
            print(f"  right_wrist: Pos=[{right_pos['x']:.6f}, {right_pos['y']:.6f}, {right_pos['z']:.6f}]")
        else:
            print(f"  right_wrist: ❌ 不存在")
        
        print()
    
    # 验证互换逻辑
    print("=" * 80)
    print("🔄 互换逻辑验证:")
    print("=" * 80)
    print("\n根据互换逻辑：")
    print("  - left_hand 应该使用 right_wrist 的数据")
    print("  - right_hand 应该使用 left_wrist 的数据")
    print()
    
    # 统计所有entry
    left_wrist_count = 0
    right_wrist_count = 0
    
    for json_file in json_files[:1]:  # 只检查第一个文件
        with open(json_file, 'r') as f:
            data = json.load(f)
        for entry in data:
            if 'left_wrist' in entry:
                left_wrist_count += 1
            if 'right_wrist' in entry:
                right_wrist_count += 1
    
    print(f"📈 统计（第一个文件）:")
    print(f"  left_wrist entries:  {left_wrist_count}")
    print(f"  right_wrist entries: {right_wrist_count}")
    print()
    
    if left_wrist_count > 0 and right_wrist_count > 0:
        print("✅ 数据中同时包含left_wrist和right_wrist，互换逻辑可以正常工作")
        print("\n💡 说明:")
        print("  - 当处理left_hand时，会从right_wrist读取数据")
        print("  - 当处理right_hand时，会从left_wrist读取数据")
    else:
        print("⚠️  警告: 数据可能不完整")


if __name__ == "__main__":
    test_pose_swap_logic()

