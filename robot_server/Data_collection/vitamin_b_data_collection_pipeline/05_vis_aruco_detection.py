#!/usr/bin/env python3
import sys
import os
from pathlib import Path
from omegaconf import OmegaConf
import argparse
import json
import numpy as np
import re
import cv2
from typing import Dict, List, Tuple, Optional
import time

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "data"
sys.path.append(str(PROJECT_ROOT))

from utils.cv_util import (
    parse_aruco_config,
    parse_fisheye_intrinsics,
    convert_fisheye_intrinsics_resolution,
    detect_localize_aruco_tags
)

def find_demos_with_images(demos_dir: Path, task_type: str, single_hand_side: str) -> List[Tuple[Path, str]]:
    """
    找到所有有图像的demo目录
    返回: [(demo_dir, hand_side), ...]
    """
    demo_hand_pairs = []
    for demo_dir in demos_dir.glob('demo_*'):
        if not demo_dir.is_dir():
            continue
        
        if task_type == "single":
            img_folder = demo_dir / f'{single_hand_side}_hand_visual_img'
            if img_folder.exists():
                demo_hand_pairs.append((demo_dir, single_hand_side))
        else:
            for hand in ['left', 'right']:
                img_folder = demo_dir / f'{hand}_hand_visual_img'
                if img_folder.exists():
                    demo_hand_pairs.append((demo_dir, hand))
    
    return sorted(demo_hand_pairs)

def load_image_files(img_folder: Path) -> List[Path]:
    """加载图像文件列表，按数字ID排序"""
    img_files = sorted(img_folder.glob('*.jpg'),
                       key=lambda p: int(re.search(r'(\d+)(?=\.jpg$)', p.name).group(1))
                       if re.search(r'(\d+)(?=\.jpg$)', p.name) else 0)
    return img_files

def draw_aruco_tags(img: np.ndarray, tag_dict: Dict[int, Dict]) -> np.ndarray:
    """
    在图像上绘制检测到的ArUco标记
    """
    img_vis = img.copy()
    
    for tag_id, info in tag_dict.items():
        corners = info['corners'].astype(int)
        
        # 绘制标记边界（绿色）
        color = (0, 255, 0)
        cv2.polylines(img_vis, [corners], True, color, 2, cv2.LINE_AA)
        
        # 绘制四个角点
        for corner in corners:
            cv2.circle(img_vis, tuple(corner), 5, color, -1)
        
        # 在中心显示ID
        center = corners.mean(axis=0).astype(int)
        cv2.putText(img_vis, f'ID:{tag_id}', tuple(center),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        
        # 显示位置信息（tvec）
        tvec = info['tvec']
        text_pos = (center[0], center[1] + 25)
        cv2.putText(img_vis, f'X:{tvec[0]:.3f}', tuple(text_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        text_pos = (center[0], center[1] + 45)
        cv2.putText(img_vis, f'Y:{tvec[1]:.3f}', tuple(text_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        text_pos = (center[0], center[1] + 65)
        cv2.putText(img_vis, f'Z:{tvec[2]:.3f}', tuple(text_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    
    return img_vis

def add_info_text(img: np.ndarray, demo_name: str, hand: str, frame_idx: int, total_frames: int, 
                  num_tags: int, detection_time: float) -> np.ndarray:
    """在图像上添加信息文本"""
    img_vis = img.copy()
    
    # 背景矩形
    h, w = img_vis.shape[:2]
    overlay = img_vis.copy()
    cv2.rectangle(overlay, (10, 10), (500, 170), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img_vis, 0.3, 0, img_vis)
    
    # 文本信息
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    color = (255, 255, 255)
    
    y_offset = 30
    line_height = 25
    
    cv2.putText(img_vis, f'Demo: {demo_name}', (20, y_offset), font, font_scale, color, thickness, cv2.LINE_AA)
    y_offset += line_height
    cv2.putText(img_vis, f'Hand: {hand}', (20, y_offset), font, font_scale, color, thickness, cv2.LINE_AA)
    y_offset += line_height
    cv2.putText(img_vis, f'Frame: {frame_idx+1}/{total_frames}', (20, y_offset), font, font_scale, color, thickness, cv2.LINE_AA)
    y_offset += line_height
    cv2.putText(img_vis, f'Tags: {num_tags}', (20, y_offset), font, font_scale, color, thickness, cv2.LINE_AA)
    y_offset += line_height
    cv2.putText(img_vis, f'Detect Time: {detection_time*1000:.1f}ms', (20, y_offset), font, font_scale, color, thickness, cv2.LINE_AA)
    y_offset += line_height
    cv2.putText(img_vis, '[REAL-TIME DETECTION]', (20, y_offset), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    
    # 控制提示
    y_offset = h - 80
    cv2.putText(img_vis, 'Controls: [a]prev [d]next [s]next demo [q]quit', (20, y_offset), 
               font, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    
    return img_vis

def visualize_detections(cfg_file: str):
    """主可视化函数 - 实时检测"""
    cfg = OmegaConf.load(cfg_file)
    
    config_dir = Path(cfg_file).resolve().parent
    task_name = cfg.task.name
    task_type = cfg.task.type
    single_hand_side = cfg.task.get("single_hand_side", "left")
    
    # 加载相机内参
    intrinsics = cfg.calculate_width.cam_intrinsic_json_path
    if not Path(intrinsics).is_absolute():
        intrinsics = str((config_dir / intrinsics).resolve())
    
    with open(intrinsics, 'r') as f:
        raw_fisheye_intr = parse_fisheye_intrinsics(json.load(f))
    
    # 加载ArUco配置
    aruco_dict_config = cfg.calculate_width.aruco_dict
    marker_size_map_config = cfg.calculate_width.marker_size_map
    aruco_config_dict = {
        'aruco_dict': OmegaConf.to_container(aruco_dict_config, resolve=True),
        'marker_size_map': OmegaConf.to_container(marker_size_map_config, resolve=True)
    }
    aruco_config = parse_aruco_config(aruco_config_dict)
    aruco_dict = aruco_config['aruco_dict']
    marker_size_map = aruco_config['marker_size_map']
    
    demos_dir = DATA_DIR / task_name / "demos"
    demo_hand_pairs = find_demos_with_images(demos_dir, task_type, single_hand_side)
    
    if not demo_hand_pairs:
        print(f"No demos with images found in {demos_dir}")
        return
    
    print(f"Found {len(demo_hand_pairs)} demo-hand pairs with images")
    print("Controls:")
    print("  [a] - Previous frame")
    print("  [d] - Next frame")
    print("  [s] - Next demo")
    print("  [q] - Quit")
    print("\nStarting real-time detection...")
    
    # 当前状态
    demo_idx = 0
    frame_idx = 0
    
    # 加载第一个demo的数据
    demo_dir, hand = demo_hand_pairs[demo_idx]
    img_folder = demo_dir / f'{hand}_hand_visual_img'
    img_files = load_image_files(img_folder)
    total_frames = len(img_files)
    
    if total_frames == 0:
        print(f"No images found in {img_folder}")
        return
    
    # 获取第一张图像的分辨率，用于转换内参
    first_img = cv2.imread(str(img_files[0]))
    if first_img is None:
        print(f"Failed to read first image: {img_files[0]}")
        return
    h, w = first_img.shape[:2]
    in_res = np.array([h, w])[::-1]
    fisheye_intr = convert_fisheye_intrinsics_resolution(
        opencv_intr_dict=raw_fisheye_intr, target_resolution=in_res)
    
    window_name = "ArUco Real-Time Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    print(f"Loaded demo: {demo_dir.name} ({hand}) - {total_frames} frames")
    
    while True:
        # 确保frame_idx在有效范围内
        frame_idx = max(0, min(frame_idx, total_frames - 1))
        
        # 加载当前帧的图像
        img_path = img_files[frame_idx]
        img_bgr = cv2.imread(str(img_path))
        
        if img_bgr is None:
            print(f"Failed to load image: {img_path}")
            frame_idx += 1
            if frame_idx >= total_frames:
                demo_idx += 1
                if demo_idx >= len(demo_hand_pairs):
                    break
                frame_idx = 0
                # 重新加载demo数据
                demo_dir, hand = demo_hand_pairs[demo_idx]
                img_folder = demo_dir / f'{hand}_hand_visual_img'
                img_files = load_image_files(img_folder)
                total_frames = len(img_files)
                if total_frames > 0:
                    first_img = cv2.imread(str(img_files[0]))
                    if first_img is not None:
                        h, w = first_img.shape[:2]
                        in_res = np.array([h, w])[::-1]
                        fisheye_intr = convert_fisheye_intrinsics_resolution(
                            opencv_intr_dict=raw_fisheye_intr, target_resolution=in_res)
                    print(f"Switched to demo: {demo_dir.name} ({hand}) - {total_frames} frames")
            continue
        
        # 转换为RGB进行检测
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 实时检测ArUco标记
        start_time = time.time()
        tag_dict = detect_localize_aruco_tags(
            img=img_rgb,
            aruco_dict=aruco_dict,
            marker_size_map=marker_size_map,
            fisheye_intr_dict=fisheye_intr,
            refine_subpix=True
        )
        detection_time = time.time() - start_time
        
        # 绘制检测结果
        img_vis = draw_aruco_tags(img_bgr, tag_dict)
        
        # 添加信息文本
        img_vis = add_info_text(
            img_vis, 
            demo_dir.name, 
            hand, 
            frame_idx, 
            total_frames,
            len(tag_dict),
            detection_time
        )
        
        # 显示图像
        cv2.imshow(window_name, img_vis)
        
        # 等待键盘输入（30ms延迟，既能实时更新又能响应按键）
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('a'):
            # 上一帧
            frame_idx -= 1
            if frame_idx < 0:
                frame_idx = total_frames - 1
        elif key == ord('d'):
            # 下一帧
            frame_idx += 1
            if frame_idx >= total_frames:
                frame_idx = 0
        elif key == ord('s'):
            # 下一个demo
            demo_idx += 1
            if demo_idx >= len(demo_hand_pairs):
                print("Reached last demo, looping to first...")
                demo_idx = 0
            frame_idx = 0
            
            # 加载新demo的数据
            demo_dir, hand = demo_hand_pairs[demo_idx]
            img_folder = demo_dir / f'{hand}_hand_visual_img'
            img_files = load_image_files(img_folder)
            total_frames = len(img_files)
            
            if total_frames > 0:
                # 更新内参（如果分辨率不同）
                first_img = cv2.imread(str(img_files[0]))
                if first_img is not None:
                    h, w = first_img.shape[:2]
                    in_res = np.array([h, w])[::-1]
                    fisheye_intr = convert_fisheye_intrinsics_resolution(
                        opencv_intr_dict=raw_fisheye_intr, target_resolution=in_res)
                print(f"Switched to demo: {demo_dir.name} ({hand}) - {total_frames} frames")
            else:
                print(f"No images found in {img_folder}")
    
    cv2.destroyAllWindows()
    print("Visualization finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize ArUco detection results")
    parser.add_argument('--cfg', type=str, default='/mnt/disk_1_4T/Chuanyu/codehub/VB-vla/Data_collection/config/VB_task_config.yaml', help='Path to config file')
    args = parser.parse_args()
    visualize_detections(args.cfg)

