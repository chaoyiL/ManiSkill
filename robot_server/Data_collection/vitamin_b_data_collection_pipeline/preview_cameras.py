#!/usr/bin/env python3
"""
实时预览相机图像脚本
使用config中的相机参数，不保存图像
按 Q 退出
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from utils.camera_device import V4L2Camera


class CameraConfig:
    def __init__(
        self,
        name: str,
        path: str,
        format: str,
        width: int,
        height: int,
        auto_exposure: int = 1,
        brightness: int = 0,
        gain: int = 100,
        gamma: int = 100,
        exposure: Optional[int] = None,
        auto_white_balance: int = 1,
        wb_temperature: Optional[int] = None,
    ):
        self.name = name
        self.path = path
        self.format = format
        self.width = width
        self.height = height
        self.auto_exposure = auto_exposure
        self.brightness = brightness
        self.gain = gain
        self.gamma = gamma
        self.exposure = exposure
        self.auto_white_balance = auto_white_balance
        self.wb_temperature = wb_temperature


class CameraPreview:
    """实时预览相机图像"""

    def __init__(self, config: OmegaConf):
        self.config = config
        self.cameras: Dict[str, CameraConfig] = self._load_cameras()
        self.camera_instances: Dict[str, V4L2Camera] = {}
        self.running = True

    def _load_cameras(self) -> Dict[str, CameraConfig]:
        """从config加载相机配置"""
        cam_cfg = self.config.recorder.camera
        paths_cfg = self.config.recorder.camera_paths
        task_type = self.config.task.type
        single_side = getattr(self.config.task, "single_hand_side", "left")

        settings = {
            "format": getattr(cam_cfg, "format", "MJPG"),
            "width": cam_cfg.width,
            "height": cam_cfg.height,
            "auto_exposure": getattr(cam_cfg, "auto_exposure", 1),
            "exposure": getattr(cam_cfg, "exposure", None),
            "auto_white_balance": getattr(cam_cfg, "auto_white_balance", 1),
            "wb_temperature": getattr(cam_cfg, "wb_temperature", None),
            "brightness": getattr(cam_cfg, "brightness", 0),
            "gain": getattr(cam_cfg, "gain", 100),
            "gamma": getattr(cam_cfg, "gamma", 100),
        }

        if task_type == "single":
            paths = {f"{single_side}_hand": paths_cfg[f"{single_side}_hand"]}
        else:
            paths = {"left_hand": paths_cfg.left_hand, "right_hand": paths_cfg.right_hand}

        cameras = {}
        for name, path in paths.items():
            cameras[name] = CameraConfig(name=name, path=path, **settings)
            print(f"[CONFIG] {name}: {path} ({settings['width']}x{settings['height']})")

        return cameras

    def _init_camera(self, config: CameraConfig) -> Optional[V4L2Camera]:
        """初始化单个相机"""
        try:
            camera = V4L2Camera(
                device_path=config.path,
                format=config.format,
                width=config.width,
                height=config.height,
            )
            camera.set_white_balance(
                auto=(config.auto_white_balance == 1), 
                temperature=config.wb_temperature
            )
            camera.set_exposure(
                auto=(config.auto_exposure == 1), 
                exposure_time=config.exposure
            )
            camera.set_brightness(brightness=config.brightness)
            camera.set_gain(config.gain)
            camera.set_gamma(gamma=config.gamma)

            print(f"[CAM] {config.name} 初始化成功: {config.width}x{config.height}")
            return camera
        except Exception as e:
            print(f"[ERROR] {config.name} 初始化失败: {e}")
            return None

    def start(self):
        """启动所有相机"""
        for name, config in self.cameras.items():
            camera = self._init_camera(config)
            if camera:
                self.camera_instances[name] = camera

        if not self.camera_instances:
            print("[ERROR] 没有可用的相机!")
            return False
        return True

    def run(self):
        """主循环：实时显示图像"""
        print("\n" + "=" * 50)
        print("  相机实时预览")
        print("=" * 50)
        print("  按 [Q] 退出")
        print("=" * 50 + "\n")

        if not self.start():
            return

        fps_counters = {name: 0 for name in self.camera_instances}
        fps_timers = {name: time.time() for name in self.camera_instances}
        fps_values = {name: 0.0 for name in self.camera_instances}

        try:
            while self.running:
                frames = {}
                
                # 读取所有相机帧
                for name, camera in self.camera_instances.items():
                    ret, frame = camera.read()
                    if ret and frame is not None:
                        frames[name] = frame
                        
                        # 计算FPS
                        fps_counters[name] += 1
                        now = time.time()
                        elapsed = now - fps_timers[name]
                        if elapsed >= 1.0:
                            fps_values[name] = fps_counters[name] / elapsed
                            fps_counters[name] = 0
                            fps_timers[name] = now

                # 显示图像
                for name, frame in frames.items():
                    # 在图像上显示FPS
                    display_frame = frame.copy()
                    fps_text = f"FPS: {fps_values[name]:.1f}"
                    cv2.putText(
                        display_frame, fps_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                    )
                    
                    # 缩放显示（如果图像太大）
                    h, w = display_frame.shape[:2]
                    max_display_width = 3000
                    if w > max_display_width:
                        scale = max_display_width / w
                        display_frame = cv2.resize(
                            display_frame, 
                            (int(w * scale), int(h * scale))
                        )
                    
                    cv2.imshow(name, display_frame)

                # 检查按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\n[EXIT] 用户退出")
                    self.running = False

        except KeyboardInterrupt:
            print("\n[EXIT] Ctrl+C 退出")
        finally:
            self.cleanup()

    def cleanup(self):
        """清理资源"""
        for name, camera in self.camera_instances.items():
            try:
                camera.release()
                print(f"[CAM] {name} 已释放")
            except Exception as e:
                print(f"[ERROR] 释放 {name} 失败: {e}")
        
        cv2.destroyAllWindows()
        print("[EXIT] 预览结束")


def main():
    parser = argparse.ArgumentParser(description="实时预览相机图像")
    parser.add_argument("--cfg", type=str, default="/home/rvsa/codehub/VB-vla/Data_collection/config/VB_task_config.yaml", help="Config文件路径")
    args = parser.parse_args()

    try:
        config = OmegaConf.load(args.cfg)
        preview = CameraPreview(config)
        preview.run()
    except FileNotFoundError:
        print(f"[ERROR] Config文件不存在: {args.cfg}")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

