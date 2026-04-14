from math import ceil
import sys
import os
from typing import Any
from queue import Empty

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import time
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from datetime import datetime
import threading
from queue import Queue
import json

import click
import cv2
import jax
import numpy as np

import plotly.graph_objects as go

from openpi.training import config as _config
from openpi.policies import policy_config

from utils.precise_sleep import precise_wait
from real_world.bimanual_umi_env import BimanualUmiEnv
from real_world.real_inference_util import get_real_umi_obs_dict, get_real_umi_action

class ObsSaver:
    """异步保存observation数据，不影响eval过程"""
    
    def __init__(self, save_dir: str, data_type: str):
        """
        Args:
            save_dir: 保存目录
            data_type: 数据类型 ('vision' 或 'vitac')
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = Path(save_dir) / f"eval_obs_{timestamp}"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.data_type = data_type
        
        # 使用队列进行异步保存
        self.save_queue = Queue(maxsize=100)  # 限制队列大小，避免内存溢出
        self.save_thread = None
        self.running = False
        self.step_count = 0
        
        print(f"[ObsSaver] Initialized. Save directory: {self.save_dir}")
    
    def start(self):
        """启动保存线程"""
        self.running = True
        self.save_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.save_thread.start()
        print(f"[ObsSaver] Started saving thread")
    
    def stop(self):
        """停止保存线程"""
        self.running = False
        if self.save_thread:
            self.save_thread.join(timeout=5.0)
        print(f"[ObsSaver] Stopped. Total steps saved: {self.step_count}")
    
    def save_obs(self, obs: dict, step_idx: int = None):
        """
        将obs添加到保存队列（非阻塞）
        
        Args:
            obs: observation字典
            step_idx: 步骤索引（如果为None，使用内部计数器）
        """
        if not self.running:
            return
        
        if step_idx is None:
            step_idx = self.step_count
            self.step_count += 1
        
        try:
            # 非阻塞添加，如果队列满了就跳过
            self.save_queue.put_nowait((step_idx, obs))
        except:
            # 队列满了，跳过这次保存
            pass
    
    def _save_worker(self):
        """后台保存线程"""
        while self.running:
            try:
                # 从队列获取数据，超时1秒
                step_idx, obs = self.save_queue.get(timeout=1.0)
                self._save_single_obs(step_idx, obs)
                self.save_queue.task_done()
            except:
                continue
    
    def _numpy_to_json_serializable(self, obj):
        """将numpy数组转换为JSON可序列化的格式"""
        if isinstance(obj, np.ndarray):
            # 转换为列表
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            # numpy标量转换为Python原生类型
            return obj.item()
        elif isinstance(obj, dict):
            return {k: self._numpy_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._numpy_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    def _save_single_obs(self, step_idx: int, obs: dict):
        """保存单个observation - 保存所有obs数据"""
        step_dir = self.save_dir / f"step_{step_idx:06d}"
        step_dir.mkdir(exist_ok=True)
        
        # 保存时间戳为JSON
        if 'timestamp' in obs:
            timestamp_data = self._numpy_to_json_serializable(obs['timestamp'])
            with open(step_dir / "timestamp.json", 'w') as f:
                json.dump(timestamp_data, f, indent=2)
        
        # 遍历所有obs数据并保存
        for key, value in obs.items():
            if key == 'timestamp':
                continue
            
            if isinstance(value, np.ndarray) and len(value.shape) >= 3:
                # 检查是否是图像数据（camera, rgb, tactile相关）
                if 'camera' in key or 'rgb' in key or 'tactile' in key:
                    # 保存为图像文件（取最后一帧）
                    if len(value.shape) == 4:  # (T, H, W, C)
                        img = value[-1]  # 取最后一帧
                    elif len(value.shape) == 3:  # (H, W, C)
                        img = value
                    else:
                        # 不是标准图像格式，保存为JSON
                        json_data = self._numpy_to_json_serializable(value)
                        with open(step_dir / f"{key}.json", 'w') as f:
                            json.dump(json_data, f, indent=2)
                        continue
                    
                    # 转换数据类型和格式
                    if img.dtype == np.float32:
                        img = (img * 255).astype(np.uint8)
                    elif img.max() <= 1.0 and img.dtype in [np.float32, np.float64]:
                        img = (img * 255).astype(np.uint8)
                    
                    # RGB转BGR用于cv2保存
                    # if len(img.shape) == 3 and img.shape[-1] == 3:
                    #     img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    # else:
                    img_path = step_dir / f"{key}.jpg"
                    cv2.imwrite(str(img_path), img)
                else:
                    # 非图像数据，保存为JSON（包括robot pose, gripper width等）
                    json_data = self._numpy_to_json_serializable(value)
                    with open(step_dir / f"{key}.json", 'w') as f:
                        json.dump(json_data, f, indent=2)
            elif isinstance(value, np.ndarray):
                # 低维数据（robot pose, gripper width等），保存为JSON
                json_data = self._numpy_to_json_serializable(value)
                with open(step_dir / f"{key}.json", 'w') as f:
                    json.dump(json_data, f, indent=2)
            else:
                # 其他类型数据，保存为JSON
                json_data = self._numpy_to_json_serializable(value)
                with open(step_dir / f"{key}.json", 'w') as f:
                    json.dump(json_data, f, indent=2)
@click.command()
@click.option('--config', '-c', default=f'pi05_single', help='Config name for policy.')
@click.option('--ckpt-dir', '-i', default='/home/rvsa/codehub/VB-VLA/checkpoints/block/140000', help='Path to checkpoint directory')
@click.option('--data_type', '-dt', default='vision',help='vision, vitac, vitacpc')
@click.option('--language_prompt', '-lp', default='Open the red pot, pick up the blue cylinder on the table and place it into the pot.', help='Language prompt')

@click.option('--save_obs', '-so', default=True, help='Save observation data for verification (saves every step)')
@click.option('--control_frequency', '-f', default=30, type=float, help="Control frequency in Hz.")
@click.option('--controller_frequency', '-cf', default=40, type=float, help="Controller frequency in Hz.")
@click.option('--cam_path', default=['/dev/video0', '/dev/video2'], type=list, help="-")

@click.option('--quest_2_ee_left', default=None, help="-") # eye-hand transform matrix
@click.option('--quest_2_ee_right', default=None, help="-") # eye-hand transform matrix
# @click.option('--width_slope', default=1.053562, type=float, help="-") # transform between gripper width and commanded width
@click.option('--width_slope', default=2.041300, type=float, help="-") # transform between gripper width and commanded width
@click.option('--width_offset', default=0.110115, type=float, help="-") # transform between gripper width and commanded width
@click.option('--vel_max', default=0.4, type=float, help="-") # max velocity of robot

@click.option('--obs_pose_repr', default='relative', help='obs pose representation')
@click.option('--action_pose_repr', default='relative', help='action pose representation')

@click.option('--single_arm_mode', default=True, help='single arm mode')
@click.option('--no_state_obs_mode', default=False, help='no state obs mode')

def main(config,
    ckpt_dir,
    data_type,
    language_prompt,
    save_obs,
    control_frequency, 
    controller_frequency,
    cam_path,
    quest_2_ee_left,
    quest_2_ee_right,
    width_slope,
    width_offset,
    vel_max,
    obs_pose_repr,
    action_pose_repr,
    single_arm_mode,
    no_state_obs_mode
    ):
    # Load default calibration matrices if not provided
    # quest_2_ee_left = np.eye(4)
    # quest_2_ee_right = np.eye(4)

    if quest_2_ee_left is None:
        quest_2_ee_left = np.load("/home/rvsa/codehub/VB-VLA/quest_2_ee_left_hand_fix_quest.npy")
    if quest_2_ee_right is None:
        quest_2_ee_right = np.load("/home/rvsa/codehub/VB-VLA/quest_2_ee_right_hand_fix_quest.npy")

    train_config = _config.get_config(config)
    checkpoint_dir = Path(ckpt_dir)
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = (Path(ROOT_DIR) / checkpoint_dir).resolve()
    policy = policy_config.create_trained_policy(train_config, checkpoint_dir)

    steps_per_inference = int(train_config.model.action_horizon)

    dt = 1/control_frequency
    # ViTaC policy inputs are resized to 224x224 in model transforms.
    obs_res = (224, 224)
    if single_arm_mode:
        cam_path = [cam_path[0]]

    # DEBUG INFO
    if not single_arm_mode:
        sides = ["left", "right"]
    else:
        sides = ["left"]
    paras = ["x", "y", "z", "rx", "ry", "rz", "g"]
    debug_info = dict()
    for side in sides:
        for para in paras:
            debug_info[f"ee_pose_{side}_{para}"] = []
            debug_info[f"target_pose_{side}_{para}"] = []
    debug_info["time"] = []

    print("steps_per_inference:", steps_per_inference)
    # print("data_type:", data_type)
    print("jax backend:", jax.default_backend())
    print("jax devices:", jax.devices())

    with SharedMemoryManager() as shm_manager:
        with BimanualUmiEnv(
                data_type=data_type,
                cam_path=cam_path,
                control_frequency=control_frequency,
                controller_frequency=controller_frequency,
                obs_image_resolution=obs_res,
                obs_float32=True,
                camera_obs_latency=0.101,  # Visual camera latency
                camera_obs_horizon=1,
                robot_obs_horizon=1,
                gripper_obs_horizon=1,
                shm_manager=shm_manager,
                quest_2_ee_left=quest_2_ee_left,
                quest_2_ee_right=quest_2_ee_right,
                width_slope=width_slope,
                width_offset=width_offset,
                vel_max=vel_max,
                single_arm_mode=single_arm_mode,
                ) as env:
            cv2.setNumThreads(2)
            
            print("Waiting for camera")
            time.sleep(3.0)

            print("Warming up policy inference")
            obs = env.get_obs()
            episode_start_pose = list()

            # record initial robot poses
            for robot_id in range(len(cam_path)):
                pose = np.concatenate([
                    obs[f'robot{robot_id}_eef_pos'],
                    obs[f'robot{robot_id}_eef_rot_axis_angle']
                ], axis=-1)[-1]
                episode_start_pose.append(pose)
                
            policy.reset()
            obs_dict = get_real_umi_obs_dict(
                env_obs=obs, shape_meta=None,
                episode_start_pose=episode_start_pose,
                # 由于现在的obs horizon=1，不需要使用批量转化函数，故这个参数实际并未使用
                # obs_pose_repr=obs_pose_repr, 
                data_type=data_type,
                cam_path=cam_path,
                task=language_prompt,
                no_state_obs_mode=no_state_obs_mode
                )
            result = policy.infer(obs_dict)
            raw_action = result['actions']
            print("raw_action:", raw_action[-1])
            assert raw_action.shape[-1] == 10 * len(cam_path)
            action = get_real_umi_action(raw_action, obs, action_pose_repr) # ABS ACTIONS
            assert action.shape[-1] == 7 * len(cam_path)

            print('################################## Ready! ##################################')
            
            obs_saver = None
            if save_obs:
                obs_save_dir = os.path.join(ROOT_DIR, "eval_obs_data")
                obs_saver = ObsSaver(obs_save_dir, data_type)
                obs_saver.start()
                print(f"[ObsSaver] Observation saving enabled. Directory: {obs_saver.save_dir}")
            
            try:

                counter = 0

                while True:
                    try:
                        policy.reset()
                        start_delay = 1.0
                        # t_start = time.monotonic() + start_delay
                        print("Started!")
                        iter_idx = 0

                        while True:
                            eval_t_start = time.monotonic()
                            # 预先计算循环结束的时间点，用于后续的精确等待
                            # t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                            # 获取obs
                            obs = env.get_obs()
                            obs_timestamps = obs['timestamp']
                            # print(f'[main] Obs latency {time.time() - obs_timestamps[-1]}')
                            
                            # 保存obs
                            if obs_saver is not None:
                                obs_saver.save_obs(obs, step_idx=iter_idx)

                            obs_dict = get_real_umi_obs_dict(
                                env_obs=obs, shape_meta=None,
                                obs_pose_repr=obs_pose_repr,
                                episode_start_pose=episode_start_pose,
                                data_type=data_type,
                                cam_path=cam_path,
                                task=language_prompt,
                                no_state_obs_mode=no_state_obs_mode
                                ) # 获取最新一帧的obs_dict
                            
                            # 根据最新一帧的obs_dict进行推理
                            result = policy.infer(obs_dict)

                            # 将输出的相对动作转换成绝对动作
                            raw_action = result['actions']
                            action = get_real_umi_action(raw_action, obs, action_pose_repr) #GET ABS ACTIONS
                            # print('[main] Inference latency:', time.time() - s)
                            this_target_poses = action
                            assert this_target_poses.shape[1] == len(cam_path) * 7
                            
                            # 计算动作执行时间戳
                            # 指定推理出来的每个动作该在什么时间点执行

                            # HACK:补偿延迟时间
                            latency_compensation = (time.time() - obs_timestamps[-1]) * 1.5
                            print(f'[main] Latency compensation: {latency_compensation}')
                            action_timestamps = (np.arange(len(action), dtype=np.float64)
                                ) * dt + obs_timestamps[-1] + latency_compensation

                            env.exec_actions(
                                actions=this_target_poses,
                                timestamps=action_timestamps
                            )
                            
                            # print(f"[main] Submitted {len(this_target_poses)} steps of actions.")
                            
                            iter_idx += steps_per_inference

                            # renew debug info
                            try:
                                debug_info_new = env.get_debug_info()
                                for key in debug_info_new:
                                    debug_info[key] += list[Any](debug_info_new[key])
                            except Empty:
                                pass

                            counter += 1
                            if counter > 1000:
                                break
                            
                            #precise_wait(t_cycle_end)
                            wait_step = 0.001
                            cur_time = time.monotonic() - eval_t_start
                            wait_range = ceil((steps_per_inference*dt + latency_compensation - cur_time) / wait_step)
                            for _ in range(wait_range):
                                time.sleep(wait_step)

                            eval_t_end = time.monotonic()
                            print("[main] Loop T:", eval_t_end - eval_t_start)

                    except KeyboardInterrupt:
                        print("Interrupted!")
                        break

            finally:
                # stop obs saver
                if obs_saver is not None:
                    obs_saver.stop()

                # draw DEBUG INFO
                t = debug_info['time']
                if len(t) > 0:
                    t_offset = t[0]
                    t = [ti - t_offset for ti in t]
                print("Plotting ee vs target")
                logs_dir = "./ee_action_logs"
                if not os.path.exists(logs_dir):
                    os.makedirs(logs_dir)
                image_export_failed = False

                for side in sides:
                    for para in paras:
                        fig = go.Figure()
                        key_ee = f'ee_pose_{side}_{para}'
                        key_target = f'target_pose_{side}_{para}'
                        fig.add_trace(go.Scatter(x=t, y=debug_info[key_ee], mode='lines', name=key_ee, line=dict(color='blue', width=2)))
                        fig.add_trace(go.Scatter(x=t, y=debug_info[key_target], mode='lines', name=key_target, line=dict(color='red', width=2)))
                        fig.update_layout(title='ee vs target', xaxis_title='t', yaxis_title=para)
                        png_path = os.path.join(logs_dir, f"{side+' '+para}.png")
                        try:
                            fig.write_image(png_path)
                        except Exception as e:
                            if not image_export_failed:
                                image_export_failed = True
                                print(
                                    "[Warning] Failed to export debug plots as PNG. "
                                    "This usually means Chrome is not available for kaleido. "
                                    "Install Chrome via `plotly_get_chrome` or system package manager. "
                                    f"Original error: {e}"
                                )

if __name__ == '__main__':
    main()
