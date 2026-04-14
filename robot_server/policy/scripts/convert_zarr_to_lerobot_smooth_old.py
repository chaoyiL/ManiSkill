#!/usr/bin/env python3
"""
将 ViTaMin-B Zarr 格式数据转换为 LeRobot 格式
优化版：流式写入 + JPEG 压缩 IPC + 分批处理，大幅降低内存占用
"""

import argparse
import sys
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from collections import OrderedDict
import numpy as np
import cv2
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

import zarr
from zarr.storage import ZipStore
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from openpi.training import config as training_config
from utils.imagecodecs_numcodecs import register_codecs
register_codecs()
from utils.pose_util import pose_to_mat, mat_to_pose


# ============================================================================
# 进程池全局变量
# ============================================================================
_PROCESS_DATA = None


def _worker_init(zarr_path):
    """每个 worker 进程独立打开 ZipStore"""
    global _PROCESS_DATA
    store = ZipStore(zarr_path, mode="r")
    root = zarr.open_group(store=store, mode="r")
    _PROCESS_DATA = root["data"]
    register_codecs()


# ============================================================================
# 🔥 优化1: 图像压缩后传输，减少 IPC pickle 体积 ~10x
# ============================================================================
def _encode_image_jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    """将 numpy 图像编码为 JPEG bytes，大幅减少进程间传输量"""
    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b''


def _decode_image_jpeg(data: bytes) -> np.ndarray:
    """将 JPEG bytes 解码回 numpy 图像"""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else np.zeros((224, 224, 3), dtype=np.uint8)


def _process_image(image_data, target_h=224, target_w=224) -> np.ndarray:
    """处理原始图像数据为标准格式"""
    if isinstance(image_data, bytes):
        img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
    elif hasattr(image_data, "shape"):
        img = image_data
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = None

    if img is None:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    if img.dtype != np.uint8:
        if img.dtype in (np.float32, np.float64):
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else np.clip(img, 0, 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    try:
        return cv2.resize(img, (target_w, target_h))
    except Exception:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)


# ============================================================================
# 位姿序列高斯平滑（SE(3) 安全）
# ============================================================================
def _smooth_pose_matrices(mats, sigma=2.0):
    """
    对 SE(3) 位姿矩阵序列进行高斯平滑。
    平移用高斯滤波；旋转逐元素滤波后 SVD 重新正交化以保证 SO(3) 合法性。

    Args:
        mats:  list[np.ndarray(4,4)]  长度为 T 的位姿序列
        sigma: 高斯核标准差（单位：帧），越大越平滑；<=0 时不做平滑
    Returns:
        list[np.ndarray(4,4)]  平滑后的位姿序列，长度不变
    """
    if sigma <= 0 or len(mats) < 3:
        return mats

    n = len(mats)
    mat_arr = np.stack(mats)  # (n, 4, 4)

    trans = mat_arr[:, :3, 3]  # (n, 3)
    trans_smooth = gaussian_filter1d(trans, sigma=sigma, axis=0)

    rot = mat_arr[:, :3, :3]  # (n, 3, 3)
    rot_smooth = gaussian_filter1d(rot, sigma=sigma, axis=0)

    smoothed = []
    for t in range(n):
        U, _, Vt = np.linalg.svd(rot_smooth[t])
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1
            R = U @ Vt
        mat = np.eye(4)
        mat[:3, :3] = R
        mat[:3, 3] = trans_smooth[t]
        smoothed.append(mat)

    return smoothed


# ============================================================================
# 🔥 优化2: Worker 只返回轻量数据（JPEG bytes + float arrays）
# ============================================================================
IMAGE_KEYS = [
    ("camera0_rgb",           "observation.images.camera0"),
    ("camera1_rgb",           "observation.images.camera1"),
    ("camera0_left_tactile",  "observation.images.tactile_left_0"),
    ("camera0_right_tactile", "observation.images.tactile_right_0"),
    ("camera1_left_tactile",  "observation.images.tactile_left_1"),
    ("camera1_right_tactile", "observation.images.tactile_right_1"),
]


def _build_episode_worker(args):
    """
    Worker: 批量读取一个 episode，构建帧数据。
    图像以 JPEG bytes 返回以减少 IPC 开销。
    """
    (ep_idx, start_idx, stop_idx, num_robots, state_dim, action_dim,
     language_instruction, smooth_sigma, no_state) = args

    global _PROCESS_DATA
    ep_len = stop_idx - start_idx

    # ------------------------------------------------------------------
    # 批量读取（一次 I/O）
    # ------------------------------------------------------------------
    ed = {}  # episode_data
    for i in range(num_robots):
        for suffix in ("eef_pos", "eef_rot_axis_angle", "gripper_width"):
            k = f"robot{i}_{suffix}"
            if k in _PROCESS_DATA: ed[k] = _PROCESS_DATA[k][start_idx:stop_idx]

    for src_key, _ in IMAGE_KEYS:
        if src_key in _PROCESS_DATA:
            ed[src_key] = _PROCESS_DATA[src_key][start_idx:stop_idx]

    # ------------------------------------------------------------------
    # 预计算所有机器人的位姿矩阵（避免重复计算）
    # 初始位姿用第一帧的 pos+rot，不再依赖 demo_start_pose
    # ------------------------------------------------------------------
    init_mats = []
    curr_mats_all = []  # [robot_idx][local_idx]
    for i in range(num_robots):
        pos = ed.get(f"robot{i}_eef_pos")
        rot = ed.get(f"robot{i}_eef_rot_axis_angle")

        if pos is not None and rot is not None:
            poses = np.concatenate([pos, rot], axis=-1)  # (ep_len, 6)
            init_mats.append(pose_to_mat(poses[0]))
            curr_mats_all.append([pose_to_mat(poses[t]) for t in range(ep_len)])
        else:
            init_mats.append(np.eye(4))
            curr_mats_all.append([np.eye(4)] * ep_len)

    # ------------------------------------------------------------------
    # 对每个机器人的位姿序列做高斯平滑
    # ------------------------------------------------------------------
    for i in range(num_robots):
        curr_mats_all[i] = _smooth_pose_matrices(curr_mats_all[i], sigma=smooth_sigma)

    # ------------------------------------------------------------------
    # 构建帧列表（轻量返回）
    # ------------------------------------------------------------------
    frames = []
    for t in range(ep_len):
        f = {}
        global_idx = start_idx + t

        # 语言指令
        f["task"] = language_instruction[min(global_idx, len(language_instruction) - 1)]

        # ---------- 图像 → JPEG bytes ----------
        for src_key, feat_key in IMAGE_KEYS:
            if src_key in ed:
                img = _process_image(ed[src_key][t])
                f[feat_key] = _encode_image_jpeg(img)
            else:
                f[feat_key] = b''  # 空标记

        # ---------- 状态 ----------
        state = []

        if no_state:
            for i in range(num_robots):
                gk = f"robot{i}_gripper_width"
                if gk in ed:
                    g = ed[gk][t]
                    state.append(float(g[0]) if hasattr(g, "__len__") else float(g))
                else:
                    state.append(0.0)
        else:
            for i in range(num_robots):
                c2w = curr_mats_all[i][t]
                c2i = np.linalg.inv(init_mats[i]) @ c2w
                state.extend(mat_to_pose(c2i))

                gk = f"robot{i}_gripper_width"
                if gk in ed:
                    g = ed[gk][t]
                    state.append(float(g[0]) if hasattr(g, "__len__") else float(g))
                else:
                    state.append(0.0)

            if num_robots >= 2:
                state.extend(mat_to_pose(
                    np.linalg.inv(curr_mats_all[1][t]) @ curr_mats_all[0][t]
                ))

        assert len(state) == state_dim, f"state dimension mismatch: {len(state)} != {state_dim}"
        f["observation.state"] = np.asarray(state, dtype=np.float32)

        # ---------- 动作 ----------
        if t < ep_len - 1:
            act = []
            for i in range(num_robots):
                c2w = curr_mats_all[i][t]
                n2w = curr_mats_all[i][t + 1]
                n2c = np.linalg.inv(c2w) @ n2w
                pos3 = mat_to_pose(n2c)[:3]
                r1 = n2c[:3, 0]
                r2 = n2c[:3, 1]
                act.extend(np.concatenate([pos3, r1, r2]))

                gk = f"robot{i}_gripper_width"
                if gk in ed:
                    act.extend(ed[gk][t])
                else:
                    act.append(0.0)
            f["actions"] = np.asarray(act[:action_dim], dtype=np.float32)
        else:
            f["actions"] = np.zeros(action_dim, dtype=np.float32)

        frames.append(f)

    # 🔥 释放大数组，减少内存峰值
    del ed, curr_mats_all
    return ep_idx, frames


# ============================================================================
# 主转换器
# ============================================================================
class ZarrToLeRobotConverter:
    def __init__(self, zarr_path, output_repo_id, fps=30,
                 state_dim=20, action_dim=20,
                 language_instruction=None, single_arm=False,
                 smooth_sigma=2.0, no_state=False):
        self.zarr_path = Path(zarr_path)
        self.output_repo_id = output_repo_id
        self.fps = fps
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.single_arm = single_arm
        self.smooth_sigma = smooth_sigma
        self.no_state = no_state
        self.language_instruction = language_instruction or ["perform bimanual manipulation task"]

        if not self.zarr_path.exists():
            raise ValueError(f"Zarr 文件不存在: {self.zarr_path}")

        # 只做结构分析，立即关闭
        store = ZipStore(self.zarr_path, mode="r")
        root = zarr.open_group(store=store, mode="r")
        data = root["data"]

        keys = list(data.keys())

        if self.single_arm:
            self.num_robots = 1
            self.num_cameras = 1
        else:
            self.num_robots = len([k for k in keys if 'eef_pos' in k])
            self.num_cameras = len([k for k in keys if 'rgb' in k])
        self.episode_ends = root["meta"]["episode_ends"][:]

        # 获取图像尺寸
        cam_keys = [k for k in keys if 'rgb' in k]
        if cam_keys:
            sample = _process_image(data[cam_keys[0]][0])
            self.img_size = sample.shape
        else:
            self.img_size = (224, 224, 3)

        store.close()

        n_ep = len(self.episode_ends)
        n_steps = int(self.episode_ends[-1]) if n_ep > 0 else 0
        mode_str = "单臂模式" if self.single_arm else "双臂模式"
        state_str = "no_state(仅夹爪宽度)" if self.no_state else "full_state"
        print(f"数据概览 [{mode_str}, {state_str}]: {n_ep} episodes, {n_steps} steps, {self.num_robots} robots, {self.num_cameras} cameras")

    def create_dataset(self):
        features = {}
        for _, feat_key in IMAGE_KEYS:
            features[feat_key] = {
                "dtype": "image", "shape": self.img_size,
                "names": ["height", "width", "channel"],
            }
        features["observation.state"] = {
            "dtype": "float32", "shape": (self.state_dim,),
            "names": ["observation.state"],
        }
        features["actions"] = {
            "dtype": "float32", "shape": (self.action_dim,),
            "names": ["actions"],
        }
        robot_type = "single_arm" if self.single_arm else "bimanual"
        return LeRobotDataset.create(
            repo_id=self.output_repo_id, fps=self.fps,
            robot_type=robot_type, features=features,
            use_videos=False,
            image_writer_threads=10, image_writer_processes=5,
        )

    def convert(self, num_workers=None, batch_size=16):
        """
        🔥 优化3: 分批提交 + 流式写入
        
        不再一次性把所有 episode 结果缓存在内存中，
        而是每收到 batch_size 个 episode 的结果就立即写入磁盘并释放。
        
        Args:
            num_workers: 进程池大小
            batch_size: 每批处理的 episode 数（控制内存峰值）
        """
        n_episodes = len(self.episode_ends)
        if num_workers is None:
            num_workers = min(8, cpu_count() or 4)
        num_workers = max(1, min(num_workers, n_episodes))

        print(f"\n{'='*60}")
        print(f"开始转换: {n_episodes} episodes, {num_workers} workers, batch={batch_size}, smooth_sigma={self.smooth_sigma}")
        print(f"{'='*60}\n")

        dataset = self.create_dataset()
        empty_img = np.zeros(self.img_size, dtype=np.uint8)

        # 构建所有 episode 的参数
        all_args = []
        for ep in range(n_episodes):
            start = 0 if ep == 0 else int(self.episode_ends[ep - 1])
            stop = int(self.episode_ends[ep])
            all_args.append((
                ep, start, stop, self.num_robots,
                self.state_dim, self.action_dim,
                self.language_instruction, self.smooth_sigma,
                self.no_state
            ))

        total_frames = 0

        # ------------------------------------------------------------------
        # 🔥 分批处理: 每 batch_size 个 episode 为一组
        # ------------------------------------------------------------------
        for batch_start in range(0, n_episodes, batch_size):
            batch_end = min(batch_start + batch_size, n_episodes)
            batch_args = all_args[batch_start:batch_end]
            batch_len = batch_end - batch_start

            # 收集本批次结果（有序缓冲区，只保存当前批次）
            batch_results = OrderedDict()

            with Pool(processes=num_workers,
                      initializer=_worker_init,
                      initargs=(str(self.zarr_path),)) as pool:

                for ep_idx, frames in tqdm(
                    pool.imap_unordered(_build_episode_worker, batch_args),
                    total=batch_len,
                    desc=f"Batch {batch_start//batch_size+1}/{(n_episodes+batch_size-1)//batch_size}",
                    ncols=70,
                ):
                    batch_results[ep_idx] = frames

            # 按顺序写入本批次
            for ep_idx in sorted(batch_results.keys()):
                frames = batch_results[ep_idx]
                for f in frames:
                    # 解码 JPEG → numpy（在主进程中做，写入前才解码）
                    for _, feat_key in IMAGE_KEYS:
                        raw = f[feat_key]
                        f[feat_key] = _decode_image_jpeg(raw) if raw else empty_img
                    dataset.add_frame(f)
                dataset.save_episode()
                total_frames += len(frames)

            # 🔥 批次写完后立即释放
            del batch_results

        print(f"\n{'='*60}")
        print(f"✅ 转换完成! {n_episodes} episodes, {total_frames} frames")
        print(f"保存位置: {dataset.root}")
        print(f"{'='*60}")
        return dataset


def _get_data_name_from_config(config_name: str) -> str:
    config = training_config.get_config(config_name)
    repo_id = getattr(config.data, "repo_id", None)
    if repo_id:
        return repo_id.split("/")[-1]
    raise ValueError(f"无法从配置 {config_name} 推断 data_name")

def _get_dim_from_config(config_name: str) -> str:
    config = training_config.get_config(config_name)
    state_dim = config.model.state_dim
    action_dim = config.model.action_dim
    return state_dim, action_dim


def main():
    parser = argparse.ArgumentParser(description='Zarr → LeRobot 转换（优化版）')
    parser.add_argument('--config_name', type=str, default="pi05_chaoyi")
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--language_instruction', type=list,
                        default=["Open the red pot, pick up the blue cylinder on the table and place it into the pot."])
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=16,
                        help='每批处理的 episode 数，越小内存越省（默认16）')

    '''单臂模式，位姿平滑和no_state模式'''
    parser.add_argument('--single_arm_mode', type=bool, default=False, 
                        help='单臂模式：强制 num_robots=1, num_cameras=1，跳过双臂相对位姿计算')
    parser.add_argument('--smooth_sigma', type=float, default=1.0,
                        help='位姿高斯平滑的 sigma（帧数），<=0 则不平滑（默认2.0）')
    parser.add_argument('--no_state', action='store_true', default=True,
                        help='no_state 模式：state 仅保留夹爪宽度，不包含位姿信息')
    
    args = parser.parse_args()

    data_name = _get_data_name_from_config(args.config_name)
    zarr_path = Path(f'./data/{data_name}.zarr.zip')
    repo_id = f'chaoyi/{data_name}'
    state_dim, action_dim = _get_dim_from_config(args.config_name)

    if not zarr_path.exists():
        print(f"错误: 找不到 {zarr_path}")
        sys.exit(1)

    mode_str = "单臂" if args.single_arm_mode else "双臂"
    state_str = "no_state" if args.no_state else "full_state"
    print(f"Zarr: {zarr_path.absolute()}")
    print(f"目标: {repo_id}")
    print(f"模式: {mode_str}, {state_str}")

    try:
        converter = ZarrToLeRobotConverter(
            zarr_path=zarr_path, output_repo_id=repo_id,
            fps=args.fps, state_dim=state_dim, action_dim=action_dim,
            language_instruction=args.language_instruction,
            single_arm=args.single_arm_mode,
            smooth_sigma=args.smooth_sigma,
            no_state=args.no_state,
        )
        converter.convert(num_workers=args.num_workers, batch_size=args.batch_size)
    except Exception as e:
        print(f"\n转换失败: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
