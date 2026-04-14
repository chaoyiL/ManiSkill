import sys
import os
import time
import argparse
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(str(ROOT_DIR))
os.chdir(ROOT_DIR)
from real_world.robot_api.arm.RobotControl_pykin import RobotControl
from utils.pose_util import pose_to_mat, mat_to_pose
import cv2
from policy.common.replay_buffer import ReplayBuffer
import zarr
from utils.imagecodecs_numcodecs import register_codecs
import transforms3d as t3d
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

'''
#测试模式：
uv run python Data_collection/check_data_quality/replay_using_quest_pose.py \
  --replay_mode action \
  --test_left_rx_action \
  --test_steps 20 \
  --test_deg_per_step 2.0 \
  --test_use_identity_tx_left

uv run python Data_collection/check_data_quality/replay_using_quest_pose.py \
  --replay_mode action \
  --test_left_tx_action \
  --test_steps 10 \
  --test_dx_per_step 0.005 \
  --test_use_identity_tx_left

uv run python Data_collection/check_data_quality/replay_using_quest_pose.py \
  --replay_mode action 
'''

def mat_to_pos_quat(mat):
    pos = list(mat[:3, 3])
    quat = list(t3d.quaternions.mat2quat(mat[:3, :3]))
    return np.array(pos+quat)

def pos_quat_to_mat(pos, quar):
    mat = np.eye(4)
    mat[:3, 3] = pos
    mat[:3, :3] = t3d.quaternions.quat2mat(quar)
    return mat

register_codecs()

_mat_norm = np.eye(4)

tx_quest_2_ee_left = np.load('/home/rvsa/codehub/VB-VLA/quest_2_ee_left_hand_fix_quest.npy')
tx_quest_2_ee_right = np.load('/home/rvsa/codehub/VB-VLA/quest_2_ee_right_hand_fix_quest.npy')
# tx_quest_2_ee_right = _mat_norm
# tx_quest_2_ee_left = _mat_norm
a = 2.041300
b = 0.110115


def get_transformation_matrix(position, rotation_vector):
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = position

    return transformation_matrix

def get_transformation_matrix_from_9d_action(action: np.ndarray):
    action = np.asarray(action)
    assert action.shape[-1] == 9, "输入的action最后一维必须为9"
    x = action[0]
    y = action[1]
    z = action[2]
    # 构造旋转矩阵
    R_col0 = np.array([action[3], action[4], action[5]])  
    R_col1 = np.stack([action[6], action[7], action[8]])
    # 保证正交性，通过叉乘得到第三列
    R_col2 = np.cross(R_col0, R_col1)
    R_col0 = R_col0.reshape([3,1])
    R_col1 = R_col1.reshape([3,1])
    R_col2 = R_col2.reshape([3,1])
    # 合成旋转矩阵
    R_mat = np.concatenate([R_col0, R_col1, R_col2], axis=-1)    # (3, 3)
    # print(R_mat)
    # 合成4x4变换矩阵
    trans_mat = np.eye(4)
    trans_mat[:3, :3] = R_mat
    trans_mat[:3, 3] = np.array([x, y, z])

    return trans_mat

def get_robot_pose_list_from_state(quest_pose_list: list, width_list: list, hand: str):
    if hand == 'left':
        tx_quest_2_ee = tx_quest_2_ee_left
    else:
        tx_quest_2_ee = tx_quest_2_ee_right

    ee_pose_list = [quest_pose @ np.linalg.inv(tx_quest_2_ee) for quest_pose in quest_pose_list]
    ee_pose_0 = ee_pose_list[0]  # First pose
    base_2_ee_0 = np.linalg.inv(ee_pose_0)
    relative_pose_list = [base_2_ee_0 @ ee_pose for ee_pose in ee_pose_list]

    return relative_pose_list, width_list

def get_robot_pose_list_from_action(action_list: list, width_list: list, hand: str):
    if hand == 'left':
        tx_quest_2_ee = tx_quest_2_ee_left
    else:
        tx_quest_2_ee = tx_quest_2_ee_right

    quest_action_list = [get_transformation_matrix_from_9d_action(action) for action in action_list]
    ee_pose_list = [tx_quest_2_ee @ quest_action @ np.linalg.inv(tx_quest_2_ee) for quest_action in quest_action_list]
    ee_pose_0 = np.eye(4)
    relative_pose_list = [ee_pose_0]

    for ee_pose in ee_pose_list:
        relative_pose_list.append(relative_pose_list[-1] @ ee_pose)
    
    width_list = [width[0] for width in width_list]

    return relative_pose_list, width_list

def build_rx_action_9d(delta_rad: float) -> np.ndarray:
    """构造单步绕 x 轴旋转的 9D action（无平移）。"""
    c = np.cos(delta_rad)
    s = np.sin(delta_rad)
    # Rx = [[1, 0, 0],
    #       [0, c,-s],
    #       [0, s, c]]
    return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, c, s], dtype=np.float32)

def build_tx_action_9d(delta_step: float) -> np.ndarray:
    """构造单步沿 x 轴平移的 9D action（无旋转）。"""
    return np.array([delta_step, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

def build_identity_action_9d() -> np.ndarray:
    """构造单步单位变换的 9D action（无平移无旋转）。"""
    return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


def build_left_rx_action_test_data(steps: int, deg_per_step: float, left_width: float, right_width: float):
    """生成 action replay 测试数据：左手绕 x 轴旋转，右手保持静止。"""
    delta_rad = np.deg2rad(deg_per_step)
    action_left_list = [build_rx_action_9d(delta_rad) for _ in range(steps)]
    action_right_list = [build_identity_action_9d() for _ in range(steps)]
    width_left_list = [np.array([left_width], dtype=np.float32) for _ in range(steps)]
    width_right_list = [np.array([right_width], dtype=np.float32) for _ in range(steps)]

    return width_left_list, width_right_list, action_left_list, action_right_list


def build_left_tx_action_test_data(steps: int, dx_per_step: float, left_width: float, right_width: float):
    """生成 action replay 测试数据：左手沿 x 轴平移，右手保持静止。"""
    action_left_list = [build_tx_action_9d(dx_per_step) for _ in range(steps)]
    action_right_list = [build_identity_action_9d() for _ in range(steps)]
    width_left_list = [np.array([left_width], dtype=np.float32) for _ in range(steps)]
    width_right_list = [np.array([right_width], dtype=np.float32) for _ in range(steps)]
    return width_left_list, width_right_list, action_left_list, action_right_list

def exec_arms(relative_pose_list_left, width_left_list, relative_pose_list_right, width_right_list):
    robot = RobotControl(vel_max=0.1)

    ee_pose = robot.get_ee_pose()
    ee2base_left_0 = pos_quat_to_mat(ee_pose["left_arm_ee2rb"][:3], ee_pose["left_arm_ee2rb"][3:])
    ee2base_right_0 = pos_quat_to_mat(ee_pose["right_arm_ee2rb"][:3], ee_pose["right_arm_ee2rb"][3:])

    next_pose_left_list = [ee2base_left_0 @ relative_pose for relative_pose in relative_pose_list_left]
    next_pose_right_list = [ee2base_right_0 @ relative_pose for relative_pose in relative_pose_list_right]
    step_idx = 0

    # print(len(next_pose_left_list), len(width_left_list), len(next_pose_right_list), len(width_right_list))
    for i, (pose_left, width_left, pose_right, width_right) in enumerate(zip(next_pose_left_list, width_left_list, next_pose_right_list, width_right_list)):

        commanded_width_left = [float((width_left - b) / a)]
        commanded_width_right = [float((width_right - b) / a)]

        ee_left = mat_to_pos_quat(pose_left)
        ee_right = mat_to_pos_quat(pose_right)

        robot.set_target_CP(
            target_pose={
                "left_arm_ee2rb": ee_left,
                "right_arm_ee2rb": ee_right,
                "left_gripper": commanded_width_left,
                "right_gripper": commanded_width_right
            }
        )
        
        # 执行动作
        target_joint_left = robot.action_target["left_arm"]
        target_joint_right = robot.action_target["right_arm"]

        exec_step = 0
        while True:
            robot.execute()
            curr_joint_left = robot.get_robot_joints()["left_arm"]
            curr_joint_right = robot.get_robot_joints()["right_arm"]
            if np.linalg.norm(target_joint_left - curr_joint_left) < 1e-3 and np.linalg.norm(target_joint_right - curr_joint_right) < 1e-3:
                break
            time.sleep(0.01)
            exec_step += 1
            if np.linalg.norm(target_joint_left - curr_joint_left) > 0.3 or np.linalg.norm(target_joint_right - curr_joint_right) > 0.3:
                print("large error in execution!")
                # return
            if exec_step > 100:
                break
            
        print(f"step {i}: left_width={commanded_width_left[0]:.4f}, right_width={commanded_width_right[0]:.4f}")
        time.sleep(0.01)


def _resolve_dataset_path(path: str) -> str:
    if path.startswith('~'):
        path = os.path.expanduser(path)
    return os.path.abspath(path) if os.path.exists(path) else path


def load_replay_from_zarr(zarr_path: str, replay_episode: int):
    with zarr.ZipStore(zarr_path, mode='r') as zip_store:
        replay_buffer = ReplayBuffer.copy_from_store(
            src_store=zip_store, store=zarr.MemoryStore())
    episode_slice = replay_buffer.get_episode_slice(replay_episode)
    start_idx = episode_slice.start
    stop_idx = episode_slice.stop

    traj_left_list = []
    traj_right_list = []
    width_left_list = []
    width_right_list = []
    action_list = []

    while True:
        pos_left = replay_buffer['robot0_eef_pos'][start_idx]
        rot_left = replay_buffer['robot0_eef_rot_axis_angle'][start_idx]
        pos_right = replay_buffer['robot1_eef_pos'][start_idx]
        rot_right = replay_buffer['robot1_eef_rot_axis_angle'][start_idx]

        traj_left = get_transformation_matrix(pos_left, rot_left)
        traj_right = get_transformation_matrix(pos_right, rot_right)

        width_left = replay_buffer['robot0_gripper_width'][start_idx]
        width_right = replay_buffer['robot1_gripper_width'][start_idx]

        traj_left_list.append(traj_left)
        traj_right_list.append(traj_right)
        width_left_list.append(width_left)
        width_right_list.append(width_right)

        if 'action' in replay_buffer.keys():
            action_list.append(np.asarray(replay_buffer['action'][start_idx], dtype=np.float32))

        if start_idx == stop_idx:
            break
        start_idx += 1

    return traj_left_list, width_left_list, traj_right_list, width_right_list, action_list


def load_replay_from_lerobot(repo_id_or_path: str, replay_episode: int):
    if load_dataset is None:
        raise ImportError("未安装 datasets，请先执行: pip install datasets")

    dataset_path = _resolve_dataset_path(repo_id_or_path)
    # 流式读取，避免一次性加载整个数据集。优先读取 train split
    try:
        dataset = load_dataset(dataset_path, split='train', streaming=True)
    except Exception:
        dataset_dict = load_dataset(dataset_path, streaming=True)
        split_name = 'train' if 'train' in dataset_dict else list(dataset_dict.keys())[0]
        dataset = dataset_dict[split_name]

    traj_left_list = []
    traj_right_list = []
    width_left_list = []
    width_right_list = []
    action_left_list = []
    action_right_list = []

    found_target_episode = False
    for item in dataset:
        episode_idx = int(item.get('episode_index', -1))
        if episode_idx < replay_episode:
            continue
        if episode_idx > replay_episode:
            if found_target_episode:
                break
            continue
        found_target_episode = True

        if 'observation.state' not in item:
            raise KeyError("LeRobot 样本缺少 observation.state 字段")

        state = np.asarray(item['observation.state'], dtype=np.float32).reshape(-1)
        if state.shape[0] < 14:
            raise ValueError(f"observation.state 维度不足，期望>=14，实际={state.shape[0]}")

        pos_left = state[0:3]
        rot_left = state[3:6]
        width_left = np.array([state[6]], dtype=np.float32)
        pos_right = state[7:10]
        rot_right = state[10:13]
        width_right = np.array([state[13]], dtype=np.float32)

        traj_left_list.append(get_transformation_matrix(pos_left, rot_left))
        traj_right_list.append(get_transformation_matrix(pos_right, rot_right))
        width_left_list.append(width_left)
        width_right_list.append(width_right)

        if 'actions' in item:
            action_left = item['actions'][0:9]
            action_right = item['actions'][10:19]
            action_left_list.append(np.asarray(action_left, dtype=np.float32).reshape(-1))
            action_right_list.append(np.asarray(action_right, dtype=np.float32).reshape(-1))

    if len(traj_left_list) == 0:
        raise ValueError(f"在 LeRobot 数据集中未找到 episode_index={replay_episode} 的样本")

    return traj_left_list, width_left_list, traj_right_list, width_right_list, action_left_list, action_right_list


if __name__ == "__main__":

    np.set_printoptions(precision=3, suppress=True)
    parser = argparse.ArgumentParser(description='回放双臂轨迹，支持 zarr / LeRobot 数据集')
    parser.add_argument(
        '--input',
        type=str,
        default='/home/rvsa/codehub/VB-VLA/data/0118_data',
        help='zarr.zip 路径或 LeRobot repo_id/本地目录'
    )
    parser.add_argument(
        '--dataset_type',
        type=str,
        choices=['zarr', 'lerobot'],
        default='lerobot',
        help='输入数据类型'
    )
    parser.add_argument('--episode', type=int, default=0, help='要回放的 episode 索引')
    parser.add_argument('--replay_mode', type=str, default='action', choices=['state', 'action'], help='回放模式')
    parser.add_argument('--test_left_rx_action', action='store_true', help='测试模式：左手绕 x 轴旋转（action replay）')
    parser.add_argument('--test_left_tx_action', action='store_true', help='测试模式：左手沿 x 轴平移（action replay）')
    parser.add_argument('--test_steps', type=int, default=20, help='测试模式旋转步数')
    parser.add_argument('--test_deg_per_step', type=float, default=2.0, help='测试模式每步旋转角度（度）')
    parser.add_argument('--test_dx_per_step', type=float, default=0.005, help='测试模式每步 x 方向平移（米）')
    parser.add_argument('--test_left_width', type=float, default=0.08, help='测试模式左夹爪宽度')
    parser.add_argument('--test_right_width', type=float, default=0.08, help='测试模式右夹爪宽度')
    parser.add_argument('--test_use_identity_tx_left', action='store_true', help='测试模式将左手 tx_quest_2_ee 设为单位阵，便于验证纯 x 轴旋转')
    args = parser.parse_args()

    if args.test_left_rx_action and args.test_left_tx_action:
        raise ValueError("test_left_rx_action 与 test_left_tx_action 只能二选一")

    if args.test_left_rx_action or args.test_left_tx_action:
        if args.replay_mode != 'action':
            raise ValueError("测试模式仅支持 --replay_mode action")
        if args.test_use_identity_tx_left:
            tx_quest_2_ee_left[:] = np.eye(4)
        if args.test_left_rx_action:
            width_left_list, width_right_list, action_left_list, action_right_list = build_left_rx_action_test_data(
                steps=args.test_steps,
                deg_per_step=args.test_deg_per_step,
                left_width=args.test_left_width,
                right_width=args.test_right_width
            )
            print(f"[TEST] 左手绕x轴旋转: steps={args.test_steps}, deg_per_step={args.test_deg_per_step}")
        else:
            width_left_list, width_right_list, action_left_list, action_right_list = build_left_tx_action_test_data(
                steps=args.test_steps,
                dx_per_step=args.test_dx_per_step,
                left_width=args.test_left_width,
                right_width=args.test_right_width
            )
            print(f"[TEST] 左手沿x轴平移: steps={args.test_steps}, dx_per_step={args.test_dx_per_step}")
    elif args.dataset_type == 'zarr':
        print("不要用这个")
    else:
        traj_left_list, width_left_list, traj_right_list, width_right_list, action_left_list, action_right_list = load_replay_from_lerobot(
            args.input, args.episode
        )

    replay_mode = args.replay_mode
    if replay_mode == 'state':
        if args.test_left_rx_action or args.test_left_tx_action:
            raise ValueError("测试模式未提供 state 轨迹，请使用 --replay_mode action")
        relative_pose_list_left, width_list_left = get_robot_pose_list_from_state(traj_left_list, width_left_list, hand='left')
        relative_pose_list_right, width_list_right = get_robot_pose_list_from_state(traj_right_list, width_right_list, hand='right')
    elif replay_mode == 'action':
        relative_pose_list_left, width_list_left = get_robot_pose_list_from_action(action_left_list, width_left_list, hand='left')
        relative_pose_list_right, width_list_right = get_robot_pose_list_from_action(action_right_list, width_right_list, hand='right')
    else:
        raise ValueError(f"Invalid replay mode: {replay_mode}")
    
    print("ready to exec!")
    exec_arms(
        relative_pose_list_left, width_list_left,
        relative_pose_list_right, width_list_right
    )