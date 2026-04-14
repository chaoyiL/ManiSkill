import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import copy
import cv2
import zarr

from policy.common.replay_buffer import ReplayBuffer
from utils.imagecodecs_numcodecs import register_codecs
register_codecs()

def get_transformation_matrix(pos, rot_axis_angle):
    rotation_matrix, _ = cv2.Rodrigues(rot_axis_angle)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = pos
    
    return transformation_matrix

def create_bimanual_combined_image(visual_1, left_tactile_1, right_tactile_1, 
                                   visual_0, left_tactile_0, right_tactile_0):
    """创建双手图像拼接：Robot0第一排，Robot1第二排"""
    
    # 第一排：Robot0图像 (visual, left_tactile, right_tactile)
    robot0_images = []
    robot0_labels = []
    
    if visual_0 is not None:
        robot0_images.append(visual_0)
        robot0_labels.append("Robot0 Visual")
    if left_tactile_0 is not None:
        robot0_images.append(left_tactile_0)
        robot0_labels.append("Robot0 L-Tactile")
    if right_tactile_0 is not None:
        robot0_images.append(right_tactile_0)
        robot0_labels.append("Robot0 R-Tactile")
    
    # 第二排：Robot1图像 (visual, left_tactile, right_tactile)
    robot1_images = []
    robot1_labels = []
    
    if visual_1 is not None:
        robot1_images.append(visual_1)
        robot1_labels.append("Robot1 Visual")
    if left_tactile_1 is not None:
        robot1_images.append(left_tactile_1)
        robot1_labels.append("Robot1 L-Tactile")
    if right_tactile_1 is not None:
        robot1_images.append(right_tactile_1)
        robot1_labels.append("Robot1 R-Tactile")
    
    # 如果没有图像，返回None
    if not robot0_images and not robot1_images:
        return None
    
    # 统一图像尺寸
    target_height = 250
    
    # 处理Robot0图像行
    robot0_row = None
    if robot0_images:
        resized_robot0 = []
        for img in robot0_images:
            h, w = img.shape[:2]
            target_width = int(w * target_height / h)
            resized_img = cv2.resize(img, (target_width, target_height))
            resized_robot0.append(resized_img)
        
        # 水平拼接Robot0图像
        robot0_row = np.hstack(resized_robot0)
        
        # 添加标签
        x_offset = 0
        for i, (img, label) in enumerate(zip(resized_robot0, robot0_labels)):
            cv2.putText(robot0_row, label, (x_offset + 10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(robot0_row, label, (x_offset + 10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)  # 红色
            
            if i < len(resized_robot0) - 1:
                x_offset += img.shape[1]
                cv2.line(robot0_row, (x_offset, 0), (x_offset, target_height), (255, 255, 255), 2)
            else:
                x_offset += img.shape[1]
    
    # 处理Robot1图像行
    robot1_row = None
    if robot1_images:
        resized_robot1 = []
        for img in robot1_images:
            h, w = img.shape[:2]
            target_width = int(w * target_height / h)
            resized_img = cv2.resize(img, (target_width, target_height))
            resized_robot1.append(resized_img)
        
        # 水平拼接Robot1图像
        robot1_row = np.hstack(resized_robot1)
        
        # 添加标签
        x_offset = 0
        for i, (img, label) in enumerate(zip(resized_robot1, robot1_labels)):
            cv2.putText(robot1_row, label, (x_offset + 10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(robot1_row, label, (x_offset + 10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)  # 绿色
            
            if i < len(resized_robot1) - 1:
                x_offset += img.shape[1]
                cv2.line(robot1_row, (x_offset, 0), (x_offset, target_height), (255, 255, 255), 2)
            else:
                x_offset += img.shape[1]
    
    # 垂直拼接两行：Robot0在上，Robot1在下
    if robot0_row is not None and robot1_row is not None:
        # 确保两行宽度相同
        max_width = max(robot0_row.shape[1], robot1_row.shape[1])
        
        if robot0_row.shape[1] < max_width:
            padding = np.zeros((target_height, max_width - robot0_row.shape[1], 3), dtype=np.uint8)
            robot0_row = np.hstack([robot0_row, padding])
        
        if robot1_row.shape[1] < max_width:
            padding = np.zeros((target_height, max_width - robot1_row.shape[1], 3), dtype=np.uint8)
            robot1_row = np.hstack([robot1_row, padding])
        
        # 在两行之间添加分割线
        separator = np.ones((5, max_width, 3), dtype=np.uint8) * 128  # 灰色分割线
        combined_image = np.vstack([robot0_row, separator, robot1_row])
        
    elif robot0_row is not None:
        combined_image = robot0_row
    elif robot1_row is not None:
        combined_image = robot1_row
    else:
        return None
    
    return combined_image

def load_episode_data(replay_buffer, episode_idx):
    """加载特定episode的数据"""
    episode_slice = replay_buffer.get_episode_slice(episode_idx)
    start_idx = episode_slice.start
    stop_idx = episode_slice.stop

    # 存储变换矩阵和对应的图像
    pos_list_0 = []  # Robot0轨迹
    pos_list_1 = []  # Robot1轨迹
    
    # 各种图像数据
    visual_images_0 = []      # Robot0 visual图像
    visual_images_1 = []      # Robot1 visual图像
    left_tactile_images_0 = []  # Robot0左触觉
    right_tactile_images_0 = [] # Robot0右触觉
    left_tactile_images_1 = []  # Robot1左触觉
    right_tactile_images_1 = [] # Robot1右触觉
    
    gripper_widths_0 = []  # Robot0夹爪宽度
    gripper_widths_1 = []  # Robot1夹爪宽度
    # first_frame_tx_0 = None
    # first_frame_tx_1 = None

    # 检查各种数据的存在性
    has_visual_0 = 'camera0_rgb' in replay_buffer.keys()
    has_visual_1 = 'camera1_rgb' in replay_buffer.keys()
    has_left_tactile_0 = 'camera0_left_tactile' in replay_buffer.keys()
    has_right_tactile_0 = 'camera0_right_tactile' in replay_buffer.keys()
    has_left_tactile_1 = 'camera1_left_tactile' in replay_buffer.keys()
    has_right_tactile_1 = 'camera1_right_tactile' in replay_buffer.keys()
    has_robot1_data = 'robot1_eef_pos' in replay_buffer.keys()
    has_gripper0_data = 'robot0_gripper_width' in replay_buffer.keys()
    has_gripper1_data = 'robot1_gripper_width' in replay_buffer.keys()
    
    print(f"📊 数据检查结果:")
    print(f"  Robot0 (Camera0):")
    print(f"    Visual RGB: {'✅' if has_visual_0 else '❌'}")
    print(f"    Left Tactile: {'✅' if has_left_tactile_0 else '❌'}")
    print(f"    Right Tactile: {'✅' if has_right_tactile_0 else '❌'}")
    print(f"    Gripper Width: {'✅' if has_gripper0_data else '❌'}")
    print(f"  Robot1 (Camera1):")
    print(f"    Visual RGB: {'✅' if has_visual_1 else '❌'}")
    print(f"    Left Tactile: {'✅' if has_left_tactile_1 else '❌'}")
    print(f"    Right Tactile: {'✅' if has_right_tactile_1 else '❌'}")
    print(f"    Gripper Width: {'✅' if has_gripper1_data else '❌'}")
    print(f"    Robot1 Pose: {'✅' if has_robot1_data else '❌'}")

            # 处理轨迹数据
    for i in range(start_idx, stop_idx, 1):
        # 处理Robot0数据
        pos_0 = replay_buffer['robot0_eef_pos'][i]
        rot_0 = replay_buffer['robot0_eef_rot_axis_angle'][i]
        
        # 加载Robot0夹爪宽度数据
        if has_gripper0_data:
            gripper_width_0 = replay_buffer['robot0_gripper_width'][i][0]
            gripper_widths_0.append(gripper_width_0)
        
        # 加载Robot0各种图像
        if has_visual_0:
            img_data = replay_buffer['camera0_rgb'][i]
            if isinstance(img_data, bytes):
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = img_data
            visual_images_0.append(img)
        
        if has_left_tactile_0:
            img_data = replay_buffer['camera0_left_tactile'][i]
            if isinstance(img_data, bytes):
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = img_data
            left_tactile_images_0.append(img)
        
        if has_right_tactile_0:
            img_data = replay_buffer['camera0_right_tactile'][i]
            if isinstance(img_data, bytes):
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = img_data
            right_tactile_images_0.append(img)
        
        # 处理Robot1数据(如果存在)
        if has_robot1_data:
            pos_1 = replay_buffer['robot1_eef_pos'][i]
            rot_1 = replay_buffer['robot1_eef_rot_axis_angle'][i]
            
            # 加载Robot1夹爪宽度数据
            if has_gripper1_data:
                gripper_width_1 = replay_buffer['robot1_gripper_width'][i][0]
                gripper_widths_1.append(gripper_width_1)
            
            # 计算Robot1变换矩阵
            transform_1 = get_transformation_matrix(pos_1, rot_1)
            
            # 保存第一帧作为参考
            # if first_frame_tx_1 is None:
            #     first_frame_tx_1 = transform_1.copy()
            
            # rel_transform_1 = np.linalg.inv(first_frame_tx_1) @ transform_1
            # pos_list_1.append(rel_transform_1)
            transform_1
            pos_list_1.append(transform_1)
        
        # 加载Robot1各种图像
        if has_visual_1:
            img_data = replay_buffer['camera1_rgb'][i]
            if isinstance(img_data, bytes):
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = img_data
            visual_images_1.append(img)
        
        if has_left_tactile_1:
            img_data = replay_buffer['camera1_left_tactile'][i]
            if isinstance(img_data, bytes):
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = img_data
            left_tactile_images_1.append(img)
        
        if has_right_tactile_1:
            img_data = replay_buffer['camera1_right_tactile'][i]
            if isinstance(img_data, bytes):
                img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = img_data
            right_tactile_images_1.append(img)
        
        # 计算Robot0变换矩阵
        transform_0 = get_transformation_matrix(pos_0, rot_0)
        
        # if first_frame_tx_0 is None:
        #     first_frame_tx_0 = transform_0.copy()
        
        # # 计算相对于第一帧的变换
        # rel_transform_0 = np.linalg.inv(first_frame_tx_0) @ transform_0
        # pos_list_0.append(rel_transform_0)
        pos_list_0.append(transform_0)

    return (pos_list_0, pos_list_1, 
            visual_images_0, visual_images_1, left_tactile_images_0, right_tactile_images_0,
            left_tactile_images_1, right_tactile_images_1, gripper_widths_0, gripper_widths_1,
            has_visual_0, has_visual_1, has_left_tactile_0, has_right_tactile_0,
            has_left_tactile_1, has_right_tactile_1, has_robot1_data, has_gripper0_data, has_gripper1_data)

# 左黄右蓝
replay_buffer_path = '/home/rvsa/codehub/VB-vla/Data_collection/vitamin_b_data_collection_pipeline/pose_data.zarr.zip'
# replay_buffer_path = '/home/rvsa/Downloads/cube_storage.zip'
# replay_buffer_path = '/home/rvsa/codehub/VB-vla/data/_0118/_0118_quest_pose.zarr.zip'
with zarr.ZipStore(replay_buffer_path, mode='r') as zip_store:
    replay_buffer = ReplayBuffer.copy_from_store(src_store=zip_store, store=zarr.MemoryStore())

# 列出可用的episode - 使用正确的方法
print("🔍 分析ReplayBuffer结构...")
print(f"总帧数: {replay_buffer.n_steps}")
print(f"Episode数量: {replay_buffer.n_episodes}")

# 直接使用episode数量来生成可用的episodes
available_episodes = np.arange(replay_buffer.n_episodes)
print(f"\n📊 Episode信息:")
print(f"Episode数量: {len(available_episodes)}")
print(f"Episode IDs: 0 到 {len(available_episodes)-1}")

# 显示每个episode的长度统计
episode_lengths = replay_buffer.episode_lengths
print(f"\n📏 Episode长度统计:")
print(f"平均长度: {np.mean(episode_lengths):.1f} 帧")
print(f"最短: {np.min(episode_lengths)} 帧")
print(f"最长: {np.max(episode_lengths)} 帧")
print(f"前5个episode长度: {episode_lengths[:5]}")
print(f"后5个episode长度: {episode_lengths[-5:]}")

# 验证总帧数
print(f"\n✅ 验证: {np.sum(episode_lengths)} 总帧数 = {replay_buffer.n_steps} ReplayBuffer帧数")

# 检查数据类型可用性
print(f"\n🔍 双手模式数据类型检查:")
available_keys = list(replay_buffer.keys())
data_types = {
    'Robot0 Visual RGB': 'camera0_rgb' in available_keys,
    'Robot0 Left Tactile': 'camera0_left_tactile' in available_keys,
    'Robot0 Right Tactile': 'camera0_right_tactile' in available_keys,
    'Robot0 Gripper Width': 'robot0_gripper_width' in available_keys,
    'Robot0 Pose': 'robot0_eef_pos' in available_keys,
    'Robot1 Visual RGB': 'camera1_rgb' in available_keys,
    'Robot1 Left Tactile': 'camera1_left_tactile' in available_keys,
    'Robot1 Right Tactile': 'camera1_right_tactile' in available_keys,
    'Robot1 Gripper Width': 'robot1_gripper_width' in available_keys,
    'Robot1 Pose': 'robot1_eef_pos' in available_keys,
}

for data_type, available in data_types.items():
    status = "✅" if available else "❌"
    print(f"  {status} {data_type}")

visual_or_tactile_available = any([
    data_types['Robot0 Visual RGB'], data_types['Robot0 Left Tactile'], data_types['Robot0 Right Tactile'],
    data_types['Robot1 Visual RGB'], data_types['Robot1 Left Tactile'], data_types['Robot1 Right Tactile']
])

if visual_or_tactile_available:
    print(f"\n🖼️  图像显示: 可用图像类型将在 'Bimanual Multi-Camera View' 窗口中分层拼接显示")
    print(f"    - 第一排: Robot0图像 (Visual + Left Tactile + Right Tactile)")
    print(f"    - 第二排: Robot1图像 (Visual + Left Tactile + Right Tactile)")
else:
    print(f"\n⚠️  无图像数据可显示")

# 创建交互式可视化器 - 支持episode切换和双手轨迹
class BimanualEpisodeVisualizerWithKeyCallback:
    def __init__(self, replay_buffer, available_episodes):
        self.replay_buffer = replay_buffer
        self.available_episodes = available_episodes
        self.current_episode_idx = 0  # 当前episode在available_episodes中的索引
        self.current_frame_idx = 0    # 当前episode中的帧索引
        self.displayed_frames_0 = []  # 右手显示的坐标系列表
        self.displayed_frames_1 = []  # 左手显示的坐标系列表
        
        # 初始化第一个episode的数据
        self.load_current_episode()
        
        # 设置Open3D可视化窗口
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("Robot0/Robot1 Episode轨迹可视化", width=1280, height=720)
        
        # 初始化可视化
        self.init_visualization()
        
        # 注册键盘回调
        self.vis.register_key_callback(ord('D'), self.next_frame)
        self.vis.register_key_callback(ord('A'), self.prev_frame)
        self.vis.register_key_callback(ord('R'), self.reset_view)
        self.vis.register_key_callback(ord('Q'), self.quit)
        
        # 显示帮助信息
        print("\n🎮 Robot0/Robot1模式控制说明:")
        print("  📱 键盘控制:")
        print("    A/D: 上一帧/下一帧 (到达最后一帧时自动进入下一个episode)")
        print("    R: 重置视图")
        print("    Q: 退出")
        print("  🖱️  鼠标控制:")
        print("    左键拖动: 旋转视角")
        print("    滚轮: 缩放")
        print("    中键拖动: 平移视角")
        print("  🖼️  图像显示:")
        print("    - Open3D窗口: 3D Robot0/Robot1轨迹和end-effector pose可视化")
        print("    - Bimanual Multi-Camera View窗口: 分层拼接显示所有可用图像")
        print("      * 第一排: Robot0图像 (红色标签)")
        print("      * 第二排: Robot1图像 (绿色标签)")
        print("    - 控制台: 打印详细的Robot0/Robot1 pose坐标和gripper宽度信息")
        print("  ✨ 视角会在帧切换时保持不变！")
        print("  🟥 红色轨迹: Robot0 (robot0)")
        print("  🟢 绿色轨迹: Robot1 (robot1)")
        print(f"\n当前Episode: {self.available_episodes[self.current_episode_idx]} ({self.current_episode_idx + 1}/{len(self.available_episodes)})")
        
        # 运行可视化
        self.vis.run()
    
    def load_current_episode(self):
        """加载当前episode的数据"""
        episode_id = self.available_episodes[self.current_episode_idx]
        
        (self.pos_list_0, self.pos_list_1, 
         self.visual_images_0, self.visual_images_1, self.left_tactile_images_0, self.right_tactile_images_0,
         self.left_tactile_images_1, self.right_tactile_images_1, self.gripper_widths_0, self.gripper_widths_1,
         self.has_visual_0, self.has_visual_1, self.has_left_tactile_0, self.has_right_tactile_0,
         self.has_left_tactile_1, self.has_right_tactile_1, self.has_robot1_data, 
         self.has_gripper0_data, self.has_gripper1_data) = load_episode_data(self.replay_buffer, episode_id)
        
        # 创建3D轨迹可视化组件
        self.create_trajectory_components()
        
        # 重置帧索引
        self.current_frame_idx = 0
        
        # 如果不是第一次加载，清空显示的坐标系列表
        if hasattr(self, 'displayed_frames_0'):
            self.displayed_frames_0 = []
        if hasattr(self, 'displayed_frames_1'):
            self.displayed_frames_1 = []
        
        print(f"\n✅ 加载Episode {episode_id} (第{self.current_episode_idx + 1}个/{len(self.available_episodes)}):")
        print(f"   Robot0轨迹点数量: {len(self.pos_list_0)}")
        if self.has_robot1_data:
            print(f"   Robot1轨迹点数量: {len(self.pos_list_1)}")
        
        # 显示各种图像数量
        if self.has_visual_0:
            print(f"   Robot0 Visual图像: {len(self.visual_images_0)}")
        if self.has_left_tactile_0:
            print(f"   Robot0左触觉图像: {len(self.left_tactile_images_0)}")
        if self.has_right_tactile_0:
            print(f"   Robot0右触觉图像: {len(self.right_tactile_images_0)}")
        if self.has_visual_1:
            print(f"   Robot1 Visual图像: {len(self.visual_images_1)}")
        if self.has_left_tactile_1:
            print(f"   Robot1左触觉图像: {len(self.left_tactile_images_1)}")
        if self.has_right_tactile_1:
            print(f"   Robot1右触觉图像: {len(self.right_tactile_images_1)}")
        
        if self.has_gripper0_data:
            print(f"   Robot0夹爪宽度范围: {np.min(self.gripper_widths_0):.4f} - {np.max(self.gripper_widths_0):.4f}")
        if self.has_gripper1_data:
            print(f"   Robot1夹爪宽度范围: {np.min(self.gripper_widths_1):.4f} - {np.max(self.gripper_widths_1):.4f}")
    
    def create_trajectory_components(self):
        """创建轨迹的3D组件"""
        # 创建Robot0坐标系列表
        self.list_frame_0 = []
        for i in range(len(self.pos_list_0)):
            curr_pos = self.pos_list_0[i]
            if i == 0:
                curr_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
            else:
                curr_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
            curr_frame.transform(curr_pos)
            self.list_frame_0.append(copy.deepcopy(curr_frame))

        # 创建Robot0轨迹线和点云
        trajectory_points_0 = np.array([pos[:3, 3] for pos in self.pos_list_0])
        self.trajectory_line_0 = o3d.geometry.LineSet()
        self.trajectory_line_0.points = o3d.utility.Vector3dVector(trajectory_points_0)
        lines_0 = [[i, i+1] for i in range(len(trajectory_points_0)-1)]
        self.trajectory_line_0.lines = o3d.utility.Vector2iVector(lines_0)
        self.trajectory_line_0.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0] for _ in range(len(lines_0))]))

        self.trajectory_cloud_0 = o3d.geometry.PointCloud()
        self.trajectory_cloud_0.points = o3d.utility.Vector3dVector(trajectory_points_0)
        self.trajectory_cloud_0.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0] for _ in range(len(trajectory_points_0))]))

        # 创建Robot1坐标系列表(如果存在)
        self.list_frame_1 = []
        self.trajectory_line_1 = None
        self.trajectory_cloud_1 = None
        
        if self.has_robot1_data:
            for i in range(len(self.pos_list_1)):
                curr_pos = self.pos_list_1[i]
                if i == 0:
                    curr_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
                else:
                    curr_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
                curr_frame.transform(curr_pos)
                self.list_frame_1.append(copy.deepcopy(curr_frame))

            # 创建Robot1轨迹线和点云
            trajectory_points_1 = np.array([pos[:3, 3] for pos in self.pos_list_1])
            self.trajectory_line_1 = o3d.geometry.LineSet()
            self.trajectory_line_1.points = o3d.utility.Vector3dVector(trajectory_points_1)
            lines_1 = [[i, i+1] for i in range(len(trajectory_points_1)-1)]
            self.trajectory_line_1.lines = o3d.utility.Vector2iVector(lines_1)
            self.trajectory_line_1.colors = o3d.utility.Vector3dVector(np.array([[0, 1, 0] for _ in range(len(lines_1))]))

            self.trajectory_cloud_1 = o3d.geometry.PointCloud()
            self.trajectory_cloud_1.points = o3d.utility.Vector3dVector(trajectory_points_1)
            self.trajectory_cloud_1.colors = o3d.utility.Vector3dVector(np.array([[0, 1, 0] for _ in range(len(trajectory_points_1))]))
    
    def init_visualization(self):
        """初始化可视化"""
        # 添加完整轨迹线和点云
        self.vis.add_geometry(self.trajectory_line_0)
        self.vis.add_geometry(self.trajectory_cloud_0)
        
        if self.trajectory_line_1 is not None:
            self.vis.add_geometry(self.trajectory_line_1)
        if self.trajectory_cloud_1 is not None:
            self.vis.add_geometry(self.trajectory_cloud_1)
        
        # 预先添加所有坐标系到可视化器中，但设置为不可见
        self.displayed_frames_0 = []
        self.displayed_frames_1 = []
        
        for i, frame in enumerate(self.list_frame_0):
            self.vis.add_geometry(frame)
            self.displayed_frames_0.append(frame)
            
        # 隐藏除第一帧外的所有Robot0坐标系
        for i in range(1, len(self.list_frame_0)):
            self.vis.remove_geometry(self.list_frame_0[i], reset_bounding_box=False)
        
        # 处理Robot1坐标系
        if self.has_robot1_data:
            for i, frame in enumerate(self.list_frame_1):
                self.vis.add_geometry(frame)
                self.displayed_frames_1.append(frame)
                
            # 隐藏除第一帧外的所有Robot1坐标系
            for i in range(1, len(self.list_frame_1)):
                self.vis.remove_geometry(self.list_frame_1[i], reset_bounding_box=False)
        
        # 更新当前显示状态
        self.current_visible_frame_0 = 0
        self.current_visible_frame_1 = 0 if self.has_robot1_data else -1
        
        # 显示第一帧信息
        self.update_visualization_info()
    
    def clear_episode_geometries(self):
        """清除当前episode的所有几何体"""
        # 清除轨迹线和点云
        try:
            self.vis.remove_geometry(self.trajectory_line_0, reset_bounding_box=False)
            self.vis.remove_geometry(self.trajectory_cloud_0, reset_bounding_box=False)
            if self.trajectory_line_1 is not None:
                self.vis.remove_geometry(self.trajectory_line_1, reset_bounding_box=False)
            if self.trajectory_cloud_1 is not None:
                self.vis.remove_geometry(self.trajectory_cloud_1, reset_bounding_box=False)
        except:
            pass
        
        # 清除所有显示的坐标系
        if hasattr(self, 'displayed_frames_0'):
            for frame in self.displayed_frames_0:
                try:
                    self.vis.remove_geometry(frame, reset_bounding_box=False)
                except:
                    pass
            self.displayed_frames_0.clear()
            
        if hasattr(self, 'displayed_frames_1'):
            for frame in self.displayed_frames_1:
                try:
                    self.vis.remove_geometry(frame, reset_bounding_box=False)
                except:
                    pass
            self.displayed_frames_1.clear()
        
        # 重置当前可见帧标记
        self.current_visible_frame_0 = -1
        self.current_visible_frame_1 = -1
    
    def reload_episode_visualization(self):
        """重新加载episode的可视化"""
        # 清除旧的几何体
        self.clear_episode_geometries()
        
        # 加载新episode数据
        self.load_current_episode()
        
        # 重新初始化可视化
        self.init_visualization()
        
        # 更新视图以确保正确渲染
        self.vis.update_renderer()
    
    def update_visualization(self):
        """更新当前帧的可视化"""
        # 更新坐标系显示
        self.update_frame_display()
        
        # 更新信息显示
        self.update_visualization_info()
        
        # 更新视图（不重置视角）
        self.vis.update_renderer()
    
    def update_frame_display(self):
        """更新坐标系显示，不影响视角"""
        # 更新Robot0坐标系
        if self.current_frame_idx < len(self.list_frame_0):
            if hasattr(self, 'current_visible_frame_0') and self.current_visible_frame_0 != self.current_frame_idx:
                if self.current_visible_frame_0 < len(self.list_frame_0):
                    self.vis.remove_geometry(self.list_frame_0[self.current_visible_frame_0], reset_bounding_box=False)
            
            if self.current_frame_idx != getattr(self, 'current_visible_frame_0', -1):
                self.vis.add_geometry(self.list_frame_0[self.current_frame_idx], reset_bounding_box=False)
                self.current_visible_frame_0 = self.current_frame_idx
        
        # 更新Robot1坐标系
        if self.has_robot1_data and self.current_frame_idx < len(self.list_frame_1):
            if hasattr(self, 'current_visible_frame_1') and self.current_visible_frame_1 != self.current_frame_idx:
                if self.current_visible_frame_1 < len(self.list_frame_1):
                    self.vis.remove_geometry(self.list_frame_1[self.current_visible_frame_1], reset_bounding_box=False)
            
            if self.current_frame_idx != getattr(self, 'current_visible_frame_1', -1):
                self.vis.add_geometry(self.list_frame_1[self.current_frame_idx], reset_bounding_box=False)
                self.current_visible_frame_1 = self.current_frame_idx
    
    def update_visualization_info(self):
        """只更新信息显示，不改变3D视图"""
        episode_id = self.available_episodes[self.current_episode_idx]
        episode_info = f"Episode ID: {episode_id} (第{self.current_episode_idx + 1}个/{len(self.available_episodes)})"     
        
        # 获取当前帧的pose信息
        current_pos_0 = self.pos_list_0[self.current_frame_idx][:3, 3]
        pose_info = f" | Robot0 Pose: [{current_pos_0[0]:.3f}, {current_pos_0[1]:.3f}, {current_pos_0[2]:.3f}]"
        
        if self.has_robot1_data and self.current_frame_idx < len(self.pos_list_1):
            current_pos_1 = self.pos_list_1[self.current_frame_idx][:3, 3]
            pose_info += f" | Robot1 Pose: [{current_pos_1[0]:.3f}, {current_pos_1[1]:.3f}, {current_pos_1[2]:.3f}]"
        
        # 获取当前夹爪宽度
        gripper_width_text = ""
        if self.has_gripper0_data and self.current_frame_idx < len(self.gripper_widths_0):
            gripper_width_0 = self.gripper_widths_0[self.current_frame_idx]
            gripper_width_text += f" | Robot0夹爪: {gripper_width_0:.4f}m"
        if self.has_gripper1_data and self.current_frame_idx < len(self.gripper_widths_1):
            gripper_width_1 = self.gripper_widths_1[self.current_frame_idx]
            gripper_width_text += f" | Robot1夹爪: {gripper_width_1:.4f}m"
        
        # 在控制台打印当前信息
        max_frames = max(len(self.pos_list_0), len(self.pos_list_1) if self.has_robot1_data else 0)
        frame_info = f"帧: {self.current_frame_idx}/{max_frames-1}"
        print(f"{episode_info} | {frame_info}{pose_info}{gripper_width_text}")
        
        # 显示相机图像
        self.display_camera_images()
    
    def display_camera_images(self):
        """显示拼接的相机图像：Robot0第一排，Robot1第二排"""
        # 准备各种图像
        visual_1 = None
        left_tactile_1 = None  
        right_tactile_1 = None
        visual_0 = None
        left_tactile_0 = None
        right_tactile_0 = None
        
        # 获取Robot1图像 (robot1/camera1)
        if self.has_visual_1 and self.current_frame_idx < len(self.visual_images_1):
            visual_1 = self.visual_images_1[self.current_frame_idx].copy()
        if self.has_left_tactile_1 and self.current_frame_idx < len(self.left_tactile_images_1):
            left_tactile_1 = self.left_tactile_images_1[self.current_frame_idx].copy()
        if self.has_right_tactile_1 and self.current_frame_idx < len(self.right_tactile_images_1):
            right_tactile_1 = self.right_tactile_images_1[self.current_frame_idx].copy()
            
        # 获取Robot0图像 (robot0/camera0)
        if self.has_visual_0 and self.current_frame_idx < len(self.visual_images_0):
            visual_0 = self.visual_images_0[self.current_frame_idx].copy()
        if self.has_left_tactile_0 and self.current_frame_idx < len(self.left_tactile_images_0):
            left_tactile_0 = self.left_tactile_images_0[self.current_frame_idx].copy()
        if self.has_right_tactile_0 and self.current_frame_idx < len(self.right_tactile_images_0):
            right_tactile_0 = self.right_tactile_images_0[self.current_frame_idx].copy()
        
        # 创建拼接图像（Robot0在上，Robot1在下）
        combined_image = create_bimanual_combined_image(
            visual_1, left_tactile_1, right_tactile_1,
            visual_0, left_tactile_0, right_tactile_0
        )
        
        if combined_image is not None:
            # 在图像底部添加信息栏
            img_height, img_width = combined_image.shape[:2]
            info_bar_height = 100
            
            # 创建信息栏 (深灰色背景)
            info_bar = np.zeros((info_bar_height, img_width, 3), dtype=np.uint8)
            info_bar[:, :] = [30, 30, 30]  # 深灰色背景
            
            # 准备信息文本
            episode_id = self.available_episodes[self.current_episode_idx]
            max_frames = max(len(self.pos_list_0), len(self.pos_list_1) if self.has_robot1_data else 0)
            
            info_lines = [
                f"Episode {episode_id} | Frame {self.current_frame_idx}/{max_frames-1}",
            ]
            
            # 添加pose信息
            current_pos_0 = self.pos_list_0[self.current_frame_idx][:3, 3]
            info_lines.append(f"Robot0 Pose: [{current_pos_0[0]:.3f}, {current_pos_0[1]:.3f}, {current_pos_0[2]:.3f}]")
            
            if self.has_robot1_data and self.current_frame_idx < len(self.pos_list_1):
                current_pos_1 = self.pos_list_1[self.current_frame_idx][:3, 3]
                info_lines.append(f"Robot1 Pose: [{current_pos_1[0]:.3f}, {current_pos_1[1]:.3f}, {current_pos_1[2]:.3f}]")
            
            # 添加gripper信息
            if self.has_gripper0_data and self.current_frame_idx < len(self.gripper_widths_0):
                gripper_width_0 = self.gripper_widths_0[self.current_frame_idx]
                info_lines.append(f"Robot0 Gripper: {gripper_width_0:.4f}m")
            if self.has_gripper1_data and self.current_frame_idx < len(self.gripper_widths_1):
                gripper_width_1 = self.gripper_widths_1[self.current_frame_idx]
                info_lines.append(f"Robot1 Gripper: {gripper_width_1:.4f}m")
            
            # 在信息栏上绘制文本
            y_offset = 20
            for line in info_lines:
                cv2.putText(info_bar, line, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                y_offset += 18
            
            # 将信息栏拼接到图像底部
            final_image = np.vstack([combined_image, info_bar])
            
            # 显示最终图像
            cv2.imshow("Bimanual Multi-Camera View", cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
    
    def next_frame(self, vis):
        """下一帧"""
        max_frames = max(len(self.pos_list_0), len(self.pos_list_1) if self.has_robot1_data else 0)
        
        if self.current_frame_idx < max_frames - 1:
            self.current_frame_idx += 1
            self.update_visualization()
        else:
            # 当前episode的最后一帧，尝试进入下一个episode
            if self.current_episode_idx < len(self.available_episodes) - 1:
                print(f"\n到达Episode {self.available_episodes[self.current_episode_idx]}的最后一帧，自动进入下一个episode...")
                self.current_episode_idx += 1
                self.reload_episode_visualization()
            else:
                print(f"\n已到达最后一个episode的最后一帧！")
        return True
    
    def prev_frame(self, vis):
        """上一帧"""
        if self.current_frame_idx > 0:
            self.current_frame_idx -= 1
            self.update_visualization()
        return True
    
    def reset_view(self, vis):
        """重置视图"""
        vis.reset_view_point(True)
        return True
    
    def quit(self, vis):
        """退出"""
        cv2.destroyAllWindows()
        self.vis.destroy_window()
        return False

# 启动交互式可视化
visualizer = BimanualEpisodeVisualizerWithKeyCallback(replay_buffer, available_episodes)