import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)  # 抑制警告
import sys
import os
import cv2
import zarr

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from utils.replay_buffer import ReplayBuffer
from utils.imagecodecs_numcodecs import register_codecs
register_codecs()

# 传感器配置
ROBOT_IDS = [0, 1]
SENSOR_KEYS = {
    'visual': 'camera{}_rgb',
    'left_tactile': 'camera{}_left_tactile',
    'right_tactile': 'camera{}_right_tactile',
    'left_pc': 'camera{}_left_tactile_points',
    'right_pc': 'camera{}_right_tactile_points',
}

def get_transform(pos, rot_axis_angle):
    """获取变换矩阵"""
    rotation_matrix, _ = cv2.Rodrigues(rot_axis_angle)
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = pos
    return T

def decode_image(img_data):
    """统一的图像解码"""
    if isinstance(img_data, bytes):
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_data

def load_pointcloud(points_data):
    """统一的点云加载和过滤"""
    if points_data is None or len(points_data) == 0:
        return np.empty((0, 3), dtype=np.float32)
    points = np.array(points_data, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        return np.empty((0, 3), dtype=np.float32)
    mask = np.any(points != 0, axis=1)
    return points[mask] if np.any(mask) else np.empty((0, 3), dtype=np.float32)

def calc_camera_params(points):
    """计算点云的相机参数"""
    if len(points) == 0:
        return {'center': [0, 0, 40], 'eye': [50, -30, 80], 'up': [0, 0, 1]}
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    center = pcd.get_center()
    extent = pcd.get_axis_aligned_bounding_box().get_extent()
    dist = np.max(extent) * 2.5
    eye = center + np.array([dist * 0.5, -dist * 0.3, dist * 0.8])
    return {'center': center.tolist(), 'eye': eye.tolist(), 'up': [0, 0, 1]}

def resize_with_label(img, label, height, color=(255, 255, 255)):
    """调整图像大小并添加标签"""
    if img is None:
        return None
    h, w = img.shape[:2]
    resized = cv2.resize(img, (int(w * height / h), height))
    cv2.putText(resized, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(resized, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return resized

def hstack_with_sep(images, sep_width=2, sep_color=255):
    """水平拼接图像，带分隔线"""
    valid = [img for img in images if img is not None]
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    h = valid[0].shape[0]
    sep = np.ones((h, sep_width, 3), dtype=np.uint8) * sep_color
    parts = []
    for i, img in enumerate(valid):
        parts.append(img)
        if i < len(valid) - 1:
            parts.append(sep)
    return np.hstack(parts)

def load_episode_data(replay_buffer, episode_idx):
    """加载episode数据"""
    ep_slice = replay_buffer.get_episode_slice(episode_idx)
    data = {f'robot{r}': {
        'poses': [], 'gripper': [],
        'visual': [], 'left_tactile': [], 'right_tactile': [],
        'left_pc': [], 'right_pc': []
    } for r in ROBOT_IDS}
    
    first_tx = {0: None, 1: None}
    keys = replay_buffer.keys()
    
    # 检查可用数据
    has = {f'robot{r}_{s}': SENSOR_KEYS[s].format(r) in keys 
           for r in ROBOT_IDS for s in SENSOR_KEYS}
    has['robot1_pose'] = 'robot1_eef_pos' in keys
    has['robot0_gripper'] = 'robot0_gripper_width' in keys
    has['robot1_gripper'] = 'robot1_gripper_width' in keys
    
    for i in range(ep_slice.start, ep_slice.stop):
        for r in ROBOT_IDS:
            prefix = f'robot{r}'
            cam_id = r
            
            # 加载位姿
            if r == 0 or has['robot1_pose']:
                pos = replay_buffer[f'robot{r}_eef_pos'][i]
                rot = replay_buffer[f'robot{r}_eef_rot_axis_angle'][i]
                tx = get_transform(pos, rot)
                if first_tx[r] is None:
                    first_tx[r] = tx.copy()
                data[prefix]['poses'].append(np.linalg.inv(first_tx[r]) @ tx)
            
            # 加载夹爪
            if has[f'{prefix}_gripper']:
                data[prefix]['gripper'].append(replay_buffer[f'{prefix}_gripper_width'][i][0])
            
            # 加载图像和点云
            for sensor, key_template in SENSOR_KEYS.items():
                key = key_template.format(cam_id)
                if key not in keys:
                    continue
                raw = replay_buffer[key][i]
                if 'pc' in sensor:
                    data[prefix][sensor].append(load_pointcloud(raw))
                else:
                    data[prefix][sensor].append(decode_image(raw))
    
    return data, has

def create_combined_image(data, frame_idx, pc_images, traj_images, height=250):
    """创建组合图像"""
    rows = []
    colors = {0: (0, 0, 255), 1: (0, 255, 0)}  # robot0红色, robot1绿色
    
    for r in ROBOT_IDS:
        prefix = f'robot{r}'
        row_parts = []
        
        # 轨迹图
        if prefix in traj_images:
            row_parts.append(resize_with_label(traj_images[prefix], f"R{r} Traj", height, colors[r]))
        
        # 点云图
        pc_imgs = []
        for side in ['left', 'right']:
            key = f'{prefix}_{side}_pc'
            if key in pc_images:
                pc_imgs.append(resize_with_label(pc_images[key], f"R{r} {side.title()} PC", height, 
                                                 (255, 255, 0) if side == 'left' else (0, 255, 255)))
        if pc_imgs:
            row_parts.append(hstack_with_sep(pc_imgs))
        
        # 相机图像
        cam_imgs = []
        for sensor, label in [('visual', 'Visual'), ('left_tactile', 'L-Tact'), ('right_tactile', 'R-Tact')]:
            imgs = data[prefix].get(sensor, [])
            if imgs and frame_idx < len(imgs):
                cam_imgs.append(resize_with_label(imgs[frame_idx].copy(), f"R{r} {label}", height, colors[r]))
        if cam_imgs:
            row_parts.append(hstack_with_sep(cam_imgs))
        
        if row_parts:
            rows.append(hstack_with_sep(row_parts, sep_width=5, sep_color=128))
    
    if not rows:
        return None
    
    # 对齐宽度
    max_w = max(r.shape[1] for r in rows)
    aligned = []
    for row in rows:
        if row.shape[1] < max_w:
            pad = np.zeros((height, max_w - row.shape[1], 3), dtype=np.uint8)
            row = np.hstack([row, pad])
        aligned.append(row)
    
    sep = np.ones((5, max_w, 3), dtype=np.uint8) * 128
    return np.vstack([aligned[0], sep, aligned[1]]) if len(aligned) > 1 else aligned[0]

class CombinedVisualizer:
    def __init__(self, replay_buffer, episodes, record_mode=False, record_episode=0, 
                 output_video=None, record_fps=30, continue_after_record=False):
        self.rb = replay_buffer
        self.episodes = episodes
        self.ep_idx = record_episode if record_mode else 0
        self.frame_idx = 0
        self.record_mode = record_mode
        self.output_video = output_video
        self.record_fps = record_fps
        self.continue_after_record = continue_after_record
        
        self.load_episode()
        self.setup_renderers()
        
        if record_mode:
            self.record_episode()
            if not continue_after_record:
                print("✅ 录制完成，退出程序")
                return
        
        self.print_help()
        self.run()
    
    def load_episode(self):
        """加载当前episode"""
        ep_id = self.episodes[self.ep_idx]
        self.data, self.has = load_episode_data(self.rb, ep_id)
        self.frame_idx = 0
        self.setup_camera_params()
        print(f"✅ 加载 Episode {ep_id} ({self.ep_idx + 1}/{len(self.episodes)}), 帧数: {len(self.data['robot0']['poses'])}")
    
    def setup_camera_params(self):
        """设置点云相机参数"""
        self.cam_params = {}
        for r in ROBOT_IDS:
            for side in ['left', 'right']:
                key = f'robot{r}_{side}_pc'
                pcs = self.data[f'robot{r}'][f'{side}_pc']
                # 找第一个非空点云设置相机
                params = None
                for pc in pcs:
                    if len(pc) > 0:
                        params = calc_camera_params(pc)
                        break
                self.cam_params[key] = params or calc_camera_params(np.empty((0, 3)))
    
    def setup_renderers(self):
        """初始化渲染器"""
        self.render_size = (400, 300)
        self.renderers = {}
        
        # 点云渲染器
        for r in ROBOT_IDS:
            for side in ['left', 'right']:
                self.renderers[f'robot{r}_{side}_pc'] = self._create_renderer()
        
        # 轨迹渲染器
        for r in ROBOT_IDS:
            self.renderers[f'robot{r}_traj'] = self._create_renderer()
    
    def _create_renderer(self):
        """创建单个渲染器"""
        vis = o3d.visualization.rendering.OffscreenRenderer(*self.render_size)
        vis.scene.set_background([0.1, 0.1, 0.1, 1.0])
        vis.scene.scene.set_sun_light([0, -1, -1], [1.0, 1.0, 1.0], 50000)
        vis.scene.scene.enable_sun_light(True)
        return vis
    
    def render_pointcloud(self, points, renderer, cam_params):
        """渲染点云"""
        renderer.scene.clear_geometry()
        
        if len(points) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
            pcd.colors = o3d.utility.Vector3dVector(np.tile([1.0, 1.0, 0.0], (len(points), 1)))
            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = 'defaultUnlit'
            mat.point_size = 8.0
            renderer.scene.add_geometry("pc", pcd, mat)
        
        coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
        coord_mat = o3d.visualization.rendering.MaterialRecord()
        coord_mat.shader = 'defaultLit'
        renderer.scene.add_geometry("coord", coord, coord_mat)
        renderer.setup_camera(60.0, cam_params['center'], cam_params['eye'], cam_params['up'])
        return np.asarray(renderer.render_to_image())
    
    def render_trajectory(self, poses, current_idx, renderer, color):
        """渲染轨迹"""
        renderer.scene.clear_geometry()
        
        if not poses or current_idx >= len(poses):
            renderer.setup_camera(60.0, [0, 0, 0], [0.5, -0.3, 0.8], [0, 0, 1])
            return np.asarray(renderer.render_to_image())
        
        pts = np.array([p[:3, 3] for p in poses[:current_idx + 1]], dtype=np.float64)
        
        # 线条材质
        line_mat = o3d.visualization.rendering.MaterialRecord()
        line_mat.shader = 'unlitLine'
        line_mat.line_width = 3.0
        
        # 点材质
        point_mat = o3d.visualization.rendering.MaterialRecord()
        point_mat.shader = 'defaultUnlit'
        point_mat.point_size = 8.0
        
        # 坐标系材质
        mesh_mat = o3d.visualization.rendering.MaterialRecord()
        mesh_mat.shader = 'defaultLit'
        
        if len(pts) > 1:
            lines = [[i, i + 1] for i in range(len(pts) - 1)]
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(pts)
            ls.lines = o3d.utility.Vector2iVector(lines)
            ls.colors = o3d.utility.Vector3dVector(np.array([color] * len(lines), dtype=np.float64))
            renderer.scene.add_geometry("lines", ls, line_mat)
        
        if len(pts) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(np.array([color] * len(pts), dtype=np.float64))
            renderer.scene.add_geometry("pts", pcd, point_mat)
        
        # 当前帧坐标系
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        frame.transform(poses[current_idx])
        renderer.scene.add_geometry("frame", frame, mesh_mat)
        
        # 原点坐标系
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
        renderer.scene.add_geometry("origin", origin, mesh_mat)
        
        # 计算相机
        center = pts.mean(axis=0)
        extent = pts.max(axis=0) - pts.min(axis=0)
        dist = max(np.max(extent) * 2.0, 0.3)
        eye = center + np.array([dist * 0.6, -dist * 0.4, dist * 0.8])
        renderer.setup_camera(60.0, center.tolist(), eye.tolist(), [0, 0, 1])
        
        return np.asarray(renderer.render_to_image())
    
    def get_frame_images(self):
        """获取当前帧所有渲染图像"""
        pc_images, traj_images = {}, {}
        colors = {0: [1, 0, 0], 1: [0, 1, 0]}
        
        for r in ROBOT_IDS:
            prefix = f'robot{r}'
            
            # 点云
            for side in ['left', 'right']:
                key = f'{prefix}_{side}_pc'
                pcs = self.data[prefix].get(f'{side}_pc', [])
                if pcs and self.frame_idx < len(pcs):
                    pc_images[key] = self.render_pointcloud(
                        pcs[self.frame_idx], self.renderers[key], self.cam_params[key])
            
            # 轨迹
            poses = self.data[prefix].get('poses', [])
            if poses:
                traj_images[prefix] = self.render_trajectory(
                    poses, self.frame_idx, self.renderers[f'{prefix}_traj'], colors[r])
        
        return pc_images, traj_images
    
    def create_info_bar(self, frame_idx, width, height=120):
        """创建信息栏"""
        bar = np.zeros((height, width, 3), dtype=np.uint8)
        bar[:] = [30, 30, 30]
        
        ep_id = self.episodes[self.ep_idx]
        max_frames = len(self.data['robot0']['poses'])
        
        lines = [f"Episode {ep_id} ({self.ep_idx + 1}/{len(self.episodes)}) | Frame {frame_idx}/{max_frames - 1}"]
        
        for r in ROBOT_IDS:
            prefix = f'robot{r}'
            poses = self.data[prefix].get('poses', [])
            if poses and frame_idx < len(poses):
                pos = poses[frame_idx][:3, 3]
                lines.append(f"Robot{r} Pose: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
            
            gripper = self.data[prefix].get('gripper', [])
            if gripper and frame_idx < len(gripper):
                lines.append(f"Robot{r} Gripper: {gripper[frame_idx]:.4f}m")
        
        # 点云信息
            pc_info = []
        for r in ROBOT_IDS:
            for side in ['left', 'right']:
                pcs = self.data[f'robot{r}'].get(f'{side}_pc', [])
                if pcs and frame_idx < len(pcs):
                    pc_info.append(f"R{r}-{side[0].upper()}:{len(pcs[frame_idx])}")
        if pc_info:
            lines.append(f"Point Clouds: {' | '.join(pc_info)}")
        
        for i, line in enumerate(lines):
            cv2.putText(bar, line, (10, 20 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        return bar
    
    def render_frame(self, frame_idx):
        """渲染单帧"""
        self.frame_idx = frame_idx
        pc_images, traj_images = self.get_frame_images()
        combined = create_combined_image(self.data, frame_idx, pc_images, traj_images)
        
        if combined is None:
            return None
        
        info_bar = self.create_info_bar(frame_idx, combined.shape[1])
        return np.vstack([combined, info_bar])
    
    def update_display(self):
        """更新显示"""
        frame = self.render_frame(self.frame_idx)
        if frame is not None:
            cv2.imshow("Combined View", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    def record_episode(self):
        """录制当前episode"""
        max_frames = len(self.data['robot0']['poses'])
        print(f"🎬 开始录制 Episode {self.episodes[self.ep_idx]}, 帧数: {max_frames}, FPS: {self.record_fps}")
        
        # 获取第一帧确定尺寸
        first_frame = self.render_frame(0)
        if first_frame is None:
            print("❌ 无法生成帧，录制失败")
            return
        
        h, w = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(self.output_video, fourcc, self.record_fps, (w, h))
        
        if not writer.isOpened():
            print(f"❌ 无法创建视频文件: {self.output_video}")
            return
        
        for i in range(max_frames):
            frame = self.render_frame(i)
            if frame is not None:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if (i + 1) % 10 == 0:
                print(f"   进度: {i + 1}/{max_frames} ({(i + 1) / max_frames * 100:.1f}%)")
        
        writer.release()
        print(f"✅ 录制完成: {self.output_video}")
        self.frame_idx = 0
    
    def print_help(self):
        """打印帮助"""
        print("\n" + "=" * 50)
        print("🎮 控制: A/D=前/后帧  W/S=前/后Episode  R=重置视角  Q=退出")
        print("=" * 50 + "\n")
    
    def run(self):
        """运行可视化循环"""
        while True:
            self.update_display()
            key = cv2.waitKey(30) & 0xFF
            
            if key in [ord('d'), ord('D')]:
                max_f = len(self.data['robot0']['poses'])
                if self.frame_idx < max_f - 1:
                    self.frame_idx += 1
                elif self.ep_idx < len(self.episodes) - 1:
                    self.ep_idx += 1
                    self.load_episode()
            elif key in [ord('a'), ord('A')]:
                if self.frame_idx > 0:
                    self.frame_idx -= 1
            elif key in [ord('w'), ord('W')]:
                if self.ep_idx < len(self.episodes) - 1:
                    self.ep_idx += 1
                    self.load_episode()
            elif key in [ord('s'), ord('S')]:
                if self.ep_idx > 0:
                    self.ep_idx -= 1
                    self.load_episode()
            elif key in [ord('r'), ord('R')]:
                self.setup_camera_params()
                print("📷 重置视角")
            elif key in [ord('q'), ord('Q')]:
                print("👋 退出")
                break
        
        cv2.destroyAllWindows()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Combined visualizer')
    parser.add_argument('zarr_path', nargs='?', 
                       default='/home/rvsa/codehub/VB-vla/data/_0118/_0118.zarr.zip')
    parser.add_argument('--record', type=bool, default=True)
    parser.add_argument('--record_episode', type=int, default=1)
    parser.add_argument('--output_video', type=str, default=None)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--continue_after_record', type=bool, default=True)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.zarr_path):
        print(f"❌ 找不到文件: {args.zarr_path}")
        return
    
    print(f"🔍 加载: {args.zarr_path}")
    
    with zarr.ZipStore(args.zarr_path, mode='r') as store:
        rb = ReplayBuffer.copy_from_store(src_store=store, store=zarr.MemoryStore())
    
    print(f"✅ 加载完成, 帧数: {rb.n_steps}, Episodes: {rb.n_episodes}")
    
    if args.record and args.record_episode >= rb.n_episodes:
        print(f"❌ Episode {args.record_episode} 超出范围 (共 {rb.n_episodes} 个)")
        return
    
    if args.record and args.output_video is None:
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = os.path.basename(args.zarr_path).replace('.zarr.zip', '')
        args.output_video = f"recorded_ep{args.record_episode}_{name}_{ts}.mp4"
    
    CombinedVisualizer(rb, np.arange(rb.n_episodes), args.record, args.record_episode,
                       args.output_video, args.fps, args.continue_after_record)

if __name__ == "__main__":
    main()
