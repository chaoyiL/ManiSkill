import os
import sys
import csv
import cv2
import numpy as np
import open3d as o3d

# Make the repo root the working directory (same pattern as other scripts)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)


def load_ee_poses_from_csv(csv_filepath: str):
    """Load left and right end-effector positions from CSV.

    Expects header with columns:
      id,left_p0,left_p1,left_p2,left_q0..left_q3,right_p0,right_p1,right_p2,right_q0..right_q3
    Returns two numpy arrays of shape (N, 3): left_positions, right_positions
    """
    left_positions = []
    right_positions = []
    with open(csv_filepath, 'r') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Skip empty rows
            if not row.get('id'):
                continue
            try:
                lx = float(row['left_p0']); ly = float(row['left_p1']); lz = float(row['left_p2'])
                rx = float(row['right_p0']); ry = float(row['right_p1']); rz = float(row['right_p2'])
            except Exception:
                # Malformed row, skip
                continue
            left_positions.append([lx, ly, lz])
            right_positions.append([rx, ry, rz])
    if len(left_positions) == 0:
        raise RuntimeError(f"No pose rows found in {csv_filepath}")
    return np.array(left_positions, dtype=float), np.array(right_positions, dtype=float)


def compute_projection_params(all_positions, image_size=(1000, 800), padding_ratio=0.05):
    """Compute transform from world XY to image pixel coordinates.
    Projects using X->width, Y->height, keeps aspect ratio and adds padding.
    """
    width, height = image_size
    xs = all_positions[:, 0]
    ys = all_positions[:, 1]
    min_x, max_x = float(np.min(xs)), float(np.max(xs))
    min_y, max_y = float(np.min(ys)), float(np.max(ys))

    # handle degenerate ranges
    range_x = max(max_x - min_x, 1e-6)
    range_y = max(max_y - min_y, 1e-6)

    # add padding in world units
    pad_x = range_x * padding_ratio
    pad_y = range_y * padding_ratio
    min_x -= pad_x; max_x += pad_x
    min_y -= pad_y; max_y += pad_y
    range_x = max(max_x - min_x, 1e-6)
    range_y = max(max_y - min_y, 1e-6)

    scale_x = (width - 20) / range_x
    scale_y = (height - 20) / range_y
    scale = min(scale_x, scale_y)

    def world_to_pixel(x, y):
        px = int((x - min_x) * scale) + 10
        # flip y for image coordinates
        py = height - (int((y - min_y) * scale) + 10)
        return px, py

    return world_to_pixel, image_size


def draw_frame(img, left_pixels, right_pixels, current_idx):
    """Draw trajectories up to current_idx. left_pixels/right_pixels are lists of (x,y)."""
    # Draw small faded trail for past points and a brighter current point
    n = len(left_pixels)
    # draw polylines
    if current_idx >= 1:
        pts_left = np.array(left_pixels[: current_idx + 1], dtype=np.int32)
        pts_right = np.array(right_pixels[: current_idx + 1], dtype=np.int32)
        if len(pts_left) >= 2:
            cv2.polylines(img, [pts_left], False, (0, 0, 180), 1, lineType=cv2.LINE_AA)   # red-ish line (BGR)
        if len(pts_right) >= 2:
            cv2.polylines(img, [pts_right], False, (0, 180, 0), 1, lineType=cv2.LINE_AA)   # green-ish line

    # draw points
    for i in range(0, current_idx + 1):
        lx, ly = left_pixels[i]
        rx, ry = right_pixels[i]
        # older points smaller and dimmer
        radius = 3
        cv2.circle(img, (lx, ly), radius, (0, 0, 200), -1)   # red (B,G,R)
        cv2.circle(img, (rx, ry), radius, (0, 200, 0), -1)   # green

    # highlight current point
    cx_l = left_pixels[current_idx]; cx_r = right_pixels[current_idx]
    cv2.circle(img, tuple(cx_l), 6, (0, 0, 255), 2)
    cv2.circle(img, tuple(cx_r), 6, (0, 255, 0), 2)

    return img


def run_visualizer(csv_path: str):
    left_positions, right_positions = load_ee_poses_from_csv(csv_path)
    n = min(len(left_positions), len(right_positions))
    left_positions = left_positions[:n]
    right_positions = right_positions[:n]

    class EE3DVisualizer:
        def __init__(self, left_pts, right_pts):
            self.left_pts = left_pts
            self.right_pts = right_pts
            self.n = len(left_pts)
            self.current_idx = 0

            # Create visualizer
            self.vis = o3d.visualization.VisualizerWithKeyCallback()
            self.vis.create_window("EE 3D Trajectory (A/Left prev, D/Right next, R reset, Q quit)", width=1280, height=720)

            # Initial pointclouds (first point)
            self.left_pc = o3d.geometry.PointCloud()
            self.right_pc = o3d.geometry.PointCloud()
            self.left_pc.points = o3d.utility.Vector3dVector(self.left_pts[:1])
            self.right_pc.points = o3d.utility.Vector3dVector(self.right_pts[:1])
            self.left_pc.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.0, 0.0], (1, 1)))   # red
            self.right_pc.colors = o3d.utility.Vector3dVector(np.tile([0.0, 1.0, 0.0], (1, 1)))  # green

            # Add geometries
            self.vis.add_geometry(self.left_pc)
            self.vis.add_geometry(self.right_pc)

            # current-frame coordinate frames
            self.current_left_frame = None
            self.current_right_frame = None
            self._add_current_frames()

            # Register callbacks
            self.vis.register_key_callback(ord('D'), self.next_frame)
            self.vis.register_key_callback(ord('d'), self.next_frame)
            self.vis.register_key_callback(ord('A'), self.prev_frame)
            self.vis.register_key_callback(ord('a'), self.prev_frame)
            self.vis.register_key_callback(ord('R'), self.reset_view)
            self.vis.register_key_callback(ord('r'), self.reset_view)
            self.vis.register_key_callback(ord('Q'), self.quit)
            self.vis.register_key_callback(ord('q'), self.quit)

            # Also support left/right arrow key codes (may vary by platform)
            try:
                # 81 left, 83 right commonly reported
                self.vis.register_key_callback(83, self.next_frame)
                self.vis.register_key_callback(81, self.prev_frame)
            except Exception:
                pass

            # Run visualization (blocking)
            self.vis.run()
            self.vis.destroy_window()

        def _add_current_frames(self):
            # remove old frames
            if self.current_left_frame is not None:
                try:
                    self.vis.remove_geometry(self.current_left_frame, reset_bounding_box=False)
                except Exception:
                    pass
            if self.current_right_frame is not None:
                try:
                    self.vis.remove_geometry(self.current_right_frame, reset_bounding_box=False)
                except Exception:
                    pass

            # create new frames at current positions
            left_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
            right_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.03)
            left_frame.translate(self.left_pts[self.current_idx])
            right_frame.translate(self.right_pts[self.current_idx])
            self.current_left_frame = left_frame
            self.current_right_frame = right_frame
            self.vis.add_geometry(self.current_left_frame)
            self.vis.add_geometry(self.current_right_frame)

        def _update_pointclouds(self):
            # update pointclouds to show trajectory up to current_idx
            left_subset = self.left_pts[: self.current_idx + 1]
            right_subset = self.right_pts[: self.current_idx + 1]
            self.left_pc.points = o3d.utility.Vector3dVector(left_subset)
            self.right_pc.points = o3d.utility.Vector3dVector(right_subset)
            self.left_pc.colors = o3d.utility.Vector3dVector(np.tile([1.0, 0.0, 0.0], (len(left_subset), 1)))
            self.right_pc.colors = o3d.utility.Vector3dVector(np.tile([0.0, 1.0, 0.0], (len(right_subset), 1)))
            self.vis.update_geometry(self.left_pc)
            self.vis.update_geometry(self.right_pc)

        def next_frame(self, vis):
            if self.current_idx < self.n - 1:
                self.current_idx += 1
                self._update_pointclouds()
                self._add_current_frames()
                self.vis.update_renderer()
            return True

        def prev_frame(self, vis):
            if self.current_idx > 0:
                self.current_idx -= 1
                self._update_pointclouds()
                self._add_current_frames()
                self.vis.update_renderer()
            return True

        def reset_view(self, vis):
            vis.reset_view_point(True)
            return True

        def quit(self, vis):
            # stop the visualizer
            return False

    # start the Open3D visualizer
    EE3DVisualizer(left_positions, right_positions)


if __name__ == "__main__":
    csv_path = os.path.join(ROOT_DIR, "Data_collection", "check_data_quality", "ee_poses.csv")
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        sys.exit(1)
    run_visualizer(csv_path)


