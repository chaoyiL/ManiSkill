# Robot Visualization

A visualization tool for VR dual-arm robot teleoperation data.

## Installation
```bash
git clone https://github.com/Jerryzhang258/robot_visualization.git
cd robot_visualization
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
```bash
# Interactive mode
python src/viz_3d_enhanced.py data/your_data.zarr.zip

# Record video
python src/viz_3d_enhanced.py data/your_data.zarr.zip -r --record_episode 1 --output_video demo.mp4
```

## Controls

- `A/D` - Previous/next frame
- `W/S` - Switch episode
- `P` - Toggle auto-play
- `1-5` - Adjust speed (0.25x, 0.5x, 1x, 2x, 5x)
- `Q` - Quit

## Data Format

Zarr format (.zarr.zip) containing:
- robot0/1_eef_pos (end-effector position)
- robot0/1_gripper_width (gripper aperture)
- robot0/1_visual (camera feed)
- robot0/1_left/right_tactile (tactile sensors)
