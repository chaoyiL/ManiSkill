"""
批量处理真实环境物体图片
支持按文件夹结构自动识别物体类别
"""
import os
import argparse
from pathlib import Path
from generate_object_description_real import generate_obj_description_from_image


def batch_generate_from_directory(images_dir, save_dir="./objects_description_real"):
    """
    批量处理图片目录
    
    支持两种目录结构：
    1. 扁平结构：
       images_dir/
         ├── bottle_001.jpg
         ├── bottle_002.jpg
         ├── hammer_001.jpg
         └── ...
       物体名称从文件名中提取（下划线前的部分）
    
    2. 分层结构：
       images_dir/
         ├── bottle/
         │   ├── 001.jpg
         │   ├── 002.jpg
         ├── hammer/
         │   ├── 001.jpg
         └── ...
       物体名称从文件夹名称中提取
    """
    images_dir = Path(images_dir)
    
    if not images_dir.exists():
        print(f"ERROR: Directory '{images_dir}' does not exist.")
        return
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    # 检测目录结构
    subdirs = [d for d in images_dir.iterdir() if d.is_dir()]
    
    if subdirs:
        # 分层结构：每个子文件夹是一个物体类别
        print("Detected hierarchical structure (object_name/images)")
        for subdir in sorted(subdirs):
            object_name = subdir.name
            image_files = [f for f in subdir.iterdir() 
                          if f.is_file() and f.suffix.lower() in image_extensions]
            
            if not image_files:
                print(f"No images found in {subdir}")
                continue
            
            print(f"\n{'='*60}")
            print(f"Processing object: {object_name} ({len(image_files)} images)")
            print(f"{'='*60}")
            
            for image_file in sorted(image_files):
                print(f"\n[{image_files.index(image_file)+1}/{len(image_files)}] Processing {image_file.name}...")
                try:
                    generate_obj_description_from_image(
                        object_name=object_name,
                        image_path=str(image_file),
                        save_dir=save_dir
                    )
                except Exception as e:
                    print(f"ERROR processing {image_file}: {e}")
                    continue
    else:
        # 扁平结构：从文件名提取物体类别
        print("Detected flat structure (object_name_xxx.ext)")
        image_files = [f for f in images_dir.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"No images found in {images_dir}")
            return
        
        print(f"Found {len(image_files)} images")
        
        for image_file in sorted(image_files):
            # 从文件名提取物体名称（假设格式为 objectname_xxx.ext）
            filename_parts = image_file.stem.split('_')
            if len(filename_parts) < 2:
                print(f"WARNING: Cannot extract object name from '{image_file.name}', skipping...")
                continue
            
            object_name = '_'.join(filename_parts[:-1])  # 最后一个下划线前的部分
            
            print(f"\nProcessing {image_file.name} (object: {object_name})...")
            try:
                generate_obj_description_from_image(
                    object_name=object_name,
                    image_path=str(image_file),
                    save_dir=save_dir
                )
            except Exception as e:
                print(f"ERROR processing {image_file}: {e}")
                continue
    
    print(f"\n{'='*60}")
    print(f"Batch processing completed!")
    print(f"Results saved to: {save_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch generate object descriptions from real-world images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  1. Hierarchical structure:
     python batch_generate_real_objects.py ./real_images/
     
     Directory structure:
       real_images/
         ├── bottle/
         │   ├── 001.jpg
         │   ├── 002.jpg
         ├── hammer/
         │   ├── 001.jpg
  
  2. Flat structure:
     python batch_generate_real_objects.py ./real_images/
     
     Directory structure:
       real_images/
         ├── bottle_001.jpg
         ├── bottle_002.jpg
         ├── hammer_001.jpg
        """
    )
    parser.add_argument("images_dir", type=str, help="Directory containing object images")
    parser.add_argument("--save_dir", type=str, default="./objects_description_real",
                       help="Directory to save generated descriptions (default: ./objects_description_real)")
    
    args = parser.parse_args()
    batch_generate_from_directory(args.images_dir, args.save_dir)

