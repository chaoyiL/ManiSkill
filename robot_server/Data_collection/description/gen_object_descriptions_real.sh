#!/bin/bash
# 真实环境物体描述生成脚本

object_name=${1}
image_path=${2}

# 检查是否提供了足够的参数
if [ -z "$object_name" ] || [ -z "$image_path" ]; then
    echo "Error: Both object_name and image_path are required."
    echo "Usage: $0 <object_name> <image_path>"
    echo ""
    echo "Example:"
    echo "  $0 bottle ./real_images/bottle_001.jpg"
    exit 1
fi

# 检查图片文件是否存在
if [ ! -f "$image_path" ]; then
    echo "Error: Image file '$image_path' does not exist."
    exit 1
fi

python utils/generate_object_description_real.py "$object_name" "$image_path"

