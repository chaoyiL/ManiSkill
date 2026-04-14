"""
真实环境物体描述生成脚本
使用真实拍摄的照片代替 3D 模型渲染
"""
import json
from agent import *
from argparse import ArgumentParser
import os
import base64
import time
import random
from pathlib import Path


class subPart(BaseModel):
    name: str
    color: str
    shape: str
    size: str
    material: str
    functionality: str
    texture: str


class ObjDescFormat(BaseModel):
    raw_description: str = Field(description="the name of the object,without index and '_'")
    wholePart: subPart = Field(description="the object as a whole")
    subParts: List[subPart] = Field(
        description="the deformable subparts of the object.If the object is not deformable, leave empty here")
    description: List[str] = Field(description="several different text descriptions describing this same object here")


with open("./_generate_object_prompt.txt", "r") as f:
    system_prompt = f.read()


def save_json(save_dir, image_file_name, ObjDescResult):
    """保存生成的描述到 JSON 文件"""
    os.makedirs(save_dir, exist_ok=True)
    # Remove image extension from the filename
    base_name = Path(image_file_name).stem
    save_path = f"{save_dir}/{base_name}.json"

    # Get all descriptions
    all_descriptions = ObjDescResult.description.copy()
    all_descriptions.sort(key=len)
    
    # Randomly select 3 indices for validation set
    val_indices = random.sample(range(len(all_descriptions)), min(3, len(all_descriptions)))

    # Separate validation and training descriptions based on indices
    shuffle_val = [all_descriptions[i] for i in val_indices]
    shuffle_train = [all_descriptions[i] for i in range(len(all_descriptions)) if i not in val_indices]

    # Sort both validation and training descriptions by character length
    shuffle_val.sort(key=len)
    shuffle_train.sort(key=len)

    # 将字典保存为 JSON 文件
    desc_dict = {
        "raw_description": ObjDescResult.raw_description,
        "seen": shuffle_train,
        "unseen": shuffle_val,
    }
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(desc_dict, file, ensure_ascii=False, indent=4)
        print(json.dumps(desc_dict, indent=2, ensure_ascii=False))


def load_image_as_base64(image_path):
    """读取图片并转换为 base64 编码"""
    with open(image_path, "rb") as f:
        img_data = f.read()
        imgstr = base64.b64encode(img_data).decode("utf-8")
    return imgstr


def make_prompt_generate(imgStr, object_name):
    """调用 VLM 生成物体描述"""
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"THE OBJECT IS A {object_name}"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{imgStr}"
                    },
                },
            ],
        },
    ]
    result = generate(messages, ObjDescFormat)
    result_dict = result.model_dump()
    print(
        json.dumps(
            {
                "wholePart": result_dict["wholePart"],
                "subParts": result_dict["subParts"],
            },
            indent=2,
            ensure_ascii=False,
        ))
    return result


def generate_obj_description_from_image(object_name, image_path, save_dir="./objects_description_real"):
    """
    从真实图片生成物体描述
    
    Args:
        object_name: 物体名称（如 "bottle", "hammer"）
        image_path: 图片路径
        save_dir: 保存目录
    """
    time_start = time.time()
    
    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"ERROR: Image file '{image_path}' does not exist.")
        return
    
    # 读取图片
    print(f"{object_name} loading image from {image_path}")
    imgstr = load_image_as_base64(image_path)
    print(f"{object_name} image loaded, time: {time.time() - time_start:.2f}s")
    
    # 生成描述
    time_start = time.time()
    print(f"{object_name} start generating descriptions...")
    result = make_prompt_generate(imgstr, object_name)
    print(f"{object_name} generated {len(result.description)} descriptions, time: {time.time() - time_start:.2f}s")
    
    # 保存结果
    image_filename = os.path.basename(image_path)
    obj_save_dir = os.path.join(save_dir, object_name)
    save_json(obj_save_dir, image_filename, result)
    print(f"Saved to {obj_save_dir}/{Path(image_filename).stem}.json")


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate object descriptions from real-world images")
    parser.add_argument("object_name", type=str, help="Object name (e.g., 'bottle', 'hammer')")
    parser.add_argument("image_path", type=str, help="Path to the object image")
    parser.add_argument("--save_dir", type=str, default="./objects_description_real", 
                        help="Directory to save generated descriptions")
    
    args = parser.parse_args()
    
    generate_obj_description_from_image(
        object_name=args.object_name,
        image_path=args.image_path,
        save_dir=args.save_dir
    )

