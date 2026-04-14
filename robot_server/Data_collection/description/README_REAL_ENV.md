# 真实环境语言标注生成指南

本指南说明如何在**真实机器人数据采集**环境中使用语言标注生成系统。

## 📋 与仿真环境的区别

| 项目 | 仿真环境 | 真实环境 |
|------|---------|---------|
| 物体表示 | 3D 模型 (.glb) | 真实照片 (.jpg/.png) |
| 图片来源 | 渲染生成 | 相机拍摄 |
| 物体描述生成 | `generate_object_description.py` | `generate_object_description_real.py` ✨ |
| 任务指令模板 | 相同 | 相同 |
| 场景指令生成 | 相同 | 相同（需修改配置） |

## 🚀 完整工作流程

### **步骤 1：准备物体图片**

#### 拍摄要求：
1. **背景简洁**：纯色背景（白色/灰色）最佳
2. **光照均匀**：避免强烈阴影和反光
3. **物体居中**：物体占据画面 60-80%
4. **角度多样**：同一物体可拍摄多张（正面、侧面、俯视）
5. **高分辨率**：建议至少 1024x1024 像素

#### 目录结构（二选一）：

**方案 A：分层结构（推荐）**
```
real_images/
├── bottle/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── 003.jpg
├── hammer/
│   ├── 001.jpg
│   └── 002.jpg
└── block/
    └── 001.jpg
```

**方案 B：扁平结构**
```
real_images/
├── bottle_001.jpg
├── bottle_002.jpg
├── bottle_003.jpg
├── hammer_001.jpg
├── hammer_002.jpg
└── block_001.jpg
```

### **步骤 2：生成物体描述**

#### 单个物体生成：
```bash
cd Data_collection/description

# 为单张图片生成描述
bash gen_object_descriptions_real.sh bottle ./real_images/bottle/001.jpg
```

生成的描述保存在：
```
objects_description_real/
└── bottle/
    └── 001.json  # 包含 seen 和 unseen 描述
```

#### 批量生成（推荐）：
```bash
cd Data_collection/description

# 批量处理整个文件夹
python utils/batch_generate_real_objects.py ./real_images/
```

### **步骤 3：创建任务指令模板**

这部分**与仿真环境完全相同**！

在 `task_instruction/` 下创建 `your_task.json`：

```json
{
  "full_description": "详细描述任务流程，用<>标注关键步骤",
  "schema": "{A} 表示第一个物体, {B} 表示第二个物体, {a} 表示左臂, {b} 表示右臂",
  "preference": "指令长度不超过15个单词",
  "seen": [],
  "unseen": []
}
```

生成指令模板：
```bash
bash gen_task_instruction_templates.sh your_task 12
```

### **步骤 4：修改 scene_info.json**

这是**关键的适配步骤**！你需要修改数据采集生成的 `scene_info.json`，让物体引用指向真实图片的描述。

**原始格式（仿真）：**
```json
{
  "episode_0000000": {
    "info": {
      "{A}": "001_bottle/base0",  // 指向 .glb 文件
      "{a}": "left"
    }
  }
}
```

**修改为（真实环境）：**
```json
{
  "episode_0000000": {
    "info": {
      "{A}": "bottle/001",  // 指向真实图片描述
      "{a}": "left"
    }
  }
}
```

**映射规则：**
- 仿真：`001_bottle/base0` → `objects_description/001_bottle/base0.json`
- 真实：`bottle/001` → `objects_description_real/bottle/001.json`

### **步骤 5：生成完整场景指令**

修改 `gen_episode_instructions.sh`，或创建新版本：

```bash
#!/bin/bash
task_name=${1}
setting=${2}
max_num=${3}

# 修改这里：使用真实环境的物体描述路径
python utils/generate_episode_instructions_real.py $task_name $setting $max_num
```

需要修改 `generate_episode_instructions.py` 中的路径：

```python
# 原始代码（第59-61行）：
json_path = os.path.join(
    os.path.join(parent_directory, "../objects_description"),
    value + ".json",
)

# 修改为：
json_path = os.path.join(
    os.path.join(parent_directory, "../objects_description_real"),  # 改这里
    value + ".json",
)
```

或者更灵活的方式：
```python
# 在第108行附近，同样修改
json_path = os.path.join(
    os.path.join(parent_directory, "../objects_description_real"),  # 改这里
    value + ".json"
)
```

运行生成：
```bash
bash gen_episode_instructions.sh your_task train 100
```

## 🎯 示例：完整流程演示

假设你要为 `adjust_bottle` 任务生成语言标注：

### 1. 准备图片
```bash
mkdir -p real_images/bottle
# 拍摄 3 个不同的瓶子，保存为：
# real_images/bottle/001.jpg
# real_images/bottle/002.jpg
# real_images/bottle/003.jpg
```

### 2. 生成物体描述
```bash
cd Data_collection/description
python utils/batch_generate_real_objects.py ../../real_images/
```

结果：
```
objects_description_real/
└── bottle/
    ├── 001.json  # {"seen": [...], "unseen": [...]}
    ├── 002.json
    └── 003.json
```

### 3. 任务模板已存在
```bash
# adjust_bottle.json 已经存在
ls task_instruction/adjust_bottle.json
```

### 4. 修改 scene_info.json
```bash
# 位置：data/adjust_bottle/train/scene_info.json
# 修改物体引用：
# "001_bottle/base0" → "bottle/001"
```

### 5. 生成场景指令
```bash
# 先修改 generate_episode_instructions.py 的路径（见上文）
bash gen_episode_instructions.sh adjust_bottle train 100
```

生成结果：
```
data/adjust_bottle/train/instructions/
├── episode0.json
├── episode1.json
└── ...
```

每个文件包含：
```json
{
  "seen": [
    "Pick up the red plastic bottle from the table head-up.",
    "Grab the bottle with yellow label using the left arm.",
    ...
  ],
  "unseen": [
    "Use the left arm to grab the shiny red bottle.",
    ...
  ]
}
```

## ⚠️ 注意事项

### 1. 物体命名一致性
确保：
- 图片文件/文件夹名称
- `scene_info.json` 中的物体引用
- 生成的 JSON 描述文件

三者的命名能够正确对应！

### 2. 描述路径配置
如果你想同时使用仿真和真实环境的描述，可以：

```python
# 在 generate_episode_instructions.py 中添加参数
parser.add_argument("--use_real", action="store_true", 
                   help="Use real-world object descriptions")

# 根据参数选择路径
desc_dir = "../objects_description_real" if args.use_real else "../objects_description"
```

### 3. 图片质量要求
- ❌ 避免：模糊、过曝、物体被遮挡
- ✅ 推荐：清晰、光照均匀、物体完整可见

### 4. 描述文件格式
确保生成的 JSON 格式与原系统一致：
```json
{
  "raw_description": "bottle",
  "seen": ["描述1", "描述2", ...],
  "unseen": ["描述A", "描述B", ...]
}
```

## 🔍 调试技巧

### 检查物体描述是否生成：
```bash
ls objects_description_real/bottle/
cat objects_description_real/bottle/001.json | jq '.seen | length'
```

### 测试单个场景指令生成：
```python
from utils.generate_episode_instructions import *

task_data = load_task_instructions("adjust_bottle")
episode_info = {"{A}": "bottle/001", "{a}": "left"}

filtered = filter_instructions(task_data["seen"], episode_info)
print(f"Matched {len(filtered)} instructions")

description = replace_placeholders(filtered[0], episode_info)
print(description)
```

### 验证描述文件路径：
```bash
# 在 generate_episode_instructions.py 中添加调试输出
print(f"Looking for: {json_path}")
print(f"File exists: {os.path.exists(json_path)}")
```

## 📚 进阶：自动化流程

创建一键生成脚本 `generate_all_real.sh`：

```bash
#!/bin/bash
set -e

# 1. 生成物体描述
echo "Step 1: Generating object descriptions..."
python utils/batch_generate_real_objects.py ../../real_images/

# 2. 修改 scene_info.json（如果需要）
echo "Step 2: Update scene_info.json manually if needed"

# 3. 生成场景指令
echo "Step 3: Generating episode instructions..."
for task in adjust_bottle pick_bottle place_bottle; do
    echo "Processing task: $task"
    bash gen_episode_instructions.sh $task train 100
done

echo "All done!"
```

## 🆚 与仿真环境对比

| 特性 | 仿真环境 | 真实环境 |
|------|---------|---------|
| 物体多样性 | 有限（需建模） | 无限（直接拍摄） |
| 描述准确性 | 可能不符合真实外观 | 高度真实 |
| 成本 | 高（3D建模） | 低（相机拍摄） |
| 灵活性 | 低 | 高 |
| 泛化能力 | 待验证 | 更强 |

## ❓ 常见问题

**Q: 一个物体需要拍多少张照片？**  
A: 建议 1-3 张，不同角度/光照条件下。每张会生成独立的描述文件。

**Q: 可以混用仿真和真实环境的描述吗？**  
A: 可以！只需在 `scene_info.json` 中灵活配置物体引用路径。

**Q: VLM 生成的描述不理想怎么办？**  
A: 可以手动编辑生成的 JSON 文件，修改 `seen` 和 `unseen` 列表。

**Q: 如何处理同一类物体的多个实例？**  
A: 拍摄多张图片，用编号区分（如 `bottle/001.jpg`, `bottle/002.jpg`）。

---

**联系方式**: 有问题请提 Issue 或查看主 README.md

