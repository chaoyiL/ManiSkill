# 🚀 真实环境语言标注 - 快速开始

## 一句话总结
用真实拍摄的物体照片替代 3D 模型，其他流程基本不变。

---

## 📸 准备工作

### 1. 拍摄物体照片
```bash
# 创建目录
mkdir -p real_images/bottle
mkdir -p real_images/hammer

# 拍摄要求：
# ✓ 纯色背景
# ✓ 光照均匀
# ✓ 物体居中
# ✓ 分辨率 ≥ 1024x1024
```

---

## ⚡ 三步生成

### 步骤 1：生成物体描述
```bash
cd Data_collection/description

# 单张图片
bash gen_object_descriptions_real.sh bottle ./real_images/bottle/001.jpg

# 批量处理（推荐）
python utils/batch_generate_real_objects.py ../../real_images/
```

**输出**: `objects_description_real/bottle/001.json`

---

### 步骤 2：修改 scene_info.json

**关键修改**：物体引用路径

```diff
{
  "episode_0000000": {
    "info": {
-     "{A}": "001_bottle/base0",  // 仿真格式
+     "{A}": "bottle/001",         // 真实格式 ✨
      "{a}": "left"
    }
  }
}
```

**位置**: `data/<task_name>/<setting>/scene_info.json`

---

### 步骤 3：生成场景指令

**一次性修改**（只需做一次）：

编辑 `utils/generate_episode_instructions.py`：

```python
# 第 59-61 行
json_path = os.path.join(
    os.path.join(parent_directory, "../objects_description_real"),  # 改这里 ✨
    value + ".json",
)

# 第 108-110 行（同样修改）
json_path = os.path.join(
    os.path.join(parent_directory, "../objects_description_real"),  # 改这里 ✨
    value + ".json"
)
```

然后运行：
```bash
bash gen_episode_instructions.sh adjust_bottle train 100
```

**输出**: `data/adjust_bottle/train/instructions/episode*.json`

---

## 🎯 完整示例

假设你在做 `adjust_bottle` 任务：

```bash
# 1. 准备图片（3 个瓶子）
ls real_images/bottle/
# 001.jpg  002.jpg  003.jpg

# 2. 生成物体描述
cd Data_collection/description
python utils/batch_generate_real_objects.py ../../real_images/

# 3. 修改 scene_info.json
# 编辑: data/adjust_bottle/train/scene_info.json
# 改为: "bottle/001" 而不是 "001_bottle/base0"

# 4. 修改 generate_episode_instructions.py
# 改路径为: ../objects_description_real

# 5. 生成场景指令
bash gen_episode_instructions.sh adjust_bottle train 100

# 6. 检查结果
cat data/adjust_bottle/train/instructions/episode0.json | jq '.seen[0]'
# "Pick up the red plastic bottle from the table head-up."
```

---

## 🔍 验证检查清单

- [ ] 图片已拍摄并放入 `real_images/`
- [ ] 物体描述生成在 `objects_description_real/`
- [ ] `scene_info.json` 物体引用格式已修改
- [ ] `generate_episode_instructions.py` 路径已修改
- [ ] 场景指令生成在 `data/.../instructions/`

---

## ⚠️ 常见问题

### Q1: 报错 "description file does not exist"
**原因**: `scene_info.json` 中的物体引用路径不正确

**解决**: 确保路径格式为 `物体名/编号`（如 `bottle/001`），而不是 `001_bottle/base0`

---

### Q2: 生成的描述不理想
**解决**: 
1. 改善照片质量（光照、背景、清晰度）
2. 手动编辑 JSON 文件的 `seen` 和 `unseen` 列表

---

### Q3: 如何处理多个相同物体
**方案**: 拍摄多张照片，用编号区分
```
bottle/001.jpg → bottle/001.json
bottle/002.jpg → bottle/002.json
bottle/003.jpg → bottle/003.json
```

在 `scene_info.json` 中引用：
```json
"episode_0": {"info": {"{A}": "bottle/001"}},
"episode_1": {"info": {"{A}": "bottle/002"}},
"episode_2": {"info": {"{A}": "bottle/003"}}
```

---

## 🆚 对比表

| 项目 | 仿真环境 | 真实环境 |
|------|---------|---------|
| 物体输入 | `.glb` 文件 | `.jpg/.png` 照片 |
| 生成脚本 | `generate_object_description.py` | `generate_object_description_real.py` |
| 描述路径 | `objects_description/` | `objects_description_real/` |
| scene_info | `"001_bottle/base0"` | `"bottle/001"` |
| 其他步骤 | 相同 | 相同 |

---

## 📚 进阶

### 自动化脚本
使用提供的 `example_real_workflow.sh`：
```bash
bash example_real_workflow.sh
```

### 混合使用
可以同时使用仿真和真实环境的描述：
- 仿真物体：`"001_bottle/base0"` → `objects_description/`
- 真实物体：`"bottle/001"` → `objects_description_real/`

只需在 `scene_info.json` 中灵活配置即可！

---

## 📖 详细文档
- [完整真实环境指南](./README_REAL_ENV.md)
- [原始系统说明](./README.md)

---

**有问题？** 查看 `README_REAL_ENV.md` 或提 Issue

