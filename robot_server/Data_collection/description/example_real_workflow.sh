#!/bin/bash
# 真实环境语言标注生成示例流程

set -e  # 出错立即退出

echo "=========================================="
echo "真实环境语言标注生成 - 示例流程"
echo "=========================================="
echo ""

# 颜色输出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ========================================
# 步骤 1：检查图片目录
# ========================================
echo -e "${BLUE}[步骤 1]${NC} 检查图片目录..."

IMAGE_DIR="../../real_images"
if [ ! -d "$IMAGE_DIR" ]; then
    echo -e "${YELLOW}警告: 图片目录不存在，创建示例目录结构...${NC}"
    mkdir -p "$IMAGE_DIR/bottle"
    mkdir -p "$IMAGE_DIR/hammer"
    mkdir -p "$IMAGE_DIR/block"
    echo ""
    echo "请将物体照片放入以下目录："
    echo "  $IMAGE_DIR/bottle/001.jpg"
    echo "  $IMAGE_DIR/bottle/002.jpg"
    echo "  $IMAGE_DIR/hammer/001.jpg"
    echo "  ..."
    echo ""
    echo "拍摄要求："
    echo "  - 背景简洁（纯色背景最佳）"
    echo "  - 光照均匀，无强烈阴影"
    echo "  - 物体居中，占画面 60-80%"
    echo "  - 分辨率至少 1024x1024"
    echo ""
    exit 1
fi

# 统计图片数量
TOTAL_IMAGES=$(find "$IMAGE_DIR" -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.jpeg" \) | wc -l)
echo -e "${GREEN}✓${NC} 发现 $TOTAL_IMAGES 张图片"
echo ""

# ========================================
# 步骤 2：生成物体描述
# ========================================
echo -e "${BLUE}[步骤 2]${NC} 生成物体描述..."
echo "这可能需要几分钟，取决于图片数量和网络速度..."
echo ""

python utils/batch_generate_real_objects.py "$IMAGE_DIR"

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}错误: 物体描述生成失败${NC}"
    echo "请检查："
    echo "  1. API 密钥是否配置正确 (utils/agent.py)"
    echo "  2. 网络连接是否正常"
    echo "  3. 图片格式是否支持"
    exit 1
fi

echo -e "${GREEN}✓${NC} 物体描述生成完成"
echo ""

# ========================================
# 步骤 3：检查任务指令模板
# ========================================
echo -e "${BLUE}[步骤 3]${NC} 检查任务指令模板..."

TASK_NAME="adjust_bottle"  # 修改为你的任务名
TASK_JSON="task_instruction/${TASK_NAME}.json"

if [ ! -f "$TASK_JSON" ]; then
    echo -e "${YELLOW}警告: 任务指令模板不存在: $TASK_JSON${NC}"
    echo ""
    echo "请创建任务指令模板，例如："
    cat <<EOF
{
  "full_description": "Pick up the bottle on the table headup with the correct arm",
  "schema": "{A} notifies the bottle, {a} notifies the arm",
  "preference": "num of words should not exceed 15",
  "seen": [],
  "unseen": []
}
EOF
    echo ""
    echo "然后运行："
    echo "  bash gen_task_instruction_templates.sh $TASK_NAME 12"
    echo ""
    exit 1
fi

# 检查是否已生成指令
SEEN_COUNT=$(cat "$TASK_JSON" | jq '.seen | length')
UNSEEN_COUNT=$(cat "$TASK_JSON" | jq '.unseen | length')

if [ "$SEEN_COUNT" -eq 0 ] || [ "$UNSEEN_COUNT" -eq 0 ]; then
    echo -e "${YELLOW}警告: 任务指令模板为空，生成指令...${NC}"
    bash gen_task_instruction_templates.sh "$TASK_NAME" 12
    
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}错误: 任务指令生成失败${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✓${NC} 任务指令模板已准备 (seen: $SEEN_COUNT, unseen: $UNSEEN_COUNT)"
echo ""

# ========================================
# 步骤 4：检查 scene_info.json
# ========================================
echo -e "${BLUE}[步骤 4]${NC} 检查 scene_info.json..."

SCENE_INFO="../../data/${TASK_NAME}/train/scene_info.json"

if [ ! -f "$SCENE_INFO" ]; then
    echo -e "${YELLOW}警告: scene_info.json 不存在: $SCENE_INFO${NC}"
    echo ""
    echo "请先运行数据采集生成 scene_info.json"
    echo "然后修改物体引用格式："
    echo "  仿真格式: '001_bottle/base0'"
    echo "  真实格式: 'bottle/001'"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓${NC} scene_info.json 存在"
echo ""
echo "请确认 scene_info.json 中的物体引用已修改为真实环境格式："
echo "  示例: {\"A\": \"bottle/001\", \"a\": \"left\"}"
echo ""
read -p "是否已修改完成? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "请修改 $SCENE_INFO 后重新运行"
    exit 1
fi

# ========================================
# 步骤 5：生成场景指令
# ========================================
echo -e "${BLUE}[步骤 5]${NC} 生成场景指令..."
echo ""

# 这里需要先修改 generate_episode_instructions.py 的路径
echo -e "${YELLOW}注意: 请确认 generate_episode_instructions.py 中的路径已修改为:${NC}"
echo "  ../objects_description_real (第59行和108行)"
echo ""
read -p "是否已修改? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "请修改后重新运行"
    exit 1
fi

bash gen_episode_instructions.sh "$TASK_NAME" train 100

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}错误: 场景指令生成失败${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} 场景指令生成完成"
echo ""

# ========================================
# 完成
# ========================================
echo "=========================================="
echo -e "${GREEN}🎉 全部完成！${NC}"
echo "=========================================="
echo ""
echo "生成的文件："
echo "  1. 物体描述: objects_description_real/"
echo "  2. 任务指令: task_instruction/${TASK_NAME}.json"
echo "  3. 场景指令: ../../data/${TASK_NAME}/train/instructions/"
echo ""
echo "检查结果："
INSTRUCTION_DIR="../../data/${TASK_NAME}/train/instructions"
if [ -d "$INSTRUCTION_DIR" ]; then
    EPISODE_COUNT=$(ls -1 "$INSTRUCTION_DIR" | wc -l)
    echo "  - 生成了 $EPISODE_COUNT 个 episode 的指令"
    
    # 显示第一个 episode 的示例
    FIRST_EPISODE=$(ls "$INSTRUCTION_DIR" | head -1)
    if [ ! -z "$FIRST_EPISODE" ]; then
        echo ""
        echo "示例 ($FIRST_EPISODE):"
        cat "$INSTRUCTION_DIR/$FIRST_EPISODE" | jq '.seen[0:3]'
    fi
fi
echo ""

