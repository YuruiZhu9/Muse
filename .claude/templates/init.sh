#!/bin/bash
# MuseRecSys 开发环境初始化脚本
# 使用方法: ./init.sh

set -e  # 遇到错误立即退出

echo "=== MuseRecSys 环境初始化 ==="

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. 检查 Python 版本
echo -e "${YELLOW}检查 Python 版本...${NC}"
if command -v python3 &> /dev/null; then
    python3 --version
else
    echo "错误: 未找到 Python3，请先安装 Python 3.8+"
    exit 1
fi

# 2. 创建虚拟环境（如果不存在）
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}创建虚拟环境...${NC}"
    python3 -m venv venv
fi

# 3. 激活虚拟环境
echo -e "${YELLOW}激活虚拟环境...${NC}"
source venv/bin/activate

# 4. 安装依赖
if [ -f "requirements.txt" ]; then
    echo -e "${YELLOW}安装依赖包...${NC}"
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "警告: 未找到 requirements.txt，跳过依赖安装"
fi

# 5. 创建必要的目录
echo -e "${YELLOW}创建项目目录...${NC}"
mkdir -p logs
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models
mkdir -p tests

# 6. 环境变量配置
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}创建 .env 配置文件...${NC}"
    cat > .env << EOF
# MuseRecSys 环境配置
ENV=development
DEBUG=True
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///musercsys.db
EOF
fi

# 7. 初始化数据库（如果需要）
# echo -e "${YELLOW}初始化数据库...${NC}"
# python -c "from yourapp import db; db.create_all()"

echo -e "${GREEN}=== 初始化完成 ===${NC}"
echo -e "${GREEN}虚拟环境已激活: venv${NC}"
echo "下一步:"
echo "  1. 根据需要修改 .env 配置"
echo "  2. 运行应用: python app.py 或 flask run"
