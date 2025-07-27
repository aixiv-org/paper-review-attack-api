# Paper Review Attack Detection

一个用于检测学术论文中提示词注入攻击的综合工具包，专门针对AI辅助同行评议系统的安全威胁。

## 🎯 项目概述

随着AI在学术同行评议中的应用增加，恶意作者可能通过在论文中嵌入隐藏的提示词来操控AI审稿系统。本项目提供了完整的攻击检测解决方案，包括：

- 🔍 **多层次检测算法**: 关键词检测、语义分析、格式检测等
- 🌐 **多语言支持**: 支持中文、英文、日文等多种语言的提示词检测
- 🎯 **多种攻击类型**: 白色字体、元数据注入、不可见字符等
- 📊 **完整评估框架**: 性能评估、可视化分析、实验报告

## 🚀 快速开始

### 环境配置

```bash
# 克隆项目
git clone https://github.com/your-repo/paper-review-attack-detection.git
cd paper-review-attack-detection

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt