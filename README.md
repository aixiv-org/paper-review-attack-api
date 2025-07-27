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
git clone https://github.com/King-play/paper-review-attack-detection.git
cd paper-review-attack-detection

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 从arXiv下载
python scripts/download_data.py --source arxiv --max-papers 100

# 使用本地PDF
python scripts/download_data.py --source local --local-dir /path/to/pdfs

# 生成白色字体攻击
python scripts/generate_attacks.py --attack-types white_text

# 生成多语言攻击
python scripts/generate_attacks.py --languages english chinese

# 使用高级攻击
python scripts/generate_attacks.py --advanced

# 标准检测
python scripts/run_detection.py --detector-type standard

# 集成检测
python scripts/run_detection.py --detector-type ensemble

# 单文件检测
python scripts/run_detection.py --single-file paper.pdf

# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_detector.py -v

# 测试覆盖率
python -m pytest --cov=src tests/

# 1. 完整实验（推荐）
python scripts/run_experiment.py --experiment-name "baseline_test"

# 2. 分步执行
python scripts/download_data.py --max-papers 50
python scripts/generate_attacks.py --attack-ratio 0.3
python scripts/run_detection.py --input-dir data/clean_papers


paper_review_attack_detection/
├── README.md                 # 项目说明
├── requirements.txt          # 依赖包列表
├── config/
│   └── config.yaml          # 配置文件
├── src/                     # 核心代码
│   ├── data_collector.py    # 数据收集
│   ├── attack_generator.py  # 攻击生成
│   ├── detector.py          # 检测算法
│   ├── evaluator.py         # 性能评估
│   └── utils.py             # 工具函数
├── scripts/                 # 执行脚本
│   ├── download_data.py     # 数据下载
│   ├── generate_attacks.py  # 攻击生成
│   ├── run_detection.py     # 运行检测
│   └── run_experiment.py    # 完整实验
├── data/                    # 数据目录
│   ├── clean_papers/        # 正常论文
│   ├── attack_samples/      # 攻击样本
│   └── results/             # 实验结果
└── tests/                   # 测试文件
```

📈 可视化
系统提供丰富的可视化功能：

🔵 混淆矩阵: 检测准确性分析
📊 ROC曲线: 分类器性能评估
📈 风险分数分布: 正常vs攻击文件对比
🎯 按攻击类型分析: 各类攻击的检测效果
🤝 贡献指南

🙏 致谢
- arXiv.org 提供的开放访问论文数据
- Hugging Face 提供的预训练模型
- 学术界对AI安全研究的支持

📚 相关文献  
1."Hidden Prompts in Manuscripts Exploit AI-Assisted Peer Review" - arXiv:2507.06185  
2."Prompt Injection Attacks in Academic Publishing" - 相关研究  
3."AI Safety in Scholarly Communication" - 理论基础  

