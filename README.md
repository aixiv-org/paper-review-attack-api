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
conda create -n paper_review_attack python==3.12.0
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```
### 下载论文

```bash
# 从arXiv下载
python scripts/download_data.py --source arxiv --max-papers 100

# 使用本地PDF
python scripts/download_data.py --source local --local-dir /path/to/pdfs
```
### 快速生成所有攻击

```bash
# 生成所有类型的攻击样本
python scripts/generate_attacks.py \
  --input-dir data/clean_papers \
  --output-dir data/attack_samples \
  --attack-types white_text metadata invisible_chars mixed_language \
  --attack-ratio 0.5 \
  --languages english chinese japanese mixed \
  --batch-size 10

# 使用高级攻击生成器
python scripts/generate_attacks.py \
  --advanced \
  --attack-types white_text metadata invisible_chars steganographic \
  --languages english chinese japanese \
  --attack-ratio 0.3
```
### 快速生成所有攻击

```bash
# 生成白色字体攻击
python scripts/generate_attacks.py --attack-types white_text

# 生成多语言攻击
python scripts/generate_attacks.py --languages english chinese

# 使用高级攻击
python scripts/generate_attacks.py --advanced

# 1. 仅生成白色字体攻击
python scripts/generate_attacks.py \
  --attack-types white_text \
  --languages english \
  --attack-ratio 0.8

# 2. 仅生成元数据攻击
python scripts/generate_attacks.py \
  --attack-types metadata \
  --languages chinese \
  --attack-ratio 0.5

# 3. 仅生成不可见字符攻击
python scripts/generate_attacks.py \
  --attack-types invisible_chars \
  --languages japanese \
  --attack-ratio 0.3

# 4. 仅生成多语言混合攻击
python scripts/generate_attacks.py \
  --attack-types mixed_language \
  --languages mixed \
  --attack-ratio 0.4
```

### 检测方法

```bash

# 标准检测
python scripts/run_detection.py --detector-type standard

# 集成检测
python scripts/run_detection.py --detector-type ensemble

# 单文件检测
python scripts/run_detection.py --single-file paper.pdf

```

### 测试

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_detector.py -v

# 测试覆盖率
python -m pytest --cov=src tests/

```

```bash
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

### Pipeline

```bash
graph TD
    A[PDF文件输入] --> B[PDF内容提取阶段]
    
    B --> B1[文本内容提取]
    B --> B2[格式信息提取]
    B --> B3[元数据提取]
    B --> B4[字体分析]
    
    B1 --> C[预处理阶段]
    B2 --> C
    B3 --> C
    B4 --> C
    
    C --> C1[文本清洗]
    C --> C2[语言检测]
    C --> C3[结构分析]
    
    C1 --> D[粗略检测-多层并行扫描]
    C2 --> D
    C3 --> D
    
    D --> D1[关键词快速匹配]
    D --> D2[格式异常检测]
    D --> D3[编码模式识别]
    D --> D4[统计特征分析]
    
    D1 --> E[候选威胁识别]
    D2 --> E
    D3 --> E
    D4 --> E
    
    E --> F{是否发现可疑内容?}
    F -->|否| G[标记为正常]
    F -->|是| H[精细检测阶段]
    
    H --> H1[深度语义分析]
    H --> H2[上下文一致性检查]
    H --> H3[多语言交叉验证]
    H --> H4[攻击模式确认]
    
    H1 --> I[威胁类型分类]
    H2 --> I
    H3 --> I
    H4 --> I
    
    I --> I1[关键词注入确认]
    I --> I2[格式攻击确认]
    I --> I3[隐写术确认]
    I --> I4[语义攻击确认]
    
    I1 --> J[风险分数计算]
    I2 --> J
    I3 --> J
    I4 --> J
    
    J --> K[加权融合]
    K --> L[最终判定]
    L --> M[结果输出]
    
    G --> M
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

