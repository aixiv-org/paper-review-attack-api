#!/usr/bin/env python3
"""
攻击样本生成脚本
Generate attack samples for prompt injection detection testing
支持多种攻击类型和语言
"""

import sys
import os
import argparse
from pathlib import Path
import json
from typing import List, Dict, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.attack_generator import AttackSampleGenerator, AdvancedAttackGenerator
from src.utils import setup_logging, load_config, ensure_dir

def parse_attack_types(attack_types_str: str) -> List[str]:
    """解析攻击类型参数"""
    if not attack_types_str:
        return ['white_text', 'metadata', 'invisible_chars', 'mixed_language']
    
    # 支持空格分隔或逗号分隔
    if ',' in attack_types_str:
        return [t.strip() for t in attack_types_str.split(',')]
    else:
        return attack_types_str.split()

def parse_languages(languages_str: str) -> List[str]:
    """解析语言参数"""
    if not languages_str:
        return ['english', 'chinese', 'japanese']
    
    # 支持空格分隔或逗号分隔
    if ',' in languages_str:
        return [l.strip() for l in languages_str.split(',')]
    else:
        return languages_str.split()

def get_clean_pdfs(input_dir: str) -> List[str]:
    """获取输入目录中的PDF文件"""
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    pdf_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    return pdf_files

def main():
    parser = argparse.ArgumentParser(description='生成攻击样本用于提示词注入检测测试')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='配置文件路径')
    parser.add_argument('--input-dir', type=str, required=True,
                       help='输入PDF目录')
    parser.add_argument('--output-dir', type=str, default='data/attack_samples',
                       help='输出目录')
    parser.add_argument('--attack-types', nargs='+', 
                       help='攻击类型: white_text metadata invisible_chars mixed_language steganographic contextual_attack')
    parser.add_argument('--attack-ratio', type=float, default=0.3,
                       help='攻击比例 (0.0-1.0)')
    parser.add_argument('--languages', nargs='+',
                       help='支持的语言: english chinese japanese mixed')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='批处理大小')
    parser.add_argument('--advanced', action='store_true',
                       help='使用高级攻击生成器')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='详细输出')
    parser.add_argument('--log-file', type=str,
                       help='日志文件路径')
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level, args.log_file)
    
    logger.info("=" * 60)
    logger.info("攻击样本生成器启动")
    logger.info("=" * 60)
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 更新配置
        if 'attack_generation' not in config:
            config['attack_generation'] = {}
        
        config['attack_generation']['output_dir'] = args.output_dir
        config['attack_generation']['batch_size'] = args.batch_size
        config['attack_generation']['attack_ratio'] = args.attack_ratio
        
        # 处理攻击类型
        if args.attack_types:
            attack_types = args.attack_types
            # 转换为等权重字典
            equal_weight = 1.0 / len(attack_types)
            config['attack_generation']['attack_types'] = {
                attack_type: equal_weight for attack_type in attack_types
            }
            logger.info(f"使用攻击类型: {attack_types}")
        else:
            # 使用默认攻击类型
            logger.info("使用默认攻击类型")
        
        # 处理语言设置（AttackSampleGenerator会使用内置的默认模板）
        if args.languages:
            languages = args.languages
            logger.info(f"使用语言: {languages}")
            # 注意：具体的提示词模板由AttackSampleGenerator内部处理
        else:
            logger.info("使用默认语言配置")
        
        # 获取输入PDF文件
        logger.info(f"扫描输入目录: {args.input_dir}")
        clean_pdfs = get_clean_pdfs(args.input_dir)
        
        if not clean_pdfs:
            logger.error(f"在目录 {args.input_dir} 中未找到PDF文件")
            return 1
        
        logger.info(f"找到 {len(clean_pdfs)} 个PDF文件")
        
        # 创建输出目录
        ensure_dir(args.output_dir)
        
        # 选择生成器
        if args.advanced:
            logger.info("使用高级攻击生成器")
            generator = AdvancedAttackGenerator(config)
        else:
            logger.info("使用标准攻击生成器")
            generator = AttackSampleGenerator(config)
        
        # 生成攻击样本
        logger.info("开始生成攻击样本...")
        generated_samples = generator.generate_attack_samples(clean_pdfs)
        
        # 保存结果
        results = {
            'input_directory': args.input_dir,
            'output_directory': args.output_dir,
            'total_input_files': len(clean_pdfs),
            'generated_samples': len(generated_samples),
            'attack_types': list(config['attack_generation'].get('attack_types', {}).keys()),
            'languages': args.languages if args.languages else ['使用内置默认语言'],
            'attack_ratio': args.attack_ratio,
            'batch_size': args.batch_size,
            'advanced_mode': args.advanced,
            'generated_files': generated_samples,
            'stats': generator.stats
        }
        
        results_file = os.path.join(args.output_dir, 'generation_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info("=" * 60)
        logger.info("攻击样本生成完成")
        logger.info(f"输入文件: {len(clean_pdfs)}")
        logger.info(f"生成样本: {len(generated_samples)}")
        logger.info(f"成功率: {len(generated_samples)/len(clean_pdfs)*args.attack_ratio:.1%}")
        logger.info(f"输出目录: {args.output_dir}")
        logger.info(f"结果文件: {results_file}")
        
        # 显示攻击类型分布
        if generator.stats.get('successful_generations', 0) > 0:
            attack_stats = generator.get_attack_statistics()
            if 'attack_types' in attack_stats:
                logger.info("攻击类型分布:")
                for attack_type, count in attack_stats['attack_types'].items():
                    logger.info(f"  {attack_type}: {count}")
        
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"攻击样本生成失败: {e}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        return 1

if __name__ == '__main__':
    sys.exit(main())