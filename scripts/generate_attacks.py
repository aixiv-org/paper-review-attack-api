#!/usr/bin/env python3
"""
攻击样本生成脚本
基于正常PDF文件生成各种类型的提示词注入攻击样本
"""

import sys
import os
import argparse
from pathlib import Path
import json

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.attack_generator import AttackSampleGenerator, AdvancedAttackGenerator
from src.utils import setup_logging, load_config, ProgressTracker

def load_file_list(file_list_path: str) -> list:
    """从文件加载PDF文件列表"""
    files = []
    
    if file_list_path.endswith('.txt'):
        with open(file_list_path, 'r', encoding='utf-8') as f:
            files = [line.strip() for line in f if line.strip()]
    elif file_list_path.endswith('.json'):
        with open(file_list_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                files = data
            elif isinstance(data, dict) and 'files' in data:
                files = data['files']
    
    # 验证文件存在性
    valid_files = []
    for file_path in files:
        if os.path.exists(file_path) and file_path.lower().endswith('.pdf'):
            valid_files.append(file_path)
    
    return valid_files

def main():
    parser = argparse.ArgumentParser(description='生成攻击样本')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='配置文件路径')
    parser.add_argument('--input-dir', type=str,
                       help='输入PDF目录')
    parser.add_argument('--file-list', type=str,
                       help='PDF文件列表文件路径（.txt或.json）')
    parser.add_argument('--output-dir', type=str,
                       help='输出目录（覆盖配置文件）')
    parser.add_argument('--attack-types', nargs='+',
                       choices=['white_text', 'metadata', 'invisible_chars', 
                               'mixed_language', 'steganographic'],
                       help='攻击类型列表（覆盖配置文件）')
    parser.add_argument('--attack-ratio', type=float,
                       help='攻击样本比例（覆盖配置文件）')
    parser.add_argument('--languages', nargs='+',
                       choices=['english', 'chinese', 'japanese', 'mixed'],
                       help='提示词语言列表')
    parser.add_argument('--advanced', action='store_true',
                       help='使用高级攻击生成器')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='批处理大小')
    parser.add_argument('--log-file', type=str,
                       help='日志文件路径')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='详细输出')
    
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
        
        # 命令行参数覆盖配置
        if args.output_dir:
            config['attack_generation']['output_dir'] = args.output_dir
        
        if args.attack_types:
            config['attack_generation']['attack_types'] = args.attack_types
        
        if args.attack_ratio:
            config['attack_generation']['attack_ratio'] = args.attack_ratio
        
        if args.languages:
            # 只保留指定语言的提示词
            original_templates = config['attack_generation']['prompt_templates']
            filtered_templates = {lang: original_templates[lang] 
                                for lang in args.languages 
                                if lang in original_templates}
            config['attack_generation']['prompt_templates'] = filtered_templates
        
        # 获取输入文件列表
        pdf_files = []
        
        if args.file_list:
            logger.info(f"从文件列表加载PDF: {args.file_list}")
            pdf_files = load_file_list(args.file_list)
        elif args.input_dir:
            logger.info(f"从目录扫描PDF: {args.input_dir}")
            input_path = Path(args.input_dir)
            pdf_files = [str(f) for f in input_path.rglob("*.pdf")]
        else:
            # 使用配置文件中的默认目录
            default_dir = config['data_collection']['download_dir']
            logger.info(f"使用默认目录: {default_dir}")
            
            # 尝试加载文件列表
            file_list_path = Path(default_dir) / "downloaded_files.txt"
            if file_list_path.exists():
                pdf_files = load_file_list(str(file_list_path))
            else:
                # 扫描目录
                pdf_files = [str(f) for f in Path(default_dir).rglob("*.pdf")]
        
        if not pdf_files:
            logger.error("没有找到PDF文件！请检查输入路径。")
            return 1
        
        logger.info(f"找到 {len(pdf_files)} 个PDF文件")
        
        # 创建攻击样本生成器
        if args.advanced:
            generator = AdvancedAttackGenerator(config)
            logger.info("使用高级攻击生成器")
        else:
            generator = AttackSampleGenerator(config)
            logger.info("使用标准攻击生成器")
        
        # 分批处理
        batch_size = args.batch_size
        total_batches = (len(pdf_files) + batch_size - 1) // batch_size
        
        all_generated_samples = []
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(pdf_files))
            batch_files = pdf_files[start_idx:end_idx]
            
            logger.info(f"处理批次 {batch_idx + 1}/{total_batches} "
                       f"({len(batch_files)} 个文件)")
            
            try:
                # 生成攻击样本
                generated_samples = generator.generate_attack_samples(batch_files)
                all_generated_samples.extend(generated_samples)
                
                logger.info(f"批次 {batch_idx + 1} 生成了 {len(generated_samples)} 个攻击样本")
                
            except Exception as e:
                logger.error(f"批次 {batch_idx + 1} 处理失败: {e}")
                continue
        
        # 获取统计信息
        stats = generator.get_attack_statistics()
        
        logger.info("=" * 60)
        logger.info("攻击样本生成完成")
        logger.info("=" * 60)
        logger.info(f"总生成样本数: {len(all_generated_samples)}")
        logger.info(f"攻击统计: {json.dumps(stats, indent=2, ensure_ascii=False)}")
        
        # 保存生成的攻击样本列表
        output_dir = Path(config['attack_generation']['output_dir'])
        attack_list_path = output_dir / "generated_attacks.json"
        
        attack_info = {
            'total_samples': len(all_generated_samples),
            'generated_files': all_generated_samples,
            'statistics': stats,
            'generation_config': {
                'attack_types': config['attack_generation']['attack_types'],
                'attack_ratio': config['attack_generation']['attack_ratio'],
                'languages': list(config['attack_generation']['prompt_templates'].keys())
            }
        }
        
        with open(attack_list_path, 'w', encoding='utf-8') as f:
            json.dump(attack_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"攻击样本信息已保存到: {attack_list_path}")
        
        # 验证生成的文件
        logger.info("验证生成的攻击样本...")
        valid_samples = 0
        invalid_samples = 0
        
        for sample_path in all_generated_samples:
            if os.path.exists(sample_path) and os.path.getsize(sample_path) > 1024:
                valid_samples += 1
            else:
                invalid_samples += 1
        
        logger.info(f"验证结果: {valid_samples} 个有效样本, {invalid_samples} 个无效样本")
        
        if invalid_samples > 0:
            logger.warning(f"存在 {invalid_samples} 个无效攻击样本，请检查生成过程")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("用户中断生成过程")
        return 1
    except Exception as e:
        logger.error(f"生成过程中发生错误: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())