#!/usr/bin/env python3
"""
完整实验运行脚本
执行完整的数据收集、攻击生成、检测和评估流程
"""

import sys
import os
import argparse
from pathlib import Path
import json
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_collector import ArxivDatasetCollector
from src.attack_generator import AttackSampleGenerator, AdvancedAttackGenerator
from src.detector import PromptInjectionDetector, EnsembleDetector
from src.evaluator import ExperimentEvaluator
from src.utils import setup_logging, load_config, save_results

def run_data_collection(config, args, logger):
    """运行数据收集阶段"""
    logger.info("🔄 阶段 1: 数据收集")
    logger.info("-" * 40)
    
    if args.skip_download:
        # 尝试加载现有文件
        download_dir = Path(config['data_collection']['download_dir'])
        file_list_path = download_dir / "downloaded_files.txt"
        
        if file_list_path.exists():
            with open(file_list_path, 'r', encoding='utf-8') as f:
                clean_files = [line.strip() for line in f if line.strip()]
            logger.info(f"跳过下载，加载现有文件: {len(clean_files)} 个")
        else:
            logger.warning("跳过下载但未找到现有文件列表，将进行下载")
            args.skip_download = False
    
    if not args.skip_download:
        collector = ArxivDatasetCollector(config)
        clean_files = collector.collect_multi_category_papers()
        
        # 保存统计信息
        stats = collector.get_paper_statistics()
        logger.info(f"数据收集完成: {json.dumps(stats, indent=2, ensure_ascii=False)}")
    
    return clean_files

def run_attack_generation(config, clean_files, args, logger):
    """运行攻击生成阶段"""
    logger.info("\n🔄 阶段 2: 攻击样本生成")
    logger.info("-" * 40)
    
    if args.skip_attack_gen:
        # 尝试加载现有攻击样本
        output_dir = Path(config['attack_generation']['output_dir'])
        attack_list_path = output_dir / "generated_attacks.json"
        
        if attack_list_path.exists():
            with open(attack_list_path, 'r', encoding='utf-8') as f:
                attack_data = json.load(f)
            attack_files = attack_data.get('generated_files', [])
            attack_info = attack_data.get('attack_info', [])
            logger.info(f"跳过攻击生成，加载现有样本: {len(attack_files)} 个")
            return attack_files, attack_info
        else:
            logger.warning("跳过攻击生成但未找到现有攻击样本，将进行生成")
            args.skip_attack_gen = False
    
    if not args.skip_attack_gen:
        if args.advanced_attacks:
            generator = AdvancedAttackGenerator(config)
            logger.info("使用高级攻击生成器")
        else:
            generator = AttackSampleGenerator(config)
            logger.info("使用标准攻击生成器")
        
        attack_files = generator.generate_attack_samples(clean_files)
        attack_info = generator.attack_samples
        
        # 保存攻击信息
        stats = generator.get_attack_statistics()
        logger.info(f"攻击生成完成: {json.dumps(stats, indent=2, ensure_ascii=False)}")
    
    return attack_files, attack_info

def run_detection_evaluation(config, clean_files, attack_files, attack_info, args, logger):
    """运行检测和评估阶段"""
    logger.info("\n🔄 阶段 3: 检测和评估")
    logger.info("-" * 40)
    
    # 创建检测器
    if args.ensemble_detector:
        detector = EnsembleDetector(config)
        logger.info("使用集成检测器")
    else:
        detector = PromptInjectionDetector(config)
        logger.info("使用标准检测器")
    
    # 创建评估器
    evaluator = ExperimentEvaluator(config)
    
    # 限制样本数量（如果指定）
    if args.max_samples:
        clean_files = clean_files[:args.max_samples]
        attack_files = attack_files[:args.max_samples]
        logger.info(f"限制样本数量为: {args.max_samples}")
    
    # 运行评估
    df_results, metrics = evaluator.evaluate_detection_performance(
        clean_files, attack_files, detector, attack_info
    )
    
    return evaluator, df_results, metrics

def main():
    parser = argparse.ArgumentParser(description='运行完整实验流程')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='配置文件路径')
    
    # 流程控制参数
    parser.add_argument('--skip-download', action='store_true',
                       help='跳过数据下载阶段')
    parser.add_argument('--skip-attack-gen', action='store_true',
                       help='跳过攻击生成阶段')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='跳过评估阶段')
    parser.add_argument('--skip-plots', action='store_true',
                       help='跳过绘图')
    
    # 实验参数
    parser.add_argument('--max-papers', type=int,
                       help='最大下载论文数')
    parser.add_argument('--max-samples', type=int,
                       help='最大评估样本数')
    parser.add_argument('--attack-ratio', type=float,
                       help='攻击样本比例')
    parser.add_argument('--advanced-attacks', action='store_true',
                       help='使用高级攻击生成器')
    parser.add_argument('--ensemble-detector', action='store_true',
                       help='使用集成检测器')
    
    # 输出控制
    parser.add_argument('--output-dir', type=str,
                       help='输出目录')
    parser.add_argument('--experiment-name', type=str,
                       help='实验名称')
    parser.add_argument('--save-all', action='store_true',
                       help='保存所有中间结果')
    
    # 日志控制
    parser.add_argument('--log-file', type=str,
                       help='日志文件路径')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='详细输出')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='静默模式')
    
    args = parser.parse_args()
    
    # 设置日志
    if args.quiet:
        log_level = "WARNING"
    elif args.verbose:
        log_level = "DEBUG"
    else:
        log_level = "INFO"
    
    logger = setup_logging(log_level, args.log_file)
    
    # 实验开始
    start_time = time.time()
    experiment_name = args.experiment_name or f"experiment_{int(start_time)}"
    
    logger.info("=" * 80)
    logger.info(f"🚀 开始实验: {experiment_name}")
    logger.info("=" * 80)
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 命令行参数覆盖配置
        if args.max_papers:
            config['data_collection']['max_papers'] = args.max_papers
        
        if args.attack_ratio:
            config['attack_generation']['attack_ratio'] = args.attack_ratio
        
        if args.output_dir:
            config['experiment']['output_dir'] = args.output_dir
        
        # 创建实验目录
        output_dir = Path(config['experiment']['output_dir'])
        experiment_dir = output_dir / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 更新配置中的输出目录
        config['experiment']['output_dir'] = str(experiment_dir)
        config['attack_generation']['output_dir'] = str(experiment_dir / "attacks")
        
        logger.info(f"实验目录: {experiment_dir}")
        
        # 保存实验配置
        config_file = experiment_dir / "experiment_config.json"
        save_results(config, str(config_file))
        
        # 阶段 1: 数据收集
        clean_files = run_data_collection(config, args, logger)
        
        if not clean_files:
            logger.error("数据收集失败，退出实验")
            return 1
        
        # 阶段 2: 攻击生成
        attack_files, attack_info = run_attack_generation(config, clean_files, args, logger)
        
        if not attack_files:
            logger.error("攻击生成失败，退出实验")
            return 1
        
        # 阶段 3: 检测和评估
        if not args.skip_evaluation:
            evaluator, df_results, metrics = run_detection_evaluation(
                config, clean_files, attack_files, attack_info, args, logger
            )
            
            # 生成报告
            logger.info("\n🔄 阶段 4: 结果分析和报告")
            logger.info("-" * 40)
            
            # 绘制图表
            if not args.skip_plots:
                try:
                    evaluator.plot_performance_analysis(df_results, metrics, save_plots=True)
                    logger.info("性能分析图表已生成")
                except Exception as e:
                    logger.error(f"绘图失败: {e}")
            
            # 生成报告
            report = evaluator.generate_report(df_results, metrics)
            logger.info("实验报告已生成")
            
            # 保存实验总结
            experiment_summary = {
                'experiment_name': experiment_name,
                'start_time': start_time,
                'end_time': time.time(),
                'duration_seconds': time.time() - start_time,
                'config': config,
                'data_stats': {
                    'clean_files': len(clean_files),
                    'attack_files': len(attack_files)
                },
                'performance_metrics': metrics,
                'args': vars(args)
            }
            
            summary_file = experiment_dir / "experiment_summary.json"
            save_results(experiment_summary, str(summary_file))
            
            # 打印最终结果
            logger.info("\n" + "=" * 80)
            logger.info("🎉 实验完成")
            logger.info("=" * 80)
            logger.info(f"实验名称: {experiment_name}")
            logger.info(f"总用时: {time.time() - start_time:.1f} 秒")
            logger.info(f"实验目录: {experiment_dir}")
            logger.info("")
            logger.info("📊 主要结果:")
            logger.info(f"  准确率: {metrics['accuracy']:.3f}")
            logger.info(f"  精确率: {metrics['precision']:.3f}")
            logger.info(f"  召回率: {metrics['recall']:.3f}")
            logger.info(f"  F1分数: {metrics['f1_score']:.3f}")
            logger.info(f"  ROC AUC: {metrics['roc_auc']:.3f}")
            
        else:
            logger.info("跳过评估阶段")
        
        logger.info("\n✅ 实验成功完成！")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("用户中断实验")
        return 1
    except Exception as e:
        logger.error(f"实验过程中发生错误: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())
