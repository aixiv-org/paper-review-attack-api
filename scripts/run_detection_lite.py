#!/usr/bin/env python3
"""
轻量级检测脚本
不依赖大型AI模型的PDF提示词注入检测
"""

import sys
import os
import argparse
from pathlib import Path
import json
import pandas as pd

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.detector_lite import LightweightPromptInjectionDetector
from src.utils import setup_logging, load_config, ProgressTracker, save_results

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
            elif isinstance(data, dict) and 'generated_files' in data:
                files = data['generated_files']
            elif isinstance(data, dict) and 'files' in data:
                files = data['files']
    
    # 验证文件存在性
    valid_files = []
    for file_path in files:
        if os.path.exists(file_path) and file_path.lower().endswith('.pdf'):
            valid_files.append(file_path)
    
    return valid_files

def main():
    parser = argparse.ArgumentParser(description='运行轻量级提示词注入检测')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='配置文件路径')
    parser.add_argument('--input-dir', type=str,
                       help='输入PDF目录')
    parser.add_argument('--file-list', type=str,
                       help='PDF文件列表文件路径（.txt或.json）')
    parser.add_argument('--single-file', type=str,
                       help='单个PDF文件路径')
    parser.add_argument('--output-dir', type=str,
                       help='输出目录（覆盖配置文件）')
    parser.add_argument('--threshold', type=float,
                       help='风险分数阈值（覆盖配置文件）')
    parser.add_argument('--batch-size', type=int, default=20,
                       help='批处理大小')
    parser.add_argument('--save-details', action='store_true',
                       help='保存详细检测结果')
    parser.add_argument('--log-file', type=str,
                       help='日志文件路径')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='详细输出')
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level, args.log_file)
    
    logger.info("=" * 60)
    logger.info("轻量级提示词注入检测器启动")
    logger.info("=" * 60)
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 命令行参数覆盖配置
        if args.output_dir:
            config['experiment']['output_dir'] = args.output_dir
        
        if args.threshold:
            config['detection']['thresholds']['risk_score'] = args.threshold
        
        # 获取输入文件列表
        pdf_files = []
        
        if args.single_file:
            if os.path.exists(args.single_file):
                pdf_files = [args.single_file]
                logger.info(f"检测单个文件: {args.single_file}")
            else:
                logger.error(f"文件不存在: {args.single_file}")
                return 1
                
        elif args.file_list:
            logger.info(f"从文件列表加载PDF: {args.file_list}")
            pdf_files = load_file_list(args.file_list)
            
        elif args.input_dir:
            logger.info(f"从目录扫描PDF: {args.input_dir}")
            input_path = Path(args.input_dir)
            pdf_files = [str(f) for f in input_path.rglob("*.pdf")]
            
        else:
            logger.error("必须指定输入源：--single-file、--file-list 或 --input-dir")
            return 1
        
        if not pdf_files:
            logger.error("没有找到PDF文件！")
            return 1
        
        logger.info(f"找到 {len(pdf_files)} 个PDF文件")
        
        # 创建轻量级检测器
        detector = LightweightPromptInjectionDetector(config)
        logger.info("使用轻量级检测器（无AI模型依赖）")
        
        # 分批处理
        batch_size = args.batch_size
        total_batches = (len(pdf_files) + batch_size - 1) // batch_size
        
        all_results = []
        progress = ProgressTracker(len(pdf_files), "检测PDF文件")
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(pdf_files))
            batch_files = pdf_files[start_idx:end_idx]
            
            logger.info(f"处理批次 {batch_idx + 1}/{total_batches} "
                       f"({len(batch_files)} 个文件)")
            
            for file_path in batch_files:
                try:
                    result = detector.detect_injection(file_path)
                    all_results.append(result)
                    
                    # 实时显示结果
                    risk_score = result['risk_score']
                    is_malicious = result['is_malicious']
                    detection_count = result['detection_count']
                    
                    status = "🚨 恶意" if is_malicious else "✅ 正常"
                    logger.info(f"{status} | {Path(file_path).name} | "
                               f"风险: {risk_score:.3f} | 检测: {detection_count}")
                    
                    progress.update()
                    
                except Exception as e:
                    logger.error(f"检测失败 {file_path}: {e}")
                    progress.update()
                    continue
        
        progress.finish()
        
        # 统计结果
        total_files = len(all_results)
        malicious_files = sum(1 for r in all_results if r['is_malicious'])
        avg_risk_score = sum(r['risk_score'] for r in all_results) / total_files if total_files > 0 else 0
        avg_detections = sum(r['detection_count'] for r in all_results) / total_files if total_files > 0 else 0
        
        logger.info("=" * 60)
        logger.info("检测完成 - 统计结果")
        logger.info("=" * 60)
        logger.info(f"总文件数: {total_files}")
        logger.info(f"检测为恶意: {malicious_files} ({malicious_files/total_files*100:.1f}%)")
        logger.info(f"平均风险分数: {avg_risk_score:.3f}")
        logger.info(f"平均检测数量: {avg_detections:.1f}")
        
        # 保存结果
        output_dir = Path(config['experiment']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存简要结果
        summary_results = []
        for result in all_results:
            summary = {
                'file': result['file'],
                'file_name': Path(result['file']).name,
                'is_malicious': result['is_malicious'],
                'risk_score': result['risk_score'],
                'detection_count': result['detection_count']
            }
            summary_results.append(summary)
        
        summary_file = output_dir / f"detection_summary_lite_{timestamp}.csv"
        pd.DataFrame(summary_results).to_csv(summary_file, index=False, encoding='utf-8')
        logger.info(f"检测摘要已保存: {summary_file}")
        
        # 保存详细结果
        if args.save_details:
            details_file = output_dir / f"detection_details_lite_{timestamp}.json"
            save_results(all_results, str(details_file))
            logger.info(f"详细结果已保存: {details_file}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("用户中断检测")
        return 1
    except Exception as e:
        logger.error(f"检测过程中发生错误: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())