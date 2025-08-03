#!/usr/bin/env python3
"""
检测脚本 (增强版)
对PDF文件运行提示词注入检测，包含性能指标计算
"""

import sys
import os
import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
import time

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.detector import PromptInjectionDetector, EnsembleDetector
from src.utils import setup_logging, load_config, ProgressTracker, save_results

# 尝试导入sklearn，如果没有则跳过性能指标
try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("警告: sklearn未安装，将跳过性能指标计算")

def calculate_performance_metrics(results: List[Dict], ground_truth: Optional[Dict] = None, 
                                threshold: Optional[float] = None) -> Dict:
    """计算性能指标 - 修复版"""
    if not results:
        return {}
    
    # 基础统计
    total_files = len(results)
    malicious_files = sum(1 for r in results if r['is_malicious'])
    risk_scores = [r['risk_score'] for r in results]
    detection_counts = [r['detection_count'] for r in results]
    
    basic_metrics = {
        'total_files': total_files,
        'malicious_files': malicious_files,
        'benign_files': total_files - malicious_files,
        'malicious_rate': malicious_files / total_files if total_files > 0 else 0,
        'avg_risk_score': np.mean(risk_scores) if risk_scores else 0,
        'std_risk_score': np.std(risk_scores) if risk_scores else 0,
        'min_risk_score': np.min(risk_scores) if risk_scores else 0,
        'max_risk_score': np.max(risk_scores) if risk_scores else 0,
        'median_risk_score': np.median(risk_scores) if risk_scores else 0,
        'avg_detection_count': np.mean(detection_counts) if detection_counts else 0,
        'threshold_used': threshold or 0.45
    }
    
    # 风险分布
    basic_metrics['risk_distribution'] = {
        'very_low': sum(1 for r in risk_scores if r < 0.1),
        'low': sum(1 for r in risk_scores if 0.1 <= r < 0.3),
        'medium': sum(1 for r in risk_scores if 0.3 <= r < 0.5),
        'high': sum(1 for r in risk_scores if 0.5 <= r < 0.7),
        'very_high': sum(1 for r in risk_scores if r >= 0.7)
    }
    
    # 检测类型统计
    detection_type_counts = {}
    total_detections = 0
    
    for result in results:
        for detection in result.get('detections', []):
            det_type = detection.get('type', 'unknown')
            detection_type_counts[det_type] = detection_type_counts.get(det_type, 0) + 1
            total_detections += 1
    
    basic_metrics['detection_statistics'] = {
        'total_detections': total_detections,
        'unique_detection_types': len(detection_type_counts),
        'type_distribution': detection_type_counts
    }
    
    # 🚀 修复：如果有真实标签且sklearn可用，计算性能指标
    if ground_truth and SKLEARN_AVAILABLE:
        print(f"\n🔍 调试信息:")
        print(f"标签文件包含 {len(ground_truth)} 个条目")
        print(f"前5个标签键: {list(ground_truth.keys())[:5]}")
        
        try:
            # 准备标签数据
            y_true = []
            y_pred = []
            y_score = []
            matched_files = 0
            unmatched_files = []
            
            for result in results:
                file_path = result['file']
                file_name = Path(file_path).name
                file_stem = Path(file_path).stem
                
                # 🚀 增强：尝试更多匹配模式
                possible_keys = [
                    file_path,           # 完整路径
                    file_name,           # 文件名
                    file_stem,           # 不含扩展名的文件名
                    file_path.replace('\\', '/'),  # 标准化路径分隔符
                    str(Path(file_path).as_posix()),  # POSIX路径格式
                ]
                
                # 🚀 新增：模糊匹配
                true_label = None
                matched_key = None
                
                # 精确匹配
                for key in possible_keys:
                    if key in ground_truth:
                        true_label = ground_truth[key]
                        matched_key = key
                        break
                
                # 如果精确匹配失败，尝试部分匹配
                if true_label is None:
                    for gt_key in ground_truth.keys():
                        # 检查文件名是否包含在标签键中，或反之
                        if (file_name in gt_key or gt_key in file_name or
                            file_stem in gt_key or gt_key in file_stem):
                            true_label = ground_truth[gt_key]
                            matched_key = gt_key
                            break
                
                if true_label is not None:
                    y_true.append(int(true_label))
                    y_pred.append(1 if result['is_malicious'] else 0)
                    y_score.append(result['risk_score'])
                    matched_files += 1
                    print(f"✅ 匹配: {file_name} -> {matched_key} (标签: {true_label})")
                else:
                    unmatched_files.append(file_name)
                    print(f"❌ 未匹配: {file_name}")
            
            print(f"\n📊 匹配结果: {matched_files}/{total_files} 个文件匹配成功")
            if unmatched_files:
                print(f"未匹配文件: {unmatched_files[:3]}...")
            
            if matched_files > 0:
                basic_metrics['ground_truth_matched'] = matched_files
                basic_metrics['ground_truth_coverage'] = matched_files / total_files
                
                # 🚀 修复：确保有足够的数据计算性能指标
                if len(set(y_true)) > 1 and len(y_true) > 0:  # 确保有正负样本
                    try:
                        basic_metrics['performance'] = {
                            'accuracy': float(accuracy_score(y_true, y_pred)),
                            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
                            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
                            'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
                        }
                        
                        # 计算混淆矩阵
                        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                        basic_metrics['confusion_matrix'] = {
                            'true_negatives': int(tn),
                            'false_positives': int(fp),
                            'false_negatives': int(fn),
                            'true_positives': int(tp)
                        }
                        
                        # 计算额外指标
                        basic_metrics['performance']['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
                        basic_metrics['performance']['false_positive_rate'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
                        basic_metrics['performance']['false_negative_rate'] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
                        
                        # 计算ROC AUC
                        try:
                            basic_metrics['performance']['roc_auc'] = float(roc_auc_score(y_true, y_score))
                        except ValueError as e:
                            print(f"ROC AUC计算失败: {e}")
                            basic_metrics['performance']['roc_auc'] = 0.0
                        
                        print(f"✅ 性能指标计算成功!")
                        
                    except Exception as e:
                        print(f"❌ 性能指标计算失败: {e}")
                        basic_metrics['performance'] = {'error': f'性能指标计算失败: {str(e)}'}
                        
                elif len(set(y_true)) <= 1:
                    basic_metrics['performance'] = {'note': f'只有单一类别 (类别: {set(y_true)})，无法计算性能指标'}
                    print(f"⚠️ 只有单一类别: {set(y_true)}")
                    
            else:
                basic_metrics['performance'] = {'note': '无匹配的真实标签'}
                print("❌ 没有成功匹配任何文件")
                
        except Exception as e:
            print(f"❌ 性能指标计算过程失败: {e}")
            basic_metrics['performance'] = {'error': f'性能指标计算失败: {str(e)}'}
    
    elif not ground_truth:
        print("⚠️ 未提供真实标签文件")
        basic_metrics['performance'] = {'note': '未提供真实标签'}
    elif not SKLEARN_AVAILABLE:
        print("⚠️ sklearn不可用")
        basic_metrics['performance'] = {'note': 'sklearn不可用'}
    
    return basic_metrics

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

def load_ground_truth(ground_truth_path: str) -> Optional[Dict]:
    """加载真实标签"""
    if not os.path.exists(ground_truth_path):
        return None
    
    try:
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            if ground_truth_path.endswith('.json'):
                return json.load(f)
            elif ground_truth_path.endswith('.csv'):
                df = pd.read_csv(ground_truth_path)
                if 'file' in df.columns and 'label' in df.columns:
                    return dict(zip(df['file'], df['label']))
                elif 'filename' in df.columns and 'is_malicious' in df.columns:
                    return dict(zip(df['filename'], df['is_malicious']))
    except Exception as e:
        print(f"警告: 无法加载真实标签文件 {ground_truth_path}: {e}")
    
    return None

def generate_detailed_report(metrics: Dict, output_path: str):
    """生成详细的检测报告"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = f"""
PDF恶意检测详细报告
==========================================
生成时间: {timestamp}
检测阈值: {metrics.get('threshold_used', 'N/A')}

📊 基础统计
-----------
总文件数: {metrics['total_files']}
恶意文件数: {metrics['malicious_files']} ({metrics['malicious_rate']:.1%})
正常文件数: {metrics['benign_files']} ({(1-metrics['malicious_rate']):.1%})

📈 风险分数分析
--------------
平均风险分数: {metrics['avg_risk_score']:.4f}
标准差: {metrics['std_risk_score']:.4f}
最小值: {metrics['min_risk_score']:.4f}
最大值: {metrics['max_risk_score']:.4f}
中位数: {metrics['median_risk_score']:.4f}
平均检测数量: {metrics['avg_detection_count']:.1f}

📊 风险分布
-----------
极低风险 (0-0.1): {metrics['risk_distribution']['very_low']} 个
低风险 (0.1-0.3): {metrics['risk_distribution']['low']} 个
中等风险 (0.3-0.5): {metrics['risk_distribution']['medium']} 个
高风险 (0.5-0.7): {metrics['risk_distribution']['high']} 个
极高风险 (≥0.7): {metrics['risk_distribution']['very_high']} 个

🔍 检测统计
-----------
总检测次数: {metrics['detection_statistics']['total_detections']}
检测类型数量: {metrics['detection_statistics']['unique_detection_types']}

检测类型分布:
"""
    
    # 检测类型统计
    for det_type, count in sorted(metrics['detection_statistics']['type_distribution'].items(), 
                                key=lambda x: x[1], reverse=True):
        percentage = count / metrics['detection_statistics']['total_detections'] * 100 if metrics['detection_statistics']['total_detections'] > 0 else 0
        report += f"  {det_type}: {count} ({percentage:.1f}%)\n"
    
    # 性能指标（如果有）
    if 'performance' in metrics and isinstance(metrics['performance'], dict) and 'accuracy' in metrics['performance']:
        perf = metrics['performance']
        report += f"""
🎯 性能指标 (与真实标签对比)
-------------------------
准确率 (Accuracy): {perf['accuracy']:.4f} ({perf['accuracy']*100:.1f}%)
精确率 (Precision): {perf['precision']:.4f} ({perf['precision']*100:.1f}%)
召回率 (Recall): {perf['recall']:.4f} ({perf['recall']*100:.1f}%)
F1分数: {perf['f1_score']:.4f} ({perf['f1_score']*100:.1f}%)
特异性 (Specificity): {perf['specificity']:.4f} ({perf['specificity']*100:.1f}%)
假正率 (FPR): {perf['false_positive_rate']:.4f} ({perf['false_positive_rate']*100:.1f}%)
假负率 (FNR): {perf['false_negative_rate']:.4f} ({perf['false_negative_rate']*100:.1f}%)
ROC AUC: {perf['roc_auc']:.4f}

📊 混淆矩阵
-----------
真阴性 (TN): {metrics['confusion_matrix']['true_negatives']}
假阳性 (FP): {metrics['confusion_matrix']['false_positives']}
假阴性 (FN): {metrics['confusion_matrix']['false_negatives']}
真阳性 (TP): {metrics['confusion_matrix']['true_positives']}

覆盖率: {metrics.get('ground_truth_coverage', 0)*100:.1f}% ({metrics.get('ground_truth_matched', 0)}/{metrics['total_files']})
"""
    elif 'performance' in metrics:
        report += f"\n📊 性能指标: {metrics['performance']}\n"
    
    # 保存报告
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report

def save_comprehensive_results(results: List[Dict], metrics: Dict, output_dir: Path, timestamp: str):
    """保存综合结果"""
    
    # 1. 保存CSV格式的简要结果
    summary_results = []
    for result in results:
        summary = {
            'file_path': result['file'],
            'file_name': Path(result['file']).name,
            'is_malicious': result['is_malicious'],
            'risk_score': result['risk_score'],
            'detection_count': result['detection_count'],
            'file_size': result.get('file_size', 0),
            'processing_time': result.get('processing_time', 0)
        }
        
        # 添加主要检测类型
        detection_types = []
        for detection in result.get('detections', []):
            detection_types.append(detection.get('type', 'unknown'))
        summary['detection_types'] = '; '.join(set(detection_types))
        summary['unique_detection_types'] = len(set(detection_types))
        
        summary_results.append(summary)
    
    summary_file = output_dir / f"detection_summary_{timestamp}.csv"
    pd.DataFrame(summary_results).to_csv(summary_file, index=False, encoding='utf-8')
    
    # 2. 保存详细的JSON结果
    details_file = output_dir / f"detection_details_{timestamp}.json"
    save_results(results, str(details_file))
    
    # 3. 保存指标
    metrics_file = output_dir / f"detection_metrics_{timestamp}.json"
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, default=str, ensure_ascii=False)
    
    # 4. 保存恶意文件列表
    malicious_results = [r for r in results if r['is_malicious']]
    if malicious_results:
        malicious_file = output_dir / f"malicious_files_{timestamp}.json"
        save_results(malicious_results, str(malicious_file))
    
    # 5. 生成详细报告
    report_file = output_dir / f"detection_report_{timestamp}.txt"
    generate_detailed_report(metrics, str(report_file))
    
    return {
        'summary_csv': str(summary_file),
        'details_json': str(details_file),
        'metrics_json': str(metrics_file),
        'malicious_json': str(malicious_file) if malicious_results else None,
        'report_txt': str(report_file)
    }

def main():
    parser = argparse.ArgumentParser(description='运行提示词注入检测 (增强版)')
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
    parser.add_argument('--detector-type', type=str, default='standard',
                       choices=['standard', 'ensemble'],
                       help='检测器类型')
    parser.add_argument('--threshold', type=float,
                       help='风险分数阈值（覆盖配置文件）')
    parser.add_argument('--batch-size', type=int, default=20,
                       help='批处理大小')
    parser.add_argument('--ground-truth', type=str,
                       help='真实标签文件路径（.json或.csv）')
    parser.add_argument('--save-details', action='store_true',
                       help='保存详细检测结果')
    parser.add_argument('--export-csv', action='store_true', default=True,
                       help='导出CSV格式结果')
    parser.add_argument('--generate-report', action='store_true', default=True,
                       help='生成详细报告')
    parser.add_argument('--save-suspicious-only', action='store_true',
                       help='只保存可疑文件的详细结果')
    parser.add_argument('--max-files', type=int,
                       help='最大处理文件数（用于测试）')
    parser.add_argument('--log-file', type=str,
                       help='日志文件路径')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='详细输出')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='静默模式，只输出关键信息')
    
    args = parser.parse_args()
    
    # 设置日志
    if args.quiet:
        log_level = "WARNING"
    elif args.verbose:
        log_level = "DEBUG"
    else:
        log_level = "INFO"
    
    logger = setup_logging(log_level, args.log_file)
    
    if not args.quiet:
        logger.info("=" * 80)
        logger.info("🔍 提示词注入检测器 (增强版) 启动")
        logger.info("=" * 80)
    
    start_time = time.time()
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 命令行参数覆盖配置
        if args.output_dir:
            config['experiment']['output_dir'] = args.output_dir
        
        original_threshold = config['detection']['thresholds']['risk_score']
        if args.threshold:
            config['detection']['thresholds']['risk_score'] = args.threshold
            logger.info(f"阈值覆盖: {original_threshold} -> {args.threshold}")
        
        # 加载真实标签
        ground_truth = None
        if args.ground_truth:
            ground_truth = load_ground_truth(args.ground_truth)
            if ground_truth:
                logger.info(f"加载真实标签: {len(ground_truth)} 个文件")
            else:
                logger.warning(f"无法加载真实标签文件: {args.ground_truth}")
        
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
            logger.error("❌ 必须指定输入源：--single-file、--file-list 或 --input-dir")
            return 1
        
        if not pdf_files:
            logger.error("❌ 没有找到PDF文件！")
            return 1
        
        # 限制文件数量（用于测试）
        if args.max_files and len(pdf_files) > args.max_files:
            pdf_files = pdf_files[:args.max_files]
            logger.info(f"限制处理文件数量: {args.max_files}")
        
        logger.info(f"📄 找到 {len(pdf_files)} 个PDF文件")
        
        # 创建检测器
        if args.detector_type == 'ensemble':
            detector = EnsembleDetector(config)
            logger.info("🔗 使用集成检测器")
        else:
            detector = PromptInjectionDetector(config)
            logger.info("🔍 使用标准检测器")
        
        # 分批处理
        batch_size = args.batch_size
        total_batches = (len(pdf_files) + batch_size - 1) // batch_size
        
        all_results = []
        progress = ProgressTracker(len(pdf_files), "检测PDF文件")
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(pdf_files))
            batch_files = pdf_files[start_idx:end_idx]
            
            if not args.quiet:
                logger.info(f"🔄 处理批次 {batch_idx + 1}/{total_batches} "
                           f"({len(batch_files)} 个文件)")
            
            for file_path in batch_files:
                try:
                    file_start_time = time.time()
                    result = detector.detect_injection(file_path)
                    processing_time = time.time() - file_start_time
                    
                    # 添加处理时间和文件大小
                    result['processing_time'] = processing_time
                    try:
                        result['file_size'] = os.path.getsize(file_path)
                    except:
                        result['file_size'] = 0
                    
                    all_results.append(result)
                    
                    # 实时显示结果
                    risk_score = result['risk_score']
                    is_malicious = result['is_malicious']
                    detection_count = result['detection_count']
                    
                    if not args.quiet:
                        status = "🚨 恶意" if is_malicious else "✅ 正常"
                        file_name = Path(file_path).name
                        if args.verbose:
                            logger.info(f"{status} | {file_name} | "
                                       f"风险: {risk_score:.3f} | 检测: {detection_count} | "
                                       f"时间: {processing_time:.2f}s")
                        elif is_malicious:  # 非详细模式下只显示恶意文件
                            logger.warning(f"{status} | {file_name} | 风险: {risk_score:.3f}")
                    
                    progress.update()
                    
                except Exception as e:
                    logger.error(f"❌ 检测失败 {Path(file_path).name}: {e}")
                    progress.update()
                    continue
        
        progress.finish()
        
        total_time = time.time() - start_time
        
        # 计算性能指标
        logger.info("📊 计算性能指标...")
        metrics = calculate_performance_metrics(
            all_results, 
            ground_truth, 
            config['detection']['thresholds']['risk_score']
        )
        
        # 添加处理统计
        metrics['processing_statistics'] = {
            'total_processing_time': total_time,
            'avg_processing_time_per_file': total_time / len(all_results) if all_results else 0,
            'files_per_second': len(all_results) / total_time if total_time > 0 else 0
        }
        
        # 显示统计结果
        logger.info("=" * 80)
        logger.info("📊 检测完成 - 统计结果")
        logger.info("=" * 80)
        logger.info(f"总文件数: {metrics['total_files']}")
        logger.info(f"检测为恶意: {metrics['malicious_files']} ({metrics['malicious_rate']*100:.1f}%)")
        logger.info(f"平均风险分数: {metrics['avg_risk_score']:.4f} ± {metrics['std_risk_score']:.4f}")
        logger.info(f"平均检测数量: {metrics['avg_detection_count']:.1f}")
        logger.info(f"总处理时间: {total_time:.1f}秒 ({metrics['processing_statistics']['files_per_second']:.1f} 文件/秒)")
        
        # 显示性能指标
        if 'performance' in metrics and isinstance(metrics['performance'], dict) and 'accuracy' in metrics['performance']:
            perf = metrics['performance']
            logger.info("=" * 50)
            logger.info("🎯 性能指标 (与真实标签对比)")
            logger.info("=" * 50)
            logger.info(f"准确率: {perf['accuracy']:.4f} ({perf['accuracy']*100:.1f}%)")
            logger.info(f"精确率: {perf['precision']:.4f} ({perf['precision']*100:.1f}%)")
            logger.info(f"召回率: {perf['recall']:.4f} ({perf['recall']*100:.1f}%)")
            logger.info(f"F1分数: {perf['f1_score']:.4f} ({perf['f1_score']*100:.1f}%)")
            logger.info(f"ROC AUC: {perf['roc_auc']:.4f}")
        
        # 保存结果
        output_dir = Path(config['experiment']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存综合结果
        output_files = save_comprehensive_results(all_results, metrics, output_dir, timestamp)
        
        logger.info("=" * 50)
        logger.info("💾 结果保存完成")
        logger.info("=" * 50)
        for file_type, file_path in output_files.items():
            if file_path:
                logger.info(f"{file_type}: {file_path}")
        
        # 显示恶意文件报告
        malicious_results = [r for r in all_results if r['is_malicious']]
        if malicious_results and not args.quiet:
            logger.info("=" * 50)
            logger.info("🚨 检测到的恶意文件")
            logger.info("=" * 50)
            
            for i, result in enumerate(malicious_results[:10], 1):  # 只显示前10个
                file_name = Path(result['file']).name
                risk_score = result['risk_score']
                detection_types = set()
                
                for detection in result.get('detections', []):
                    detection_types.add(detection.get('type', 'unknown'))
                
                logger.info(f"{i:2d}. {file_name}")
                logger.info(f"    风险分数: {risk_score:.4f}")
                logger.info(f"    检测类型: {', '.join(sorted(detection_types))}")
                logger.info("")
            
            if len(malicious_results) > 10:
                logger.info(f"... 还有 {len(malicious_results) - 10} 个恶意文件")
        
        # 检测类型统计
        if metrics['detection_statistics']['type_distribution'] and not args.quiet:
            logger.info("=" * 50)
            logger.info("📈 检测类型统计")
            logger.info("=" * 50)
            for det_type, count in sorted(metrics['detection_statistics']['type_distribution'].items(), 
                                        key=lambda x: x[1], reverse=True):
                percentage = count / metrics['detection_statistics']['total_detections'] * 100
                logger.info(f"  {det_type}: {count} ({percentage:.1f}%)")
        
        logger.info("=" * 80)
        logger.info("🎉 检测任务完成")
        logger.info("=" * 80)
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("⚠️  用户中断检测")
        return 1
    except Exception as e:
        logger.error(f"❌ 检测过程中发生错误: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())
