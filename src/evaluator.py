import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import json
from pathlib import Path
from .utils import setup_logging, ensure_dir, save_results

logger = setup_logging()

class ExperimentEvaluator:
    """实验评估器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.experiment_config = config['experiment']
        self.output_dir = ensure_dir(self.experiment_config['output_dir'])
        self.results_history = []
        
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        logger.info(f"实验评估器初始化完成，输出目录: {self.output_dir}")
    
    def evaluate_detection_performance(self, 
                                     clean_files: List[str], 
                                     attack_files: List[str], 
                                     detector,
                                     attack_info: Optional[List[Dict]] = None) -> Tuple[pd.DataFrame, Dict]:
        """评估检测性能"""
        
        logger.info(f"开始性能评估: {len(clean_files)} 个正常文件, {len(attack_files)} 个攻击文件")
        
        all_results = []
        
        # 测试正常文件（负样本）
        logger.info("测试正常文件...")
        for i, file_path in enumerate(clean_files):
            try:
                result = detector.detect_injection(file_path)
                
                file_result = {
                    'file_path': file_path,
                    'file_name': Path(file_path).name,
                    'label': 0,  # 正常文件
                    'predicted': 1 if result['is_malicious'] else 0,
                    'risk_score': result['risk_score'],
                    'detection_count': result['detection_count'],
                    'file_type': 'clean',
                    'attack_type': None,
                    'language': None
                }
                
                # 添加详细检测信息
                detection_types = [d['type'] for d in result.get('detections', [])]
                file_result['detection_types'] = ', '.join(set(detection_types))
                
                all_results.append(file_result)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"已处理正常文件: {i + 1}/{len(clean_files)}")
                    
            except Exception as e:
                logger.error(f"处理正常文件失败 {file_path}: {e}")
        
        # 测试攻击文件（正样本）
        logger.info("测试攻击文件...")
        attack_info_dict = {}
        if attack_info:
            attack_info_dict = {info['attack_file']: info for info in attack_info}
        
        for i, file_path in enumerate(attack_files):
            try:
                result = detector.detect_injection(file_path)
                
                # 获取攻击信息
                attack_details = attack_info_dict.get(file_path, {})
                
                file_result = {
                    'file_path': file_path,
                    'file_name': Path(file_path).name,
                    'label': 1,  # 攻击文件
                    'predicted': 1 if result['is_malicious'] else 0,
                    'risk_score': result['risk_score'],
                    'detection_count': result['detection_count'],
                    'file_type': 'attack',
                    'attack_type': attack_details.get('attack_type', 'unknown'),
                    'language': attack_details.get('language', 'unknown')
                }
                
                # 添加详细检测信息
                detection_types = [d['type'] for d in result.get('detections', [])]
                file_result['detection_types'] = ', '.join(set(detection_types))
                
                all_results.append(file_result)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"已处理攻击文件: {i + 1}/{len(attack_files)}")
                    
            except Exception as e:
                logger.error(f"处理攻击文件失败 {file_path}: {e}")
        
        # 创建结果DataFrame
        df_results = pd.DataFrame(all_results)
        
        if df_results.empty:
            logger.error("没有有效的检测结果")
            return df_results, {}
        
        # 计算评估指标
        metrics = self._calculate_metrics(df_results)
        
        # 保存详细结果
        self._save_detailed_results(df_results, metrics)
        
        logger.info("性能评估完成")
        return df_results, metrics
    
    def _calculate_metrics(self, df_results: pd.DataFrame) -> Dict[str, Any]:
        """计算评估指标"""
        y_true = df_results['label'].values
        y_pred = df_results['predicted'].values
        y_scores = df_results['risk_score'].values
        
        # 基础指标
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'support': {
                'clean': np.sum(y_true == 0),
                'attack': np.sum(y_true == 1)
            }
        }
        
        # ROC AUC（如果有概率分数）
        if len(np.unique(y_scores)) > 1:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
            except Exception as e:
                logger.warning(f"ROC AUC计算失败: {e}")
                metrics['roc_auc'] = 0.0
        else:
            metrics['roc_auc'] = 0.0
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['confusion_matrix_details'] = {
                'true_negative': int(tn),
                'false_positive': int(fp),
                'false_negative': int(fn),
                'true_positive': int(tp)
            }
            
            # 计算误报率和漏报率
            metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # 按攻击类型的性能
        if 'attack_type' in df_results.columns:
            attack_performance = {}
            for attack_type in df_results['attack_type'].dropna().unique():
                mask = df_results['attack_type'] == attack_type
                if mask.sum() > 0:
                    attack_data = df_results[mask]
                    attack_performance[attack_type] = {
                        'count': len(attack_data),
                        'detection_rate': attack_data['predicted'].mean(),
                        'avg_risk_score': attack_data['risk_score'].mean()
                    }
            metrics['performance_by_attack_type'] = attack_performance
        
        # 按语言的性能
        if 'language' in df_results.columns:
            language_performance = {}
            for language in df_results['language'].dropna().unique():
                mask = df_results['language'] == language
                if mask.sum() > 0:
                    lang_data = df_results[mask]
                    language_performance[language] = {
                        'count': len(lang_data),
                        'detection_rate': lang_data['predicted'].mean(),
                        'avg_risk_score': lang_data['risk_score'].mean()
                    }
            metrics['performance_by_language'] = language_performance
        
        return metrics
    
    def _save_detailed_results(self, df_results: pd.DataFrame, metrics: Dict):
        """保存详细结果"""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存CSV结果
        csv_file = Path(self.output_dir) / f"detection_results_{timestamp}.csv"
        df_results.to_csv(csv_file, index=False, encoding='utf-8')
        
        # 保存JSON指标
        json_file = Path(self.output_dir) / f"metrics_{timestamp}.json"
        save_results(metrics, str(json_file))
        
        logger.info(f"详细结果已保存: {csv_file}, {json_file}")
    
    def plot_performance_analysis(self, df_results: pd.DataFrame, 
                                metrics: Dict, save_plots: bool = True) -> Dict[str, Any]:
        """绘制性能分析图表"""
        
        config = self.config['experiment']['visualization']
        figsize = tuple(config['figsize'])
        dpi = config['dpi']
        
        # 创建图表
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 混淆矩阵 (2x3 grid, position 1)
        ax1 = plt.subplot(3, 4, 1)
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'])
        ax1.set_title('Confusion Matrix')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # 2. 风险分数分布 (position 2)
        ax2 = plt.subplot(3, 4, 2)
        clean_scores = df_results[df_results['label']==0]['risk_score']
        attack_scores = df_results[df_results['label']==1]['risk_score']
        
        ax2.hist(clean_scores, alpha=0.7, label='Normal Files', bins=20, density=True)
        ax2.hist(attack_scores, alpha=0.7, label='Attack Files', bins=20, density=True)
        ax2.axvline(x=self.config['detection']['thresholds']['risk_score'], 
                   color='red', linestyle='--', label='Threshold')
        ax2.set_title('Risk Score Distribution')
        ax2.set_xlabel('Risk Score')
        ax2.set_ylabel('Density')
        ax2.legend()
        
        # 3. 性能指标条形图 (position 3)
        ax3 = plt.subplot(3, 4, 3)
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        metric_values = [
            metrics['accuracy'], metrics['precision'], 
            metrics['recall'], metrics['f1_score'], metrics['roc_auc']
        ]
        
        bars = ax3.bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'orange', 'pink', 'lightcoral'])
        ax3.set_title('Performance Metrics')
        ax3.set_ylabel('Score')
        ax3.set_ylim(0, 1)
        
        # 在条形图上添加数值
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        
        # 4. 检测数量箱线图 (position 4)
        ax4 = plt.subplot(3, 4, 4)
        detection_data = [
            df_results[df_results['label']==0]['detection_count'],
            df_results[df_results['label']==1]['detection_count']
        ]
        ax4.boxplot(detection_data, labels=['Normal', 'Attack'])
        ax4.set_title('Detection Count Distribution')
        ax4.set_ylabel('Number of Detections')
        
        # 5. ROC曲线 (position 5)
        ax5 = plt.subplot(3, 4, 5)
        try:
            fpr, tpr, _ = roc_curve(df_results['label'], df_results['risk_score'])
            ax5.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {metrics["roc_auc"]:.3f})')
            ax5.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax5.set_xlim([0.0, 1.0])
            ax5.set_ylim([0.0, 1.05])
            ax5.set_xlabel('False Positive Rate')
            ax5.set_ylabel('True Positive Rate')
            ax5.set_title('ROC Curve')
            ax5.legend(loc="lower right")
        except Exception as e:
            ax5.text(0.5, 0.5, f'ROC curve error: {str(e)}', 
                    ha='center', va='center', transform=ax5.transAxes)
        
        # 6. 精确率-召回率曲线 (position 6)
        ax6 = plt.subplot(3, 4, 6)
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(
                df_results['label'], df_results['risk_score'])
            ax6.plot(recall_curve, precision_curve, color='blue', lw=2)
            ax6.set_xlabel('Recall')
            ax6.set_ylabel('Precision')
            ax6.set_title('Precision-Recall Curve')
            ax6.set_xlim([0.0, 1.0])
            ax6.set_ylim([0.0, 1.05])
        except Exception as e:
            ax6.text(0.5, 0.5, f'PR curve error: {str(e)}', 
                    ha='center', va='center', transform=ax6.transAxes)
        
        # 7. 按攻击类型的性能 (position 7-8)
        if 'performance_by_attack_type' in metrics:
            ax7 = plt.subplot(3, 4, (7, 8))
            attack_types = list(metrics['performance_by_attack_type'].keys())
            detection_rates = [metrics['performance_by_attack_type'][at]['detection_rate'] 
                             for at in attack_types]
            
            bars = ax7.bar(attack_types, detection_rates, color='lightblue')
            ax7.set_title('Detection Rate by Attack Type')
            ax7.set_ylabel('Detection Rate')
            ax7.set_ylim(0, 1)
            plt.xticks(rotation=45)
            
            # 添加数值标签
            for bar, rate in zip(bars, detection_rates):
                height = bar.get_height()
                ax7.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{rate:.3f}', ha='center', va='bottom')
        
        # 8. 按语言的性能 (position 9-10)
        if 'performance_by_language' in metrics:
            ax8 = plt.subplot(3, 4, (9, 10))
            languages = list(metrics['performance_by_language'].keys())
            detection_rates = [metrics['performance_by_language'][lang]['detection_rate'] 
                             for lang in languages]
            
            bars = ax8.bar(languages, detection_rates, color='lightgreen')
            ax8.set_title('Detection Rate by Language')
            ax8.set_ylabel('Detection Rate')
            ax8.set_ylim(0, 1)
            plt.xticks(rotation=45)
            
            # 添加数值标签
            for bar, rate in zip(bars, detection_rates):
                height = bar.get_height()
                ax8.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{rate:.3f}', ha='center', va='bottom')
        
        # 9. 风险分数散点图 (position 11-12)
        ax9 = plt.subplot(3, 4, (11, 12))
        
        # 正常文件
        clean_data = df_results[df_results['label']==0]
        ax9.scatter(range(len(clean_data)), clean_data['risk_score'], 
                   alpha=0.6, label='Normal Files', color='blue', s=10)
        
        # 攻击文件
        attack_data = df_results[df_results['label']==1]
        ax9.scatter(range(len(clean_data), len(clean_data) + len(attack_data)), 
                   attack_data['risk_score'], alpha=0.6, label='Attack Files', 
                   color='red', s=10)
        
        ax9.axhline(y=self.config['detection']['thresholds']['risk_score'], 
                   color='green', linestyle='--', label='Threshold')
        ax9.set_title('Risk Score Distribution by File')
        ax9.set_xlabel('File Index')
        ax9.set_ylabel('Risk Score')
        ax9.legend()
        
        plt.tight_layout()
        
        # 保存图表
        if save_plots:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            plot_file = Path(self.output_dir) / f"performance_analysis_{timestamp}.png"
            plt.savefig(plot_file, dpi=dpi, bbox_inches='tight')
            logger.info(f"性能分析图表已保存: {plot_file}")
        
        plt.show()
        
        return {'figure': fig}
    
    def generate_report(self, df_results: pd.DataFrame, metrics: Dict) -> str:
        """生成详细报告"""
        
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
# 论文审稿攻击检测实验报告

生成时间: {timestamp}

## 1. 实验概述

### 数据集统计
- 总文件数: {len(df_results)}
- 正常文件数: {metrics['support']['clean']}
- 攻击文件数: {metrics['support']['attack']}
- 攻击文件比例: {metrics['support']['attack'] / len(df_results):.1%}

## 2. 检测性能

### 总体性能指标
- **准确率 (Accuracy)**: {metrics['accuracy']:.3f}
- **精确率 (Precision)**: {metrics['precision']:.3f}
- **召回率 (Recall)**: {metrics['recall']:.3f}
- **F1分数**: {metrics['f1_score']:.3f}
- **ROC AUC**: {metrics['roc_auc']:.3f}

### 混淆矩阵
"""
        
        if 'confusion_matrix_details' in metrics:
            cm_details = metrics['confusion_matrix_details']
            report += f"""
- 真阴性 (True Negative): {cm_details['true_negative']}
- 假阳性 (False Positive): {cm_details['false_positive']}
- 假阴性 (False Negative): {cm_details['false_negative']}
- 真阳性 (True Positive): {cm_details['true_positive']}

### 误报分析
- 误报率 (False Positive Rate): {metrics['false_positive_rate']:.3f}
- 漏报率 (False Negative Rate): {metrics['false_negative_rate']:.3f}
"""
        
        # 按攻击类型的性能
        if 'performance_by_attack_type' in metrics:
            report += "\n## 3. 按攻击类型的检测性能\n\n"
            for attack_type, perf in metrics['performance_by_attack_type'].items():
                report += f"### {attack_type}\n"
                report += f"- 样本数量: {perf['count']}\n"
                report += f"- 检测率: {perf['detection_rate']:.3f}\n"
                report += f"- 平均风险分数: {perf['avg_risk_score']:.3f}\n\n"
        
        # 按语言的性能
        if 'performance_by_language' in metrics:
            report += "\n## 4. 按语言的检测性能\n\n"
            for language, perf in metrics['performance_by_language'].items():
                report += f"### {language}\n"
                report += f"- 样本数量: {perf['count']}\n"
                report += f"- 检测率: {perf['detection_rate']:.3f}\n"
                report += f"- 平均风险分数: {perf['avg_risk_score']:.3f}\n\n"
        
        # 风险分数分析
        clean_scores = df_results[df_results['label']==0]['risk_score']
        attack_scores = df_results[df_results['label']==1]['risk_score']
        
        report += f"""
## 5. 风险分数分析

### 正常文件风险分数
- 平均值: {clean_scores.mean():.3f}
- 标准差: {clean_scores.std():.3f}
- 最大值: {clean_scores.max():.3f}
- 超过阈值的比例: {(clean_scores > self.config['detection']['thresholds']['risk_score']).mean():.3f}

### 攻击文件风险分数
- 平均值: {attack_scores.mean():.3f}
- 标准差: {attack_scores.std():.3f}
- 最小值: {attack_scores.min():.3f}
- 超过阈值的比例: {(attack_scores > self.config['detection']['thresholds']['risk_score']).mean():.3f}

## 6. 建议和改进方向

### 基于实验结果的建议:
"""
        
        # 根据结果提供建议
        if metrics['false_positive_rate'] > 0.1:
            report += "- **高误报率**: 建议调整检测阈值或优化关键词库以减少误报\n"
        
        if metrics['false_negative_rate'] > 0.1:
            report += "- **高漏报率**: 建议增强检测算法或添加新的检测维度\n"
        
        if metrics['roc_auc'] < 0.8:
            report += "- **ROC AUC较低**: 建议改进风险分数计算方法\n"
        
        # 保存报告
        timestamp_file = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        report_file = Path(self.output_dir) / f"experiment_report_{timestamp_file}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"实验报告已保存: {report_file}")
        
        return report
    
    def compare_experiments(self, experiment_results: List[Dict]) -> Dict:
        """比较多个实验结果"""
        if len(experiment_results) < 2:
            logger.warning("需要至少2个实验结果进行比较")
            return {}
        
        comparison = {
            'experiment_count': len(experiment_results),
            'metrics_comparison': {},
            'best_experiment': None,
            'improvement_suggestions': []
        }
        
        # 比较指标
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        for metric in metrics_to_compare:
            values = [exp.get(metric, 0) for exp in experiment_results]
            comparison['metrics_comparison'][metric] = {
                'values': values,
                'best_index': np.argmax(values),
                'best_value': max(values),
                'worst_value': min(values),
                'improvement': max(values) - min(values)
            }
        
        # 确定最佳实验（基于F1分数）
        f1_scores = [exp.get('f1_score', 0) for exp in experiment_results]
        best_idx = np.argmax(f1_scores)
        comparison['best_experiment'] = {
            'index': best_idx,
            'f1_score': f1_scores[best_idx]
        }
        
        return comparison