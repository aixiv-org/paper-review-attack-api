import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc
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
                    'language': None,
                    'detections': result.get('detections', [])  # 保存详细检测信息
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
                    'attack_type': attack_details.get('attack_type', self._extract_attack_type_from_filename(file_path)),
                    'language': attack_details.get('language', self._extract_language_from_filename(file_path)),
                    'detections': result.get('detections', [])
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
    
    def _extract_attack_type_from_filename(self, file_path: str) -> str:
        """从文件名提取攻击类型"""
        filename = Path(file_path).name.lower()
        if 'white_text' in filename:
            return 'white_text'
        elif 'metadata' in filename:
            return 'metadata'
        elif 'invisible' in filename:
            return 'invisible_chars'
        elif 'mixed' in filename:
            return 'mixed_language'
        elif 'steganographic' in filename:
            return 'steganographic'
        else:
            return 'unknown'
    
    def _extract_language_from_filename(self, file_path: str) -> str:
        """从文件名提取语言"""
        filename = Path(file_path).name.lower()
        if 'english' in filename:
            return 'english'
        elif 'chinese' in filename:
            return 'chinese'
        elif 'japanese' in filename:
            return 'japanese'
        elif 'mixed' in filename:
            return 'mixed'
        else:
            return 'unknown'
    
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
                'clean': int(np.sum(y_true == 0)),
                'attack': int(np.sum(y_true == 1))
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
                if attack_type and attack_type != 'unknown':
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
                if language and language != 'unknown':
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
        """绘制性能分析图表 - 分离版本"""
        
        config = self.config['experiment']['visualization']
        figsize = tuple(config.get('figsize', [12, 8]))
        dpi = config.get('dpi', 300)
        
        # 创建输出目录
        if save_plots:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            plots_dir = Path(self.output_dir) / f"plots_{timestamp}"
            plots_dir.mkdir(parents=True, exist_ok=True)
        
        plot_files = {}
        
        # 1. 混淆矩阵
        logger.info("生成混淆矩阵...")
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        cm = np.array(metrics['confusion_matrix'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'],
                   cbar_kws={'label': 'Count'},
                   annot_kws={'size': 16, 'weight': 'bold'})
        
        ax1.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Predicted Label', fontsize=12)
        ax1.set_ylabel('True Label', fontsize=12)
        
        # 添加准确率信息
        accuracy = metrics['accuracy']
        ax1.text(0.5, -0.15, f'Accuracy: {accuracy:.3f}', 
                 ha='center', transform=ax1.transAxes, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        if save_plots:
            plot_file = plots_dir / "01_confusion_matrix.png"
            plt.savefig(plot_file, dpi=dpi, bbox_inches='tight')
            plot_files['confusion_matrix'] = plot_file
            logger.info(f"混淆矩阵已保存: {plot_file}")
        plt.show()
        plt.close()
        
        # 2. 风险分数分布
        logger.info("生成风险分数分布图...")
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        clean_scores = df_results[df_results['label']==0]['risk_score']
        attack_scores = df_results[df_results['label']==1]['risk_score']
        
        # 绘制直方图
        bins = np.linspace(0, 1, 21)
        alpha = 0.7
        
        if len(clean_scores) > 0:
            ax2.hist(clean_scores, alpha=alpha, label=f'Normal Files (n={len(clean_scores)})', 
                     bins=bins, density=True, color='skyblue', edgecolor='navy', linewidth=1)
        
        if len(attack_scores) > 0:
            ax2.hist(attack_scores, alpha=alpha, label=f'Attack Files (n={len(attack_scores)})', 
                     bins=bins, density=True, color='lightcoral', edgecolor='darkred', linewidth=1)
        
        # 添加阈值线
        threshold = self.config['detection']['thresholds']['risk_score']
        ax2.axvline(x=threshold, color='red', linestyle='--', linewidth=3, 
                   label=f'Threshold ({threshold})')
        
        ax2.set_title('Risk Score Distribution', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Risk Score', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.legend(fontsize=11, loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 添加统计信息
        if len(clean_scores) > 0 and len(attack_scores) > 0:
            stats_text = f"""Normal Files:
Mean: {clean_scores.mean():.3f}
Std: {clean_scores.std():.3f}

Attack Files:
Mean: {attack_scores.mean():.3f}
Std: {attack_scores.std():.3f}"""
            
            ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
                     verticalalignment='top', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        if save_plots:
            plot_file = plots_dir / "02_risk_score_distribution.png"
            plt.savefig(plot_file, dpi=dpi, bbox_inches='tight')
            plot_files['risk_distribution'] = plot_file
            logger.info(f"风险分数分布图已保存: {plot_file}")
        plt.show()
        plt.close()
        
        # 3. 性能指标条形图
        logger.info("生成性能指标图...")
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        metric_values = [
            metrics['accuracy'], metrics['precision'], 
            metrics['recall'], metrics['f1_score'], metrics['roc_auc']
        ]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        bars = ax3.bar(metric_names, metric_values, color=colors, alpha=0.8, 
                      edgecolor='black', linewidth=1.5)
        
        # 添加数值标签
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax3.set_title('Performance Metrics', fontsize=16, fontweight='bold')
        ax3.set_ylabel('Score', fontsize=12)
        ax3.set_ylim(0, 1.1)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 添加基准线
        ax3.axhline(y=0.8, color='red', linestyle=':', alpha=0.7, linewidth=2, 
                   label='Good Baseline (0.8)')
        ax3.legend(loc='upper right')
        
        plt.xticks(rotation=0)
        plt.tight_layout()
        if save_plots:
            plot_file = plots_dir / "03_performance_metrics.png"
            plt.savefig(plot_file, dpi=dpi, bbox_inches='tight')
            plot_files['performance_metrics'] = plot_file
            logger.info(f"性能指标图已保存: {plot_file}")
        plt.show()
        plt.close()
        
        # 4. ROC曲线和PR曲线
        logger.info("生成ROC和PR曲线...")
        fig4, (ax4_1, ax4_2) = plt.subplots(1, 2, figsize=(15, 6))
        
        try:
            # ROC曲线
            fpr, tpr, _ = roc_curve(df_results['label'], df_results['risk_score'])
            roc_auc = auc(fpr, tpr)
            
            ax4_1.plot(fpr, tpr, color='darkorange', lw=3, 
                      label=f'ROC Curve (AUC = {roc_auc:.3f})')
            ax4_1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                      label='Random Classifier')
            ax4_1.set_xlim([0.0, 1.0])
            ax4_1.set_ylim([0.0, 1.05])
            ax4_1.set_xlabel('False Positive Rate', fontsize=12)
            ax4_1.set_ylabel('True Positive Rate', fontsize=12)
            ax4_1.set_title('ROC Curve', fontsize=14, fontweight='bold')
            ax4_1.legend(loc="lower right")
            ax4_1.grid(True, alpha=0.3)
            
            # PR曲线
            precision_curve, recall_curve, _ = precision_recall_curve(
                df_results['label'], df_results['risk_score'])
            pr_auc = auc(recall_curve, precision_curve)
            
            ax4_2.plot(recall_curve, precision_curve, color='blue', lw=3,
                      label=f'PR Curve (AUC = {pr_auc:.3f})')
            ax4_2.set_xlabel('Recall', fontsize=12)
            ax4_2.set_ylabel('Precision', fontsize=12)
            ax4_2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
            ax4_2.set_xlim([0.0, 1.0])
            ax4_2.set_ylim([0.0, 1.05])
            ax4_2.legend()
            ax4_2.grid(True, alpha=0.3)
            
        except Exception as e:
            logger.error(f"ROC/PR曲线生成失败: {e}")
            ax4_1.text(0.5, 0.5, f'ROC curve error: {str(e)}', 
                      ha='center', va='center', transform=ax4_1.transAxes)
            ax4_2.text(0.5, 0.5, f'PR curve error: {str(e)}', 
                      ha='center', va='center', transform=ax4_2.transAxes)
        
        plt.tight_layout()
        if save_plots:
            plot_file = plots_dir / "04_roc_pr_curves.png"
            plt.savefig(plot_file, dpi=dpi, bbox_inches='tight')
            plot_files['roc_pr_curves'] = plot_file
            logger.info(f"ROC和PR曲线已保存: {plot_file}")
        plt.show()
        plt.close()
        
        # 5. 按攻击类型的性能
        if 'performance_by_attack_type' in metrics and metrics['performance_by_attack_type']:
            logger.info("生成攻击类型性能图...")
            fig5, ax5 = plt.subplots(figsize=(12, 6))
            attack_types = list(metrics['performance_by_attack_type'].keys())
            detection_rates = [metrics['performance_by_attack_type'][at]['detection_rate'] 
                              for at in attack_types]
            counts = [metrics['performance_by_attack_type'][at]['count'] 
                     for at in attack_types]
            
            bars = ax5.bar(attack_types, detection_rates, color='lightblue', 
                          alpha=0.8, edgecolor='darkblue', linewidth=1.5)
            
            # 添加样本数量标签
            for bar, rate, count in zip(bars, detection_rates, counts):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{rate:.3f}\n(n={count})', ha='center', va='bottom', 
                        fontweight='bold')
            
            ax5.set_title('Detection Rate by Attack Type', fontsize=16, fontweight='bold')
            ax5.set_ylabel('Detection Rate', fontsize=12)
            ax5.set_ylim(0, 1.1)
            ax5.grid(True, alpha=0.3, axis='y')
            plt.xticks(rotation=45)
            plt.tight_layout()
            if save_plots:
                plot_file = plots_dir / "05_attack_type_performance.png"
                plt.savefig(plot_file, dpi=dpi, bbox_inches='tight')
                plot_files['attack_type_performance'] = plot_file
                logger.info(f"攻击类型性能图已保存: {plot_file}")
            plt.show()
            plt.close()
        
        # 6. 按语言的性能
        if 'performance_by_language' in metrics and metrics['performance_by_language']:
            logger.info("生成语言性能图...")
            fig6, ax6 = plt.subplots(figsize=(10, 6))
            languages = list(metrics['performance_by_language'].keys())
            detection_rates = [metrics['performance_by_language'][lang]['detection_rate'] 
                              for lang in languages]
            avg_scores = [metrics['performance_by_language'][lang]['avg_risk_score'] 
                         for lang in languages]
            counts = [metrics['performance_by_language'][lang]['count'] 
                     for lang in languages]
            
            # 双y轴图
            ax6_twin = ax6.twinx()
            
            x_pos = np.arange(len(languages))
            width = 0.35
            
            bars1 = ax6.bar(x_pos - width/2, detection_rates, width, 
                           label='Detection Rate', color='lightblue', alpha=0.8,
                           edgecolor='darkblue')
            bars2 = ax6_twin.bar(x_pos + width/2, avg_scores, width,
                                label='Avg Risk Score', color='lightcoral', alpha=0.8,
                                edgecolor='darkred')
            
            # 添加标签
            for i, (bar1, bar2, rate, score, count) in enumerate(zip(bars1, bars2, detection_rates, avg_scores, counts)):
                ax6.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height() + 0.01,
                        f'{rate:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                ax6_twin.text(bar2.get_x() + bar2.get_width()/2., bar2.get_height() + 0.01,
                             f'{score:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
                ax6.text(i, -0.15, f'n={count}', ha='center', va='top', 
                        transform=ax6.get_xaxis_transform(), fontsize=9)
            
            ax6.set_title('Performance by Language', fontsize=16, fontweight='bold')
            ax6.set_ylabel('Detection Rate', fontsize=12, color='blue')
            ax6_twin.set_ylabel('Average Risk Score', fontsize=12, color='red')
            ax6.set_xticks(x_pos)
            ax6.set_xticklabels(languages)
            ax6.set_ylim(0, 1.1)
            ax6_twin.set_ylim(0, 1.1)
            
            # 图例
            lines1, labels1 = ax6.get_legend_handles_labels()
            lines2, labels2 = ax6_twin.get_legend_handles_labels()
            ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            ax6.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            if save_plots:
                plot_file = plots_dir / "06_language_performance.png"
                plt.savefig(plot_file, dpi=dpi, bbox_inches='tight')
                plot_files['language_performance'] = plot_file
                logger.info(f"语言性能图已保存: {plot_file}")
            plt.show()
            plt.close()
        
        # 7. 风险分数散点图
        logger.info("生成风险分数散点图...")
        fig7, ax7 = plt.subplots(figsize=(14, 6))
        
        # 正常文件
        clean_data = df_results[df_results['label']==0]
        if len(clean_data) > 0:
            ax7.scatter(range(len(clean_data)), clean_data['risk_score'], 
                       alpha=0.7, label=f'Normal Files (n={len(clean_data)})', 
                       color='blue', s=40, marker='o', edgecolors='darkblue')
        
        # 攻击文件
        attack_data = df_results[df_results['label']==1]
        if len(attack_data) > 0:
            attack_start = len(clean_data) if len(clean_data) > 0 else 0
            ax7.scatter(range(attack_start, attack_start + len(attack_data)), 
                       attack_data['risk_score'], alpha=0.7, 
                       label=f'Attack Files (n={len(attack_data)})', 
                       color='red', s=40, marker='^', edgecolors='darkred')
        
        # 阈值线
        threshold = self.config['detection']['thresholds']['risk_score']
        ax7.axhline(y=threshold, color='green', linestyle='--', linewidth=3,
                   label=f'Threshold ({threshold})')
        
        ax7.set_title('Risk Score Distribution by File', fontsize=16, fontweight='bold')
        ax7.set_xlabel('File Index', fontsize=12)
        ax7.set_ylabel('Risk Score', fontsize=12)
        ax7.legend(fontsize=11)
        ax7.grid(True, alpha=0.3)
        ax7.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        if save_plots:
            plot_file = plots_dir / "07_risk_score_scatter.png"
            plt.savefig(plot_file, dpi=dpi, bbox_inches='tight')
            plot_files['risk_score_scatter'] = plot_file
            logger.info(f"风险分数散点图已保存: {plot_file}")
        plt.show()
        plt.close()
        
        # 8. 检测类型统计
        logger.info("生成检测类型统计图...")
        fig8, ax8 = plt.subplots(figsize=(12, 8))
        
        # 统计检测类型
        detection_type_counts = {}
        for _, result in df_results.iterrows():
            for detection in result.get('detections', []):
                det_type = detection.get('type', 'unknown')
                detection_type_counts[det_type] = detection_type_counts.get(det_type, 0) + 1
        
        if detection_type_counts:
            # 按数量排序
            sorted_items = sorted(detection_type_counts.items(), key=lambda x: x[1], reverse=True)
            types = [item[0] for item in sorted_items]
            counts = [item[1] for item in sorted_items]
            
            # 横向条形图
            bars = ax8.barh(types, counts, color='lightgreen', alpha=0.8, 
                           edgecolor='darkgreen', linewidth=1.5)
            
            # 添加数值标签
            for bar, count in zip(bars, counts):
                width = bar.get_width()
                ax8.text(width + max(counts) * 0.01, bar.get_y() + bar.get_height()/2.,
                        f'{count}', ha='left', va='center', fontweight='bold', fontsize=11)
            
            ax8.set_title('Detection Type Frequency', fontsize=16, fontweight='bold')
            ax8.set_xlabel('Count', fontsize=12)
            ax8.set_ylabel('Detection Type', fontsize=12)
            ax8.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            if save_plots:
                plot_file = plots_dir / "08_detection_type_stats.png"
                plt.savefig(plot_file, dpi=dpi, bbox_inches='tight')
                plot_files['detection_type_stats'] = plot_file
                logger.info(f"检测类型统计图已保存: {plot_file}")
            plt.show()
            plt.close()
        
        # 9. 检测数量分布
        logger.info("生成检测数量分布图...")
        fig9, ax9 = plt.subplots(figsize=(10, 6))
        
        clean_detection_counts = df_results[df_results['label']==0]['detection_count']
        attack_detection_counts = df_results[df_results['label']==1]['detection_count']
        
        box_data = []
        labels = []
        if len(clean_detection_counts) > 0:
            box_data.append(clean_detection_counts)
            labels.append(f'Normal\n(n={len(clean_detection_counts)})')
        
        if len(attack_detection_counts) > 0:
            box_data.append(attack_detection_counts)
            labels.append(f'Attack\n(n={len(attack_detection_counts)})')
        
        if box_data:
            bp = ax9.boxplot(box_data, labels=labels, patch_artist=True)
            
            # 设置颜色
            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)
        
        ax9.set_title('Detection Count Distribution', fontsize=16, fontweight='bold')
        ax9.set_ylabel('Number of Detections', fontsize=12)
        ax9.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_plots:
            plot_file = plots_dir / "09_detection_count_distribution.png"
            plt.savefig(plot_file, dpi=dpi, bbox_inches='tight')
            plot_files['detection_count_distribution'] = plot_file
            logger.info(f"检测数量分布图已保存: {plot_file}")
        plt.show()
        plt.close()
        
        # 生成图表索引HTML文件
        if save_plots:
            self._generate_plots_index(plots_dir, plot_files, metrics)
            logger.info(f"所有图表已保存到: {plots_dir}")
        
        return {'plots_directory': plots_dir if save_plots else None, 'plot_files': plot_files}
    
    def _generate_plots_index(self, plots_dir: Path, plot_files: Dict[str, Path], metrics: Dict):
        """生成图表索引HTML文件"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Detection Performance Analysis Report</title>
    <meta charset="UTF-8">
    <style>
        body {{ 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .metrics-summary {{
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .metric-card {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #27ae60;
        }}
        .plot-container {{ 
            margin: 20px 0; 
            padding: 20px; 
            border: 1px solid #ddd; 
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .plot-title {{ 
            font-size: 18px; 
            font-weight: bold; 
            margin-bottom: 10px; 
            color: #2c3e50;
        }}
        img {{ 
            max-width: 100%; 
            height: auto; 
            border-radius: 5px;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-style: italic;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Detection Performance Analysis Report</h1>
        <p class="timestamp">Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    
    <div class="metrics-summary">
        <h2>📈 Performance Summary</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{metrics.get('accuracy', 0):.3f}</div>
                <div>Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('precision', 0):.3f}</div>
                <div>Precision</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('recall', 0):.3f}</div>
                <div>Recall</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('f1_score', 0):.3f}</div>
                <div>F1-Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.get('roc_auc', 0):.3f}</div>
                <div>ROC AUC</div>
            </div>
        </div>
        
        <h3>📁 Dataset Information</h3>
        <p>Total Files: {metrics.get('support', {}).get('clean', 0) + metrics.get('support', {}).get('attack', 0)}</p>
        <p>Normal Files: {metrics.get('support', {}).get('clean', 0)} | Attack Files: {metrics.get('support', {}).get('attack', 0)}</p>
    </div>
"""
        
        plot_descriptions = {
            'confusion_matrix': '🎯 混淆矩阵 - 显示分类准确性，真阳性、假阳性、真阴性、假阴性的分布',
            'risk_distribution': '📊 风险分数分布 - 正常文件 vs 攻击文件的风险分数对比分析',
            'performance_metrics': '📈 性能指标 - 准确率、精确率、召回率、F1分数、ROC AUC等关键指标',
            'roc_pr_curves': '📉 ROC和PR曲线 - 分类器在不同阈值下的性能评估曲线',
            'attack_type_performance': '🎭 攻击类型性能 - 按不同攻击类型分析的检测成功率',
            'language_performance': '🌍 语言性能分析 - 按不同语言分析的检测效果对比',
            'risk_score_scatter': '🔍 风险分数散点图 - 每个文件的风险分数分布可视化',
            'detection_type_stats': '📋 检测类型统计 - 各种检测机制的触发频率统计',
            'detection_count_distribution': '📦 检测数量分布 - 正常文件 vs 攻击文件的检测次数箱线图'
        }
        
        for plot_key, plot_file in plot_files.items():
            if plot_file.exists():
                description = plot_descriptions.get(plot_key, plot_key)
                html_content += f"""
            <div class="plot-container">
                <div class="plot-title">{description}</div>
                <img src="{plot_file.name}" alt="{description}">
            </div>
            """
        
        html_content += """
    <div style="text-align: center; margin-top: 40px; color: #7f8c8d;">
        <p>📧 Generated by Paper Review Attack Detection System</p>
    </div>
</body>
</html>
"""
        
        html_file = plots_dir / "index.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"📋 图表索引已生成: {html_file}")
    
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
