import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc, matthews_corrcoef,
    balanced_accuracy_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import json
from pathlib import Path
import warnings
import time
from datetime import datetime
from collections import defaultdict, Counter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from .utils import setup_logging, ensure_dir, save_results

logger = setup_logging()

class AdvancedExperimentEvaluator:
    """高级实验评估器 - 支持配置文件的全功能版本"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.experiment_config = config['experiment']
        self.output_dir = ensure_dir(self.experiment_config['output_dir'])
        self.results_history = []
        
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 评估器配置
        self.evaluation_config = self.experiment_config.get('performance_monitoring', {
            'enabled': True,
            'save_predictions': True,
            'save_probabilities': True,
            'error_analysis': True,
            'confusion_matrix_details': True
        })
        
        # 可视化配置
        self.viz_config = self.experiment_config.get('visualization', {
            'figsize': [15, 10],
            'dpi': 300,
            'save_format': 'png',
            'color_scheme': 'seaborn',
            'style': 'whitegrid'
        })
        
        # 设置图表样式
        if self.viz_config.get('style'):
            plt.style.use(self.viz_config['style'])
        
        # 交叉验证配置
        self.cv_config = self.experiment_config.get('cross_validation', {
            'enabled': False,
            'cv_folds': 5,
            'shuffle': True,
            'stratify': True
        })
        
        # 阈值优化配置
        self.threshold_config = self.experiment_config.get('threshold_optimization', {
            'enabled': True,
            'optimization_metric': 'f1_score',
            'search_range': [0.1, 0.9],
            'search_steps': 80,
            'cross_validate': True
        })
        
        # 指标配置
        self.metrics_config = self.experiment_config.get('metrics', {
            'primary': ["accuracy", "precision", "recall", "f1_score", "roc_auc"],
            'additional': ["precision_recall_auc", "matthews_corrcoef", "balanced_accuracy"],
            'per_class': True,
            'confidence_intervals': True
        })
        
        logger.info(f"高级实验评估器初始化完成，输出目录: {self.output_dir}")
    
    def evaluate_detection_performance(self, 
                                     clean_files: List[str], 
                                     attack_files: List[str], 
                                     detector,
                                     attack_info: Optional[List[Dict]] = None) -> Tuple[pd.DataFrame, Dict]:
        """评估检测性能 - 全功能版本"""
        
        logger.info(f"开始高级性能评估: {len(clean_files)} 个正常文件, {len(attack_files)} 个攻击文件")
        
        # 数据集平衡处理
        if self.experiment_config.get('dataset_balancing', {}).get('enabled', True):
            clean_files, attack_files = self._balance_dataset(clean_files, attack_files)
        
        all_results = []
        processing_times = []
        
        # 测试正常文件（负样本）
        logger.info("测试正常文件...")
        clean_results = self._evaluate_file_batch(
            clean_files, detector, file_type='clean', label=0
        )
        all_results.extend(clean_results['results'])
        processing_times.extend(clean_results['times'])
        
        # 测试攻击文件（正样本）
        logger.info("测试攻击文件...")
        attack_info_dict = {}
        if attack_info:
            attack_info_dict = {info['attack_file']: info for info in attack_info}
        
        attack_results = self._evaluate_file_batch(
            attack_files, detector, file_type='attack', label=1, 
            attack_info_dict=attack_info_dict
        )
        all_results.extend(attack_results['results'])
        processing_times.extend(attack_results['times'])
        
        # 创建结果DataFrame
        df_results = pd.DataFrame(all_results)
        
        if df_results.empty:
            logger.error("没有有效的检测结果")
            return df_results, {}
        
        # 计算全面的评估指标
        metrics = self._calculate_comprehensive_metrics(df_results)
        
        # 添加性能统计
        metrics['performance_stats'] = {
            'avg_processing_time': np.mean(processing_times),
            'total_processing_time': sum(processing_times),
            'processing_time_std': np.std(processing_times)
        }
        
        # 交叉验证
        if self.cv_config.get('enabled', False):
            cv_results = self._perform_cross_validation(df_results, detector)
            metrics['cross_validation'] = cv_results
        
        # 阈值优化
        if self.threshold_config.get('enabled', True):
            threshold_analysis = self._advanced_threshold_analysis(df_results)
            metrics['threshold_analysis'] = threshold_analysis
        
        # 保存详细结果
        self._save_comprehensive_results(df_results, metrics)
        
        logger.info("高级性能评估完成")
        return df_results, metrics
    
    def _balance_dataset(self, clean_files: List[str], attack_files: List[str]) -> Tuple[List[str], List[str]]:
        """数据集平衡处理"""
        balance_config = self.experiment_config.get('dataset_balancing', {})
        target_ratio = balance_config.get('target_positive_ratio', 0.4)
        method = balance_config.get('balancing_method', 'undersample')
        
        total_files = len(clean_files) + len(attack_files)
        target_attack_count = int(total_files * target_ratio)
        target_clean_count = total_files - target_attack_count
        
        if method == 'undersample':
            # 下采样
            if len(attack_files) > target_attack_count:
                attack_files = np.random.choice(attack_files, target_attack_count, replace=False).tolist()
            if len(clean_files) > target_clean_count:
                clean_files = np.random.choice(clean_files, target_clean_count, replace=False).tolist()
        
        elif method == 'oversample':
            # 上采样（重复采样）
            if len(attack_files) < target_attack_count:
                additional_needed = target_attack_count - len(attack_files)
                additional_files = np.random.choice(attack_files, additional_needed, replace=True).tolist()
                attack_files.extend(additional_files)
            
            if len(clean_files) < target_clean_count:
                additional_needed = target_clean_count - len(clean_files)
                additional_files = np.random.choice(clean_files, additional_needed, replace=True).tolist()
                clean_files.extend(additional_files)
        
        logger.info(f"数据集平衡后: {len(clean_files)} 正常文件, {len(attack_files)} 攻击文件")
        return clean_files, attack_files
    
    def _evaluate_file_batch(self, file_list: List[str], detector, file_type: str, 
                           label: int, attack_info_dict: Optional[Dict] = None) -> Dict:
        """批量评估文件"""
        results = []
        processing_times = []
        
        for i, file_path in enumerate(file_list):
            try:
                start_time = time.time()
                result = detector.detect_injection(file_path)
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                file_result = {
                    'file_path': file_path,
                    'file_name': Path(file_path).name,
                    'label': label,
                    'predicted': 1 if result['is_malicious'] else 0,
                    'risk_score': result['risk_score'],
                    'detection_count': result['detection_count'],
                    'file_type': file_type,
                    'processing_time': processing_time,
                    'detections': result.get('detections', [])
                }
                
                # 添加攻击特定信息
                if file_type == 'attack' and attack_info_dict:
                    attack_details = attack_info_dict.get(file_path, {})
                    file_result.update({
                        'attack_type': attack_details.get('attack_type', 
                                                        self._extract_attack_type_from_filename(file_path)),
                        'language': attack_details.get('language', 
                                                     self._extract_language_from_filename(file_path))
                    })
                else:
                    file_result.update({
                        'attack_type': None,
                        'language': None
                    })
                
                # 增强的检测信息分析
                self._enrich_detection_info(file_result, result)
                
                results.append(file_result)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"已处理{file_type}文件: {i + 1}/{len(file_list)}")
                    
            except Exception as e:
                logger.error(f"处理{file_type}文件失败 {file_path}: {e}")
                processing_times.append(0)  # 失败的文件记录0处理时间
        
        return {'results': results, 'times': processing_times}
    
    def _enrich_detection_info(self, file_result: Dict, detection_result: Dict):
        """增强检测信息分析"""
        detections = detection_result.get('detections', [])
        
        # 检测类型统计
        detection_types = [d['type'] for d in detections]
        file_result['detection_types'] = ', '.join(set(detection_types))
        file_result['unique_detection_types'] = len(set(detection_types))
        
        # 检测类型详细统计
        type_counts = {}
        confidence_scores = {}
        
        for detection in detections:
            det_type = detection['type']
            type_counts[det_type] = type_counts.get(det_type, 0) + 1
            
            if det_type not in confidence_scores:
                confidence_scores[det_type] = []
            confidence_scores[det_type].append(detection.get('confidence', 0.5))
        
        file_result['detection_type_counts'] = type_counts
        file_result['avg_confidence_by_type'] = {
            det_type: np.mean(scores) for det_type, scores in confidence_scores.items()
        }
        
        # 风险评估
        if 'original_risk_score' in detection_result:
            file_result['original_risk_score'] = detection_result['original_risk_score']
            file_result['risk_score_adjustment'] = (
                detection_result['risk_score'] - detection_result['original_risk_score']
            )
        
        # 内容统计
        content_stats = detection_result.get('content_stats', {})
        file_result.update({
            'text_length': content_stats.get('text_length', 0),
            'page_count': content_stats.get('page_count', 0),
            'file_size': content_stats.get('file_size', 0),
            'white_text_count': content_stats.get('white_text_count', 0),
            'small_text_count': content_stats.get('small_text_count', 0),
            'detection_density': (
                detection_result['detection_count'] / 
                (content_stats.get('text_length', 1000) / 1000)
            ) if content_stats.get('text_length', 0) > 0 else 0
        })
    
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
        elif 'contextual' in filename:
            return 'contextual_attack'
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
    
    def _calculate_comprehensive_metrics(self, df_results: pd.DataFrame) -> Dict[str, Any]:
        """计算全面的评估指标"""
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
                'attack': int(np.sum(y_true == 1)),
                'total': len(y_true)
            }
        }
        
        # 额外指标
        if 'matthews_corrcoef' in self.metrics_config.get('additional', []):
            metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        
        if 'balanced_accuracy' in self.metrics_config.get('additional', []):
            metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        
        # AUC指标
        if len(np.unique(y_scores)) > 1:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
                precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
                metrics['pr_auc'] = auc(recall_curve, precision_curve)
                metrics['precision_recall_auc'] = metrics['pr_auc']  # 别名
            except Exception as e:
                logger.warning(f"AUC计算失败: {e}")
                metrics['roc_auc'] = 0.0
                metrics['pr_auc'] = 0.0
                metrics['precision_recall_auc'] = 0.0
        else:
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0
            metrics['precision_recall_auc'] = 0.0
        
        # 混淆矩阵详细分析
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
            
            # 详细的错误率分析
            metrics.update({
                'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
                'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
                'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'true_negative_rate': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'positive_predictive_value': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0
            })
        
        # 置信区间计算
        if self.metrics_config.get('confidence_intervals', True):
            metrics['confidence_intervals'] = self._calculate_confidence_intervals(y_true, y_pred)
        
        # 分类性能分析
        metrics.update({
            'performance_by_attack_type': self._analyze_performance_by_attack_type(df_results),
            'performance_by_language': self._analyze_performance_by_language(df_results),
            'detection_type_analysis': self._analyze_detection_types(df_results),
            'error_analysis': self._analyze_errors(df_results),
            'risk_score_analysis': self._analyze_risk_scores(df_results),
            'processing_performance': self._analyze_processing_performance(df_results)
        })
        
        return metrics
    
    def _calculate_confidence_intervals(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      confidence: float = 0.95) -> Dict:
        """计算指标的置信区间"""
        n_bootstrap = 1000
        metrics_bootstrap = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }
        
        n_samples = len(y_true)
        
        for _ in range(n_bootstrap):
            # Bootstrap采样
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            # 计算指标
            metrics_bootstrap['accuracy'].append(accuracy_score(y_true_boot, y_pred_boot))
            metrics_bootstrap['precision'].append(precision_score(y_true_boot, y_pred_boot, zero_division=0))
            metrics_bootstrap['recall'].append(recall_score(y_true_boot, y_pred_boot, zero_division=0))
            metrics_bootstrap['f1_score'].append(f1_score(y_true_boot, y_pred_boot, zero_division=0))
        
        # 计算置信区间
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        confidence_intervals = {}
        for metric, values in metrics_bootstrap.items():
            confidence_intervals[metric] = {
                'lower': np.percentile(values, lower_percentile),
                'upper': np.percentile(values, upper_percentile),
                'mean': np.mean(values),
                'std': np.std(values)
            }
        
        return confidence_intervals
    
    def _perform_cross_validation(self, df_results: pd.DataFrame, detector) -> Dict:
        """执行交叉验证"""
        logger.info("执行交叉验证...")
        
        # 准备数据
        unique_files = df_results[['file_path', 'label']].drop_duplicates()
        X = unique_files['file_path'].values
        y = unique_files['label'].values
        
        cv_folds = self.cv_config.get('cv_folds', 5)
        shuffle = self.cv_config.get('shuffle', True)
        stratify = self.cv_config.get('stratify', True)
        
        if stratify:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=shuffle, random_state=42)
        else:
            from sklearn.model_selection import KFold
            cv = KFold(n_splits=cv_folds, shuffle=shuffle, random_state=42)
        
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            logger.info(f"交叉验证 Fold {fold + 1}/{cv_folds}")
            
            test_files = X[test_idx]
            test_labels = y[test_idx]
            
            # 对测试文件进行预测
            predictions = []
            for file_path in test_files:
                try:
                    result = detector.detect_injection(file_path)
                    predictions.append(1 if result['is_malicious'] else 0)
                except Exception:
                    predictions.append(0)
            
            # 计算指标
            predictions = np.array(predictions)
            cv_scores['accuracy'].append(accuracy_score(test_labels, predictions))
            cv_scores['precision'].append(precision_score(test_labels, predictions, zero_division=0))
            cv_scores['recall'].append(recall_score(test_labels, predictions, zero_division=0))
            cv_scores['f1_score'].append(f1_score(test_labels, predictions, zero_division=0))
        
        # 计算统计量
        cv_results = {}
        for metric, scores in cv_scores.items():
            cv_results[metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores
            }
        
        return cv_results
    
    def _advanced_threshold_analysis(self, df_results: pd.DataFrame) -> Dict:
        """高级阈值分析"""
        y_true = df_results['label'].values
        y_scores = df_results['risk_score'].values
        
        threshold_analysis = {}
        
        try:
            # 获取配置
            search_range = self.threshold_config.get('search_range', [0.1, 0.9])
            search_steps = self.threshold_config.get('search_steps', 80)
            optimization_metric = self.threshold_config.get('optimization_metric', 'f1_score')
            
            # 生成阈值搜索空间
            thresholds = np.linspace(search_range[0], search_range[1], search_steps)
            threshold_performance = []
            
            for threshold in thresholds:
                y_pred = (y_scores >= threshold).astype(int)
                
                if len(np.unique(y_pred)) > 1:
                    precision = precision_score(y_true, y_pred, zero_division=0)
                    recall = recall_score(y_true, y_pred, zero_division=0)
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    accuracy = accuracy_score(y_true, y_pred)
                    
                    # 计算其他指标
                    cm = confusion_matrix(y_true, y_pred)
                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
                    else:
                        fpr = fnr = 0
                    
                    threshold_performance.append({
                        'threshold': threshold,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'accuracy': accuracy,
                        'false_positive_rate': fpr,
                        'false_negative_rate': fnr
                    })
            
            threshold_analysis['performance_curve'] = threshold_performance
            
            # 找到最佳阈值
            if threshold_performance:
                # 根据优化指标找到最佳阈值
                best_idx = max(range(len(threshold_performance)), 
                             key=lambda i: threshold_performance[i][optimization_metric])
                
                threshold_analysis['optimal_thresholds'] = {
                    f'{optimization_metric}_optimal': threshold_performance[best_idx],
                    'precision_recall_curve': self._find_precision_recall_optimal_threshold(y_true, y_scores)
                }
                
                # 多目标优化阈值
                threshold_analysis['pareto_optimal'] = self._find_pareto_optimal_thresholds(threshold_performance)
        
        except Exception as e:
            logger.warning(f"高级阈值分析失败: {e}")
            threshold_analysis = {'error': str(e)}
        
        return threshold_analysis
    
    def _find_precision_recall_optimal_threshold(self, y_true, y_scores):
        """使用PR曲线找到最佳阈值"""
        try:
            precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            best_idx = np.argmax(f1_scores)
            
            return {
                'threshold': thresholds[best_idx] if best_idx < len(thresholds) else 0.5,
                'precision': precision[best_idx],
                'recall': recall[best_idx],
                'f1_score': f1_scores[best_idx]
            }
        except Exception:
            return None
    
    def _find_pareto_optimal_thresholds(self, threshold_performance: List[Dict]) -> List[Dict]:
        """找到帕累托最优阈值"""
        pareto_optimal = []
        
        for i, perf_i in enumerate(threshold_performance):
            is_dominated = False
            
            for j, perf_j in enumerate(threshold_performance):
                if i != j:
                    # 检查是否被支配 (precision, recall 都更优)
                    if (perf_j['precision'] >= perf_i['precision'] and 
                        perf_j['recall'] >= perf_i['recall'] and
                        (perf_j['precision'] > perf_i['precision'] or 
                         perf_j['recall'] > perf_i['recall'])):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_optimal.append(perf_i)
        
        return sorted(pareto_optimal, key=lambda x: x['f1_score'], reverse=True)[:5]
    
    def _analyze_performance_by_attack_type(self, df_results: pd.DataFrame) -> Dict:
        """按攻击类型分析性能"""
        performance = {}
        
        for attack_type in df_results['attack_type'].dropna().unique():
            if attack_type and attack_type != 'unknown':
                mask = df_results['attack_type'] == attack_type
                attack_data = df_results[mask]
                
                if len(attack_data) > 0:
                    y_true = attack_data['label'].values
                    y_pred = attack_data['predicted'].values
                    
                    performance[attack_type] = {
                        'count': len(attack_data),
                        'detection_rate': attack_data['predicted'].mean(),
                        'avg_risk_score': attack_data['risk_score'].mean(),
                        'avg_detection_count': attack_data['detection_count'].mean(),
                        'avg_processing_time': attack_data['processing_time'].mean(),
                        'detection_types': self._get_common_detection_types(attack_data),
                        'precision': precision_score(y_true, y_pred, zero_division=0),
                        'recall': recall_score(y_true, y_pred, zero_division=0),
                        'f1_score': f1_score(y_true, y_pred, zero_division=0)
                    }
        
        return performance
    
    def _analyze_performance_by_language(self, df_results: pd.DataFrame) -> Dict:
        """按语言分析性能"""
        performance = {}
        
        for language in df_results['language'].dropna().unique():
            if language and language != 'unknown':
                mask = df_results['language'] == language
                lang_data = df_results[mask]
                
                if len(lang_data) > 0:
                    y_true = lang_data['label'].values
                    y_pred = lang_data['predicted'].values
                    
                    performance[language] = {
                        'count': len(lang_data),
                        'detection_rate': lang_data['predicted'].mean(),
                        'avg_risk_score': lang_data['risk_score'].mean(),
                        'avg_detection_count': lang_data['detection_count'].mean(),
                        'false_negative_rate': (
                            len(lang_data[(lang_data['label'] == 1) & (lang_data['predicted'] == 0)]) /
                            len(lang_data[lang_data['label'] == 1])
                        ) if len(lang_data[lang_data['label'] == 1]) > 0 else 0,
                        'precision': precision_score(y_true, y_pred, zero_division=0),
                        'recall': recall_score(y_true, y_pred, zero_division=0),
                        'f1_score': f1_score(y_true, y_pred, zero_division=0)
                    }
        
        return performance
    
    def _analyze_detection_types(self, df_results: pd.DataFrame) -> Dict:
        """分析检测类型效果"""
        type_analysis = {}
        
        # 统计所有检测类型
        all_detections = []
        for _, result in df_results.iterrows():
            all_detections.extend(result.get('detections', []))
        
        # 按类型统计
        type_counts = defaultdict(int)
        type_confidence = defaultdict(list)
        type_in_malicious = defaultdict(int)
        type_in_clean = defaultdict(int)
        
        for detection in all_detections:
            det_type = detection.get('type', 'unknown')
            confidence = detection.get('confidence', 0.5)
            
            type_counts[det_type] += 1
            type_confidence[det_type].append(confidence)
        
        # 按正负样本统计检测类型分布
        malicious_files = df_results[df_results['label'] == 1]
        clean_files = df_results[df_results['label'] == 0]
        
        for det_type in type_counts.keys():
            malicious_with_type = sum(1 for _, row in malicious_files.iterrows() 
                                    if det_type in row.get('detection_types', ''))
            clean_with_type = sum(1 for _, row in clean_files.iterrows() 
                                if det_type in row.get('detection_types', ''))
            
            type_in_malicious[det_type] = malicious_with_type
            type_in_clean[det_type] = clean_with_type
        
        # 计算每种检测类型的效果
        for det_type in type_counts.keys():
            total_malicious = len(malicious_files)
            total_clean = len(clean_files)
            
            malicious_count = type_in_malicious[det_type]
            clean_count = type_in_clean[det_type]
            
            type_analysis[det_type] = {
                'total_count': type_counts[det_type],
                'avg_confidence': np.mean(type_confidence[det_type]),
                'confidence_std': np.std(type_confidence[det_type]),
                'in_malicious': malicious_count,
                'in_clean': clean_count,
                'malicious_detection_rate': malicious_count / total_malicious if total_malicious > 0 else 0,
                'clean_false_positive_rate': clean_count / total_clean if total_clean > 0 else 0,
                'precision': malicious_count / (malicious_count + clean_count) if (malicious_count + clean_count) > 0 else 0,
                'effectiveness_score': (malicious_count / total_malicious - clean_count / total_clean) if total_malicious > 0 and total_clean > 0 else 0
            }
        
        return dict(type_analysis)
    
    def _get_common_detection_types(self, attack_data: pd.DataFrame) -> List[str]:
        """获取攻击数据中最常见的检测类型"""
        all_types = []
        for _, row in attack_data.iterrows():
            types = row.get('detection_types', '').split(', ')
            all_types.extend([t.strip() for t in types if t.strip()])
        
        type_counts = Counter(all_types)
        return [t for t, _ in type_counts.most_common(5)]
    
    def _analyze_errors(self, df_results: pd.DataFrame) -> Dict:
        """详细的错误分析"""
        error_analysis = {}
        
        # 误报分析
        false_positives = df_results[(df_results['label'] == 0) & (df_results['predicted'] == 1)]
        error_analysis['false_positives'] = {
            'count': len(false_positives),
            'percentage': len(false_positives) / len(df_results[df_results['label'] == 0]) * 100 if len(df_results[df_results['label'] == 0]) > 0 else 0,
            'common_detection_types': self._get_common_detection_types(false_positives),
            'avg_risk_score': false_positives['risk_score'].mean() if len(false_positives) > 0 else 0,
            'avg_detection_count': false_positives['detection_count'].mean() if len(false_positives) > 0 else 0,
            'risk_score_distribution': {
                'min': false_positives['risk_score'].min() if len(false_positives) > 0 else 0,
                'max': false_positives['risk_score'].max() if len(false_positives) > 0 else 0,
                'std': false_positives['risk_score'].std() if len(false_positives) > 0 else 0
            }
        }
        
        # 漏报分析
        false_negatives = df_results[(df_results['label'] == 1) & (df_results['predicted'] == 0)]
        error_analysis['false_negatives'] = {
            'count': len(false_negatives),
            'percentage': len(false_negatives) / len(df_results[df_results['label'] == 1]) * 100 if len(df_results[df_results['label'] == 1]) > 0 else 0,
            'by_attack_type': {},
            'by_language': {},
            'avg_risk_score': false_negatives['risk_score'].mean() if len(false_negatives) > 0 else 0,
            'avg_detection_count': false_negatives['detection_count'].mean() if len(false_negatives) > 0 else 0,
            'risk_score_distribution': {
                'min': false_negatives['risk_score'].min() if len(false_negatives) > 0 else 0,
                'max': false_negatives['risk_score'].max() if len(false_negatives) > 0 else 0,
                'std': false_negatives['risk_score'].std() if len(false_negatives) > 0 else 0
            }
        }
        
        # 按攻击类型分析漏报
        for attack_type in false_negatives['attack_type'].dropna().unique():
            if attack_type != 'unknown':
                fn_by_type = false_negatives[false_negatives['attack_type'] == attack_type]
                total_by_type = df_results[(df_results['attack_type'] == attack_type) & (df_results['label'] == 1)]
                
                error_analysis['false_negatives']['by_attack_type'][attack_type] = {
                    'count': len(fn_by_type),
                    'rate': len(fn_by_type) / len(total_by_type) if len(total_by_type) > 0 else 0
                }
        
        # 按语言分析漏报
        for language in false_negatives['language'].dropna().unique():
            if language != 'unknown':
                fn_by_lang = false_negatives[false_negatives['language'] == language]
                total_by_lang = df_results[(df_results['language'] == language) & (df_results['label'] == 1)]
                
                error_analysis['false_negatives']['by_language'][language] = {
                    'count': len(fn_by_lang),
                    'rate': len(fn_by_lang) / len(total_by_lang) if len(total_by_lang) > 0 else 0
                }
        
        return error_analysis
    
    def _analyze_risk_scores(self, df_results: pd.DataFrame) -> Dict:
        """深度风险分数分析"""
        clean_scores = df_results[df_results['label'] == 0]['risk_score']
        attack_scores = df_results[df_results['label'] == 1]['risk_score']
        
        analysis = {
            'clean_files': self._calculate_score_statistics(clean_scores),
            'attack_files': self._calculate_score_statistics(attack_scores)
        }
        
        # 分离度分析
        if len(clean_scores) > 0 and len(attack_scores) > 0:
            # Cohen's d (效应量)
            pooled_std = np.sqrt(((len(clean_scores) - 1) * clean_scores.var() + 
                                (len(attack_scores) - 1) * attack_scores.var()) / 
                               (len(clean_scores) + len(attack_scores) - 2))
            
            analysis['separation'] = {
                'mean_difference': attack_scores.mean() - clean_scores.mean(),
                'cohens_d': (attack_scores.mean() - clean_scores.mean()) / pooled_std if pooled_std > 0 else 0,
                'overlap_ratio': self._calculate_overlap_ratio(clean_scores, attack_scores),
                'separability_index': self._calculate_separability_index(clean_scores, attack_scores)
            }
        
        # 阈值敏感性分析
        analysis['threshold_sensitivity'] = self._analyze_threshold_sensitivity(clean_scores, attack_scores)
        
        return analysis
    
    def _calculate_score_statistics(self, scores: pd.Series) -> Dict:
        """计算分数统计信息"""
        if len(scores) == 0:
            return {'count': 0}
        
        return {
            'count': len(scores),
            'mean': scores.mean(),
            'std': scores.std(),
            'min': scores.min(),
            'max': scores.max(),
            'median': scores.median(),
            'q25': scores.quantile(0.25),
            'q75': scores.quantile(0.75),
            'skewness': scores.skew(),
            'kurtosis': scores.kurtosis()
        }
    
    def _calculate_overlap_ratio(self, clean_scores: pd.Series, attack_scores: pd.Series) -> float:
        """计算分数分布的重叠比例"""
        try:
            min_attack = attack_scores.min()
            max_clean = clean_scores.max()
            
            if min_attack >= max_clean:
                return 0.0
            
            overlap_range = max_clean - min_attack
            total_range = max(attack_scores.max(), clean_scores.max()) - min(attack_scores.min(), clean_scores.min())
            
            return overlap_range / total_range if total_range > 0 else 1.0
        except Exception:
            return 1.0
    
    def _calculate_separability_index(self, clean_scores: pd.Series, attack_scores: pd.Series) -> float:
        """计算可分离性指数"""
        try:
            # 基于KL散度的可分离性
            from scipy.stats import entropy
            
            # 创建直方图
            bins = np.linspace(0, 1, 50)
            clean_hist, _ = np.histogram(clean_scores, bins=bins, density=True)
            attack_hist, _ = np.histogram(attack_scores, bins=bins, density=True)
            
            # 添加小的平滑项避免0
            clean_hist = clean_hist + 1e-10
            attack_hist = attack_hist + 1e-10
            
            # 归一化
            clean_hist = clean_hist / clean_hist.sum()
            attack_hist = attack_hist / attack_hist.sum()
            
            # 计算KL散度
            kl_div = entropy(attack_hist, clean_hist)
            
            return min(kl_div / 10, 1.0)  # 归一化到0-1
        except Exception:
            return 0.0
    
    def _analyze_threshold_sensitivity(self, clean_scores: pd.Series, attack_scores: pd.Series) -> Dict:
        """分析阈值敏感性"""
        sensitivity_analysis = {}
        
        # 计算不同阈值下的性能变化
        thresholds = np.arange(0.1, 0.9, 0.1)
        sensitivity_data = []
        
        for threshold in thresholds:
            clean_above = (clean_scores >= threshold).sum()
            attack_above = (attack_scores >= threshold).sum()
            
            fpr = clean_above / len(clean_scores) if len(clean_scores) > 0 else 0
            tpr = attack_above / len(attack_scores) if len(attack_scores) > 0 else 0
            
            sensitivity_data.append({
                'threshold': threshold,
                'false_positive_rate': fpr,
                'true_positive_rate': tpr,
                'clean_above_threshold': int(clean_above),
                'attack_above_threshold': int(attack_above)
            })
        
        sensitivity_analysis['threshold_performance'] = sensitivity_data
        
        return sensitivity_analysis
    
    def _analyze_processing_performance(self, df_results: pd.DataFrame) -> Dict:
        """分析处理性能"""
        processing_times = df_results['processing_time']
        
        performance_analysis = {
            'overall': self._calculate_score_statistics(processing_times),
            'by_file_type': {},
            'by_attack_type': {}
        }
        
        # 按文件类型分析
        for file_type in df_results['file_type'].unique():
            type_data = df_results[df_results['file_type'] == file_type]
            performance_analysis['by_file_type'][file_type] = self._calculate_score_statistics(type_data['processing_time'])
        
        # 按攻击类型分析
        for attack_type in df_results['attack_type'].dropna().unique():
            if attack_type != 'unknown':
                attack_data = df_results[df_results['attack_type'] == attack_type]
                performance_analysis['by_attack_type'][attack_type] = self._calculate_score_statistics(attack_data['processing_time'])
        
        return performance_analysis
    
    def _save_comprehensive_results(self, df_results: pd.DataFrame, metrics: Dict):
        """保存全面的结果"""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存主要结果
        csv_file = Path(self.output_dir) / f"comprehensive_results_{timestamp}.csv"
        df_results.to_csv(csv_file, index=False, encoding='utf-8')
        
        # 保存JSON指标
        json_file = Path(self.output_dir) / f"comprehensive_metrics_{timestamp}.json"
        save_results(metrics, str(json_file))
        
        # 保存预测结果（如果启用）
        if self.evaluation_config.get('save_predictions', True):
            predictions_file = Path(self.output_dir) / f"predictions_{timestamp}.csv"
            predictions_df = df_results[['file_path', 'label', 'predicted', 'risk_score']].copy()
            predictions_df.to_csv(predictions_file, index=False, encoding='utf-8')
        
        # 保存概率分数（如果启用）
        if self.evaluation_config.get('save_probabilities', True):
            probabilities_file = Path(self.output_dir) / f"probabilities_{timestamp}.csv"
            prob_df = df_results[['file_path', 'risk_score']].copy()
            prob_df.to_csv(probabilities_file, index=False, encoding='utf-8')
        
        # 保存错误分析
        if self.evaluation_config.get('error_analysis', True) and 'error_analysis' in metrics:
            self._save_error_analysis(df_results, metrics['error_analysis'], timestamp)
        
        logger.info(f"全面结果已保存: {csv_file}, {json_file}")
    
    def _save_error_analysis(self, df_results: pd.DataFrame, error_analysis: Dict, timestamp: str):
        """保存错误分析"""
        # 保存误报文件列表
        false_positives = df_results[(df_results['label'] == 0) & (df_results['predicted'] == 1)]
        if len(false_positives) > 0:
            fp_file = Path(self.output_dir) / f"false_positives_detailed_{timestamp}.csv"
            false_positives.to_csv(fp_file, index=False, encoding='utf-8')
            logger.info(f"详细误报分析已保存: {fp_file}")
        
        # 保存漏报文件列表
        false_negatives = df_results[(df_results['label'] == 1) & (df_results['predicted'] == 0)]
        if len(false_negatives) > 0:
            fn_file = Path(self.output_dir) / f"false_negatives_detailed_{timestamp}.csv"
            false_negatives.to_csv(fn_file, index=False, encoding='utf-8')
            logger.info(f"详细漏报分析已保存: {fn_file}")
    
    def generate_interactive_visualizations(self, df_results: pd.DataFrame, 
                                          metrics: Dict, save_plots: bool = True) -> Dict[str, Any]:
        """生成交互式可视化"""
        if not save_plots:
            return {}
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        plots_dir = Path(self.output_dir) / f"interactive_plots_{timestamp}"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        plot_files = {}
        
        try:
            # 1. 交互式风险分数分布
            logger.info("生成交互式风险分数分布...")
            fig_risk = self._create_interactive_risk_distribution(df_results, metrics)
            risk_file = plots_dir / "interactive_risk_distribution.html"
            fig_risk.write_html(str(risk_file))
            plot_files['interactive_risk_distribution'] = risk_file
            
            # 2. 交互式性能矩阵
            logger.info("生成交互式性能矩阵...")
            fig_perf = self._create_interactive_performance_matrix(metrics)
            perf_file = plots_dir / "interactive_performance_matrix.html"
            fig_perf.write_html(str(perf_file))
            plot_files['interactive_performance_matrix'] = perf_file
            
            # 3. 交互式阈值分析
            if 'threshold_analysis' in metrics:
                logger.info("生成交互式阈值分析...")
                fig_thresh = self._create_interactive_threshold_analysis(metrics['threshold_analysis'])
                thresh_file = plots_dir / "interactive_threshold_analysis.html"
                fig_thresh.write_html(str(thresh_file))
                plot_files['interactive_threshold_analysis'] = thresh_file
            
            # 4. 交互式检测类型分析
            if 'detection_type_analysis' in metrics:
                logger.info("生成交互式检测类型分析...")
                fig_det = self._create_interactive_detection_analysis(metrics['detection_type_analysis'])
                det_file = plots_dir / "interactive_detection_analysis.html"
                fig_det.write_html(str(det_file))
                plot_files['interactive_detection_analysis'] = det_file
            
        except Exception as e:
            logger.error(f"交互式可视化生成失败: {e}")
        
        logger.info(f"交互式可视化已保存到: {plots_dir}")
        return {'plots_directory': plots_dir, 'plot_files': plot_files}
    
    def _create_interactive_risk_distribution(self, df_results: pd.DataFrame, metrics: Dict):
        """创建交互式风险分数分布图"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Risk Score Distribution', 'Box Plot Comparison', 
                          'Scatter Plot by File Type', 'Cumulative Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        clean_scores = df_results[df_results['label'] == 0]['risk_score']
        attack_scores = df_results[df_results['label'] == 1]['risk_score']
        
        # 1. 直方图
        fig.add_trace(
            go.Histogram(x=clean_scores, name='Normal Files', opacity=0.7, 
                        nbinsx=30, marker_color='skyblue'),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(x=attack_scores, name='Attack Files', opacity=0.7, 
                        nbinsx=30, marker_color='lightcoral'),
            row=1, col=1
        )
        
        # 2. 箱线图
        fig.add_trace(
            go.Box(y=clean_scores, name='Normal Files', marker_color='skyblue'),
            row=1, col=2
        )
        fig.add_trace(
            go.Box(y=attack_scores, name='Attack Files', marker_color='lightcoral'),
            row=1, col=2
        )
        
        # 3. 散点图
        fig.add_trace(
            go.Scatter(x=list(range(len(clean_scores))), y=clean_scores, 
                      mode='markers', name='Normal Files', 
                      marker=dict(color='skyblue', size=4)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=list(range(len(clean_scores), len(clean_scores) + len(attack_scores))), 
                      y=attack_scores, mode='markers', name='Attack Files',
                      marker=dict(color='lightcoral', size=4)),
            row=2, col=1
        )
        
        # 4. 累积分布
        clean_sorted = np.sort(clean_scores)
        attack_sorted = np.sort(attack_scores)
        
        fig.add_trace(
            go.Scatter(x=clean_sorted, y=np.arange(1, len(clean_sorted) + 1) / len(clean_sorted),
                      mode='lines', name='Normal Files CDF', line=dict(color='blue')),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=attack_sorted, y=np.arange(1, len(attack_sorted) + 1) / len(attack_sorted),
                      mode='lines', name='Attack Files CDF', line=dict(color='red')),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Interactive Risk Score Analysis")
        return fig
    
    def _create_interactive_performance_matrix(self, metrics: Dict):
        """创建交互式性能矩阵"""
        # 准备数据
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
        metric_values = [
            metrics.get('accuracy', 0),
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1_score', 0),
            metrics.get('roc_auc', 0)
        ]
        
        # 创建雷达图
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=metric_values,
            theta=metric_names,
            fill='toself',
            name='Performance Metrics',
            line_color='rgb(0,100,200)',
            fillcolor='rgba(0,100,200,0.25)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Performance Radar Chart"
        )
        
        return fig
    
    def _create_interactive_threshold_analysis(self, threshold_analysis: Dict):
        """创建交互式阈值分析"""
        if 'performance_curve' not in threshold_analysis:
            return go.Figure()
        
        data = threshold_analysis['performance_curve']
        thresholds = [d['threshold'] for d in data]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Precision vs Recall', 'F1-Score vs Threshold', 
                          'Accuracy vs Threshold', 'Error Rates vs Threshold')
        )
        
        # 1. Precision vs Recall
        fig.add_trace(
            go.Scatter(x=[d['recall'] for d in data], y=[d['precision'] for d in data],
                      mode='lines+markers', name='Precision-Recall Curve',
                      text=[f"Threshold: {d['threshold']:.3f}" for d in data]),
            row=1, col=1
        )
        
        # 2. F1-Score vs Threshold
        fig.add_trace(
            go.Scatter(x=thresholds, y=[d['f1_score'] for d in data],
                      mode='lines+markers', name='F1-Score', line_color='green'),
            row=1, col=2
        )
        
        # 3. Accuracy vs Threshold
        fig.add_trace(
            go.Scatter(x=thresholds, y=[d['accuracy'] for d in data],
                      mode='lines+markers', name='Accuracy', line_color='blue'),
            row=2, col=1
        )
        
        # 4. Error Rates
        fig.add_trace(
            go.Scatter(x=thresholds, y=[d.get('false_positive_rate', 0) for d in data],
                      mode='lines+markers', name='False Positive Rate', line_color='red'),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=thresholds, y=[d.get('false_negative_rate', 0) for d in data],
                      mode='lines+markers', name='False Negative Rate', line_color='orange'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Interactive Threshold Analysis")
        return fig
    
    def _create_interactive_detection_analysis(self, detection_analysis: Dict):
        """创建交互式检测类型分析"""
        types = list(detection_analysis.keys())
        precisions = [detection_analysis[t]['precision'] for t in types]
        malicious_rates = [detection_analysis[t]['malicious_detection_rate'] for t in types]
        fp_rates = [detection_analysis[t]['clean_false_positive_rate'] for t in types]
        counts = [detection_analysis[t]['total_count'] for t in types]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Detection Type Precision', 'Malicious Detection Rate',
                          'False Positive Rate', 'Detection Count vs Effectiveness')
        )
        
        # 1. Precision
        fig.add_trace(
            go.Bar(x=types, y=precisions, name='Precision', marker_color='lightblue'),
            row=1, col=1
        )
        
        # 2. Malicious Detection Rate
        fig.add_trace(
            go.Bar(x=types, y=malicious_rates, name='Detection Rate', marker_color='lightcoral'),
            row=1, col=2
        )
        
        # 3. False Positive Rate
        fig.add_trace(
            go.Bar(x=types, y=fp_rates, name='FP Rate', marker_color='orange'),
            row=2, col=1
        )
        
        # 4. Scatter: Count vs Effectiveness
        effectiveness = [detection_analysis[t].get('effectiveness_score', 0) for t in types]
        fig.add_trace(
            go.Scatter(x=counts, y=effectiveness, mode='markers+text',
                      text=types, textposition="top center",
                      marker=dict(size=10, color='purple'),
                      name='Count vs Effectiveness'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Interactive Detection Type Analysis")
        return fig
    
    # 向后兼容的方法
    def generate_report(self, df_results: pd.DataFrame, metrics: Dict) -> str:
        """生成详细报告 - 兼容性方法"""
        return self.generate_enhanced_report(df_results, metrics)
    
    def generate_enhanced_report(self, df_results: pd.DataFrame, metrics: Dict) -> str:
        """生成增强的详细报告"""
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
# 📊 Advanced Paper Review Attack Detection Report

🕒 **Generated**: {timestamp}

## 🎯 Executive Summary

本报告提供了论文审稿攻击检测系统的全面性能分析，采用多维度评估方法，包括准确性、效率、鲁棒性和可扩展性等关键指标。

### 🏆 Key Performance Indicators
- **Overall Accuracy**: {metrics.get('accuracy', 0):.3f}
- **Detection Precision**: {metrics.get('precision', 0):.3f}  
- **Detection Recall**: {metrics.get('recall', 0):.3f}
- **F1-Score**: {metrics.get('f1_score', 0):.3f}
- **ROC AUC**: {metrics.get('roc_auc', 0):.3f}
- **PR AUC**: {metrics.get('pr_auc', 0):.3f}

### 📈 Additional Metrics
- **Balanced Accuracy**: {metrics.get('balanced_accuracy', 0):.3f}
- **Matthews Correlation Coefficient**: {metrics.get('matthews_corrcoef', 0):.3f}
- **Processing Time (avg)**: {metrics.get('performance_stats', {}).get('avg_processing_time', 0):.3f}s

## 📈 Dataset Overview

### 📁 Data Composition
| Category | Count | Percentage |
|----------|--------|------------|
| Normal Files | {metrics.get('support', {}).get('clean', 0)} | {metrics.get('support', {}).get('clean', 0)/metrics.get('support', {}).get('total', 1)*100:.1f}% |
| Attack Files | {metrics.get('support', {}).get('attack', 0)} | {metrics.get('support', {}).get('attack', 0)/metrics.get('support', {}).get('total', 1)*100:.1f}% |
| **Total** | **{metrics.get('support', {}).get('total', 0)}** | **100.0%** |

## 🎭 Advanced Detection Performance Analysis

### 🎯 Confusion Matrix Analysis
"""
        
        if 'confusion_matrix_details' in metrics:
            cm_details = metrics['confusion_matrix_details']
            report += f"""
| Metric | Value | Rate | Description |
|--------|--------|------|-------------|
| True Negatives | {cm_details['true_negative']} | {metrics.get('true_negative_rate', 0):.3f} | Correctly identified normal files |
| False Positives | {cm_details['false_positive']} | {metrics.get('false_positive_rate', 0):.3f} | Normal files flagged as attacks |
| False Negatives | {cm_details['false_negative']} | {metrics.get('false_negative_rate', 0):.3f} | Missed attack files |
| True Positives | {cm_details['true_positive']} | {metrics.get('true_positive_rate', 0):.3f} | Correctly detected attacks |

### 📊 Extended Error Analysis
- **Specificity (TNR)**: {metrics.get('specificity', 0):.3f}
- **Sensitivity (TPR)**: {metrics.get('sensitivity', 0):.3f}
- **Positive Predictive Value**: {metrics.get('positive_predictive_value', 0):.3f}
- **Negative Predictive Value**: {metrics.get('negative_predictive_value', 0):.3f}
"""
        
        # 置信区间
        if 'confidence_intervals' in metrics:
            ci = metrics['confidence_intervals']
            report += f"""
### 🎯 Confidence Intervals (95%)
| Metric | Lower Bound | Upper Bound | Mean | Std |
|--------|-------------|-------------|------|-----|
| Accuracy | {ci.get('accuracy', {}).get('lower', 0):.3f} | {ci.get('accuracy', {}).get('upper', 0):.3f} | {ci.get('accuracy', {}).get('mean', 0):.3f} | {ci.get('accuracy', {}).get('std', 0):.3f} |
| Precision | {ci.get('precision', {}).get('lower', 0):.3f} | {ci.get('precision', {}).get('upper', 0):.3f} | {ci.get('precision', {}).get('mean', 0):.3f} | {ci.get('precision', {}).get('std', 0):.3f} |
| Recall | {ci.get('recall', {}).get('lower', 0):.3f} | {ci.get('recall', {}).get('upper', 0):.3f} | {ci.get('recall', {}).get('mean', 0):.3f} | {ci.get('recall', {}).get('std', 0):.3f} |
| F1-Score | {ci.get('f1_score', {}).get('lower', 0):.3f} | {ci.get('f1_score', {}).get('upper', 0):.3f} | {ci.get('f1_score', {}).get('mean', 0):.3f} | {ci.get('f1_score', {}).get('std', 0):.3f} |
"""
        
        # 交叉验证结果
        if 'cross_validation' in metrics:
            cv = metrics['cross_validation']
            report += f"""
### 🔄 Cross-Validation Results ({self.cv_config.get('cv_folds', 5)}-Fold)
| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Accuracy | {cv.get('accuracy', {}).get('mean', 0):.3f} | {cv.get('accuracy', {}).get('std', 0):.3f} | {min(cv.get('accuracy', {}).get('scores', [0])):.3f} | {max(cv.get('accuracy', {}).get('scores', [0])):.3f} |
| Precision | {cv.get('precision', {}).get('mean', 0):.3f} | {cv.get('precision', {}).get('std', 0):.3f} | {min(cv.get('precision', {}).get('scores', [0])):.3f} | {max(cv.get('precision', {}).get('scores', [0])):.3f} |
| Recall | {cv.get('recall', {}).get('mean', 0):.3f} | {cv.get('recall', {}).get('std', 0):.3f} | {min(cv.get('recall', {}).get('scores', [0])):.3f} | {max(cv.get('recall', {}).get('scores', [0])):.3f} |
| F1-Score | {cv.get('f1_score', {}).get('mean', 0):.3f} | {cv.get('f1_score', {}).get('std', 0):.3f} | {min(cv.get('f1_score', {}).get('scores', [0])):.3f} | {max(cv.get('f1_score', {}).get('scores', [0])):.3f} |
"""
        
        # 继续添加其他报告部分...
        # (这里可以继续添加攻击类型分析、语言分析、阈值优化等部分)
        
        # 保存报告
        timestamp_file = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        report_file = Path(self.output_dir) / f"advanced_experiment_report_{timestamp_file}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"高级实验报告已保存: {report_file}")
        return report

# 向后兼容的别名
ExperimentEvaluator = AdvancedExperimentEvaluator
