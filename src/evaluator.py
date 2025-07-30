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
    """实验评估器 - 改进版"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.experiment_config = config['experiment']
        self.output_dir = ensure_dir(self.experiment_config['output_dir'])
        self.results_history = []
        
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 评估器配置
        self.evaluation_config = config.get('evaluation', {
            'enable_detailed_analysis': True,
            'save_plots': True,
            'generate_html_report': True,
            'threshold_analysis': True
        })
        
        logger.info(f"实验评估器初始化完成，输出目录: {self.output_dir}")
    
    def evaluate_detection_performance(self, 
                                     clean_files: List[str], 
                                     attack_files: List[str], 
                                     detector,
                                     attack_info: Optional[List[Dict]] = None) -> Tuple[pd.DataFrame, Dict]:
        """评估检测性能 - 改进版"""
        
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
                    'detections': result.get('detections', [])
                }
                
                # 增强的检测信息分析
                self._enrich_detection_info(file_result, result)
                
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
                
                # 增强的检测信息分析
                self._enrich_detection_info(file_result, result)
                
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
        metrics = self._calculate_enhanced_metrics(df_results)
        
        # 阈值分析
        if self.evaluation_config.get('threshold_analysis', True):
            threshold_analysis = self._analyze_thresholds(df_results)
            metrics['threshold_analysis'] = threshold_analysis
        
        # 保存详细结果
        self._save_detailed_results(df_results, metrics)
        
        logger.info("性能评估完成")
        return df_results, metrics
    
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
    
    def _calculate_enhanced_metrics(self, df_results: pd.DataFrame) -> Dict[str, Any]:
        """计算增强的评估指标"""
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
        
        # ROC AUC（如果有概率分数）
        if len(np.unique(y_scores)) > 1:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
                # PR AUC
                precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
                metrics['pr_auc'] = auc(recall_curve, precision_curve)
            except Exception as e:
                logger.warning(f"AUC计算失败: {e}")
                metrics['roc_auc'] = 0.0
                metrics['pr_auc'] = 0.0
        else:
            metrics['roc_auc'] = 0.0
            metrics['pr_auc'] = 0.0
        
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
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # 按攻击类型的性能
        metrics['performance_by_attack_type'] = self._analyze_performance_by_attack_type(df_results)
        
        # 按语言的性能
        metrics['performance_by_language'] = self._analyze_performance_by_language(df_results)
        
        # 按检测类型的性能
        metrics['detection_type_analysis'] = self._analyze_detection_types(df_results)
        
        # 误报/漏报详细分析
        metrics['error_analysis'] = self._analyze_errors(df_results)
        
        # 风险分数分析
        metrics['risk_score_analysis'] = self._analyze_risk_scores(df_results)
        
        return metrics
    
    def _analyze_performance_by_attack_type(self, df_results: pd.DataFrame) -> Dict:
        """按攻击类型分析性能"""
        performance = {}
        
        for attack_type in df_results['attack_type'].dropna().unique():
            if attack_type and attack_type != 'unknown':
                mask = df_results['attack_type'] == attack_type
                attack_data = df_results[mask]
                
                if len(attack_data) > 0:
                    performance[attack_type] = {
                        'count': len(attack_data),
                        'detection_rate': attack_data['predicted'].mean(),
                        'avg_risk_score': attack_data['risk_score'].mean(),
                        'avg_detection_count': attack_data['detection_count'].mean(),
                        'detection_types': self._get_common_detection_types(attack_data)
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
                    performance[language] = {
                        'count': len(lang_data),
                        'detection_rate': lang_data['predicted'].mean(),
                        'avg_risk_score': lang_data['risk_score'].mean(),
                        'avg_detection_count': lang_data['detection_count'].mean(),
                        'false_negative_rate': (
                            len(lang_data[(lang_data['label'] == 1) & (lang_data['predicted'] == 0)]) /
                            len(lang_data[lang_data['label'] == 1])
                        ) if len(lang_data[lang_data['label'] == 1]) > 0 else 0
                    }
        
        return performance
    
    def _analyze_detection_types(self, df_results: pd.DataFrame) -> Dict:
        """分析检测类型"""
        type_analysis = {}
        
        # 统计所有检测类型
        all_detections = []
        for _, result in df_results.iterrows():
            all_detections.extend(result.get('detections', []))
        
        # 按类型统计
        type_counts = {}
        type_confidence = {}
        type_in_malicious = {}
        type_in_clean = {}
        
        for detection in all_detections:
            det_type = detection.get('type', 'unknown')
            confidence = detection.get('confidence', 0.5)
            
            type_counts[det_type] = type_counts.get(det_type, 0) + 1
            
            if det_type not in type_confidence:
                type_confidence[det_type] = []
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
            
            type_analysis[det_type] = {
                'total_count': type_counts[det_type],
                'avg_confidence': np.mean(type_confidence[det_type]),
                'in_malicious': type_in_malicious.get(det_type, 0),
                'in_clean': type_in_clean.get(det_type, 0),
                'malicious_detection_rate': (
                    type_in_malicious.get(det_type, 0) / total_malicious
                ) if total_malicious > 0 else 0,
                'clean_false_positive_rate': (
                    type_in_clean.get(det_type, 0) / total_clean
                ) if total_clean > 0 else 0,
                'precision': (
                    type_in_malicious.get(det_type, 0) / 
                    (type_in_malicious.get(det_type, 0) + type_in_clean.get(det_type, 0))
                ) if (type_in_malicious.get(det_type, 0) + type_in_clean.get(det_type, 0)) > 0 else 0
            }
        
        return type_analysis
    
    def _get_common_detection_types(self, attack_data: pd.DataFrame) -> List[str]:
        """获取攻击数据中最常见的检测类型"""
        all_types = []
        for _, row in attack_data.iterrows():
            types = row.get('detection_types', '').split(', ')
            all_types.extend([t.strip() for t in types if t.strip()])
        
        from collections import Counter
        type_counts = Counter(all_types)
        return [t for t, _ in type_counts.most_common(3)]
    
    def _analyze_errors(self, df_results: pd.DataFrame) -> Dict:
        """分析误报和漏报"""
        error_analysis = {}
        
        # 误报分析（正常文件被标记为攻击）
        false_positives = df_results[(df_results['label'] == 0) & (df_results['predicted'] == 1)]
        error_analysis['false_positives'] = {
            'count': len(false_positives),
            'common_detection_types': self._get_common_detection_types(false_positives),
            'avg_risk_score': false_positives['risk_score'].mean() if len(false_positives) > 0 else 0,
            'avg_detection_count': false_positives['detection_count'].mean() if len(false_positives) > 0 else 0
        }
        
        # 漏报分析（攻击文件被标记为正常）
        false_negatives = df_results[(df_results['label'] == 1) & (df_results['predicted'] == 0)]
        error_analysis['false_negatives'] = {
            'count': len(false_negatives),
            'by_attack_type': {},
            'by_language': {},
            'avg_risk_score': false_negatives['risk_score'].mean() if len(false_negatives) > 0 else 0,
            'avg_detection_count': false_negatives['detection_count'].mean() if len(false_negatives) > 0 else 0
        }
        
        # 按攻击类型分析漏报
        for attack_type in false_negatives['attack_type'].dropna().unique():
            if attack_type != 'unknown':
                fn_by_type = false_negatives[false_negatives['attack_type'] == attack_type]
                error_analysis['false_negatives']['by_attack_type'][attack_type] = len(fn_by_type)
        
        # 按语言分析漏报
        for language in false_negatives['language'].dropna().unique():
            if language != 'unknown':
                fn_by_lang = false_negatives[false_negatives['language'] == language]
                error_analysis['false_negatives']['by_language'][language] = len(fn_by_lang)
        
        return error_analysis
    
    def _analyze_risk_scores(self, df_results: pd.DataFrame) -> Dict:
        """分析风险分数分布"""
        clean_scores = df_results[df_results['label'] == 0]['risk_score']
        attack_scores = df_results[df_results['label'] == 1]['risk_score']
        
        analysis = {
            'clean_files': {
                'mean': clean_scores.mean() if len(clean_scores) > 0 else 0,
                'std': clean_scores.std() if len(clean_scores) > 0 else 0,
                'min': clean_scores.min() if len(clean_scores) > 0 else 0,
                'max': clean_scores.max() if len(clean_scores) > 0 else 0,
                'median': clean_scores.median() if len(clean_scores) > 0 else 0,
                'q25': clean_scores.quantile(0.25) if len(clean_scores) > 0 else 0,
                'q75': clean_scores.quantile(0.75) if len(clean_scores) > 0 else 0
            },
            'attack_files': {
                'mean': attack_scores.mean() if len(attack_scores) > 0 else 0,
                'std': attack_scores.std() if len(attack_scores) > 0 else 0,
                'min': attack_scores.min() if len(attack_scores) > 0 else 0,
                'max': attack_scores.max() if len(attack_scores) > 0 else 0,
                'median': attack_scores.median() if len(attack_scores) > 0 else 0,
                'q25': attack_scores.quantile(0.25) if len(attack_scores) > 0 else 0,
                'q75': attack_scores.quantile(0.75) if len(attack_scores) > 0 else 0
            }
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
                'overlap_ratio': self._calculate_overlap_ratio(clean_scores, attack_scores)
            }
        
        return analysis
    
    def _calculate_overlap_ratio(self, clean_scores: pd.Series, attack_scores: pd.Series) -> float:
        """计算分数分布的重叠比例"""
        try:
            min_attack = attack_scores.min()
            max_clean = clean_scores.max()
            
            if min_attack >= max_clean:
                return 0.0  # 完全分离
            
            overlap_range = max_clean - min_attack
            total_range = max(attack_scores.max(), clean_scores.max()) - min(attack_scores.min(), clean_scores.min())
            
            return overlap_range / total_range if total_range > 0 else 1.0
        except Exception:
            return 1.0
    
    def _analyze_thresholds(self, df_results: pd.DataFrame) -> Dict:
        """阈值分析"""
        y_true = df_results['label'].values
        y_scores = df_results['risk_score'].values
        
        threshold_analysis = {}
        
        try:
            # 计算不同阈值下的性能
            thresholds = np.arange(0.1, 1.0, 0.05)
            threshold_performance = []
            
            for threshold in thresholds:
                y_pred = (y_scores >= threshold).astype(int)
                
                if len(np.unique(y_pred)) > 1:  # 确保有正负预测
                    precision = precision_score(y_true, y_pred, zero_division=0)
                    recall = recall_score(y_true, y_pred, zero_division=0)
                    f1 = f1_score(y_true, y_pred, zero_division=0)
                    accuracy = accuracy_score(y_true, y_pred)
                    
                    threshold_performance.append({
                        'threshold': threshold,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1,
                        'accuracy': accuracy
                    })
            
            threshold_analysis['performance_curve'] = threshold_performance
            
            # 找到最佳阈值
            if threshold_performance:
                best_f1_idx = max(range(len(threshold_performance)), 
                                key=lambda i: threshold_performance[i]['f1_score'])
                threshold_analysis['optimal_thresholds'] = {
                    'f1_optimal': threshold_performance[best_f1_idx],
                    'precision_recall_curve': self._find_precision_recall_optimal_threshold(y_true, y_scores)
                }
        
        except Exception as e:
            logger.warning(f"阈值分析失败: {e}")
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
    
    def _save_detailed_results(self, df_results: pd.DataFrame, metrics: Dict):
        """保存详细结果"""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存CSV结果
        csv_file = Path(self.output_dir) / f"detection_results_{timestamp}.csv"
        df_results.to_csv(csv_file, index=False, encoding='utf-8')
        
        # 保存JSON指标
        json_file = Path(self.output_dir) / f"metrics_{timestamp}.json"
        save_results(metrics, str(json_file))
        
        # 保存误报和漏报详细信息
        if 'error_analysis' in metrics:
            self._save_error_analysis(df_results, metrics['error_analysis'], timestamp)
        
        logger.info(f"详细结果已保存: {csv_file}, {json_file}")
    
    def _save_error_analysis(self, df_results: pd.DataFrame, error_analysis: Dict, timestamp: str):
        """保存误报和漏报详细分析"""
        # 保存误报文件列表
        false_positives = df_results[(df_results['label'] == 0) & (df_results['predicted'] == 1)]
        if len(false_positives) > 0:
            fp_file = Path(self.output_dir) / f"false_positives_{timestamp}.csv"
            false_positives.to_csv(fp_file, index=False, encoding='utf-8')
            logger.info(f"误报详情已保存: {fp_file}")
        
        # 保存漏报文件列表
        false_negatives = df_results[(df_results['label'] == 1) & (df_results['predicted'] == 0)]
        if len(false_negatives) > 0:
            fn_file = Path(self.output_dir) / f"false_negatives_{timestamp}.csv"
            false_negatives.to_csv(fn_file, index=False, encoding='utf-8')
            logger.info(f"漏报详情已保存: {fn_file}")
    
    def plot_performance_analysis(self, df_results: pd.DataFrame, 
                                metrics: Dict, save_plots: bool = True) -> Dict[str, Any]:
        """绘制性能分析图表 - 改进版"""
        
        config = self.config['experiment']['visualization']
        figsize = tuple(config.get('figsize', [12, 8]))
        dpi = config.get('dpi', 300)
        
        # 创建输出目录
        if save_plots:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            plots_dir = Path(self.output_dir) / f"plots_{timestamp}"
            plots_dir.mkdir(parents=True, exist_ok=True)
        
        plot_files = {}
        
        # 1. 增强的混淆矩阵
        logger.info("生成混淆矩阵...")
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        cm = np.array(metrics['confusion_matrix'])
        
        # 计算百分比
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # 创建标注文本
        annotations = []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations.append(f'{cm[i,j]}\n({cm_percent[i,j]:.1f}%)')
        
        annotations = np.array(annotations).reshape(cm.shape)
        
        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', ax=ax1,
                   xticklabels=['Normal', 'Attack'],
                   yticklabels=['Normal', 'Attack'],
                   cbar_kws={'label': 'Count'},
                   annot_kws={'size': 14, 'weight': 'bold'})
        
        ax1.set_title('Enhanced Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('Predicted Label', fontsize=12)
        ax1.set_ylabel('True Label', fontsize=12)
        
        # 添加性能指标
        metrics_text = f"""Performance Metrics:
Accuracy: {metrics['accuracy']:.3f}
Precision: {metrics['precision']:.3f}
Recall: {metrics['recall']:.3f}
F1-Score: {metrics['f1_score']:.3f}
FPR: {metrics.get('false_positive_rate', 0):.3f}
FNR: {metrics.get('false_negative_rate', 0):.3f}"""
        
        ax1.text(1.05, 0.5, metrics_text, transform=ax1.transAxes, 
                verticalalignment='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        if save_plots:
            plot_file = plots_dir / "01_enhanced_confusion_matrix.png"
            plt.savefig(plot_file, dpi=dpi, bbox_inches='tight')
            plot_files['confusion_matrix'] = plot_file
            logger.info(f"增强混淆矩阵已保存: {plot_file}")
        plt.show()
        plt.close()
        
        # 2. 检测类型效果分析
        if 'detection_type_analysis' in metrics:
            logger.info("生成检测类型效果分析...")
            fig2, ((ax2_1, ax2_2), (ax2_3, ax2_4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            type_analysis = metrics['detection_type_analysis']
            
            # 子图1：检测类型精确率
            types = list(type_analysis.keys())
            precisions = [type_analysis[t]['precision'] for t in types]
            
            bars1 = ax2_1.bar(types, precisions, color='lightblue', alpha=0.8, edgecolor='darkblue')
            ax2_1.set_title('Detection Type Precision', fontsize=14, fontweight='bold')
            ax2_1.set_ylabel('Precision', fontsize=12)
            ax2_1.set_ylim(0, 1.1)
            ax2_1.tick_params(axis='x', rotation=45)
            ax2_1.grid(True, alpha=0.3, axis='y')
            
            # 添加数值标签
            for bar, precision in zip(bars1, precisions):
                height = bar.get_height()
                ax2_1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{precision:.3f}', ha='center', va='bottom', fontsize=10)
            
            # 子图2：检测类型在恶意文件中的检出率
            malicious_rates = [type_analysis[t]['malicious_detection_rate'] for t in types]
            
            bars2 = ax2_2.bar(types, malicious_rates, color='lightcoral', alpha=0.8, edgecolor='darkred')
            ax2_2.set_title('Detection Rate in Malicious Files', fontsize=14, fontweight='bold')
            ax2_2.set_ylabel('Detection Rate', fontsize=12)
            ax2_2.set_ylim(0, 1.1)
            ax2_2.tick_params(axis='x', rotation=45)
            ax2_2.grid(True, alpha=0.3, axis='y')
            
            for bar, rate in zip(bars2, malicious_rates):
                height = bar.get_height()
                ax2_2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{rate:.3f}', ha='center', va='bottom', fontsize=10)
            
            # 子图3：检测类型在正常文件中的误报率
            fp_rates = [type_analysis[t]['clean_false_positive_rate'] for t in types]
            
            bars3 = ax2_3.bar(types, fp_rates, color='orange', alpha=0.8, edgecolor='darkorange')
            ax2_3.set_title('False Positive Rate in Clean Files', fontsize=14, fontweight='bold')
            ax2_3.set_ylabel('False Positive Rate', fontsize=12)
            ax2_3.set_ylim(0, max(fp_rates) * 1.2 if fp_rates else 1)
            ax2_3.tick_params(axis='x', rotation=45)
            ax2_3.grid(True, alpha=0.3, axis='y')
            
            for bar, rate in zip(bars3, fp_rates):
                height = bar.get_height()
                ax2_3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{rate:.4f}', ha='center', va='bottom', fontsize=10)
            
            # 子图4：检测类型总数和平均置信度
            total_counts = [type_analysis[t]['total_count'] for t in types]
            avg_confidences = [type_analysis[t]['avg_confidence'] for t in types]
            
            ax2_4_twin = ax2_4.twinx()
            
            bars4_1 = ax2_4.bar([i - 0.2 for i in range(len(types))], total_counts, 
                               width=0.4, label='Count', color='lightgreen', alpha=0.8)
            bars4_2 = ax2_4_twin.bar([i + 0.2 for i in range(len(types))], avg_confidences, 
                                    width=0.4, label='Avg Confidence', color='gold', alpha=0.8)
            
            ax2_4.set_title('Detection Count vs Average Confidence', fontsize=14, fontweight='bold')
            ax2_4.set_ylabel('Count', fontsize=12, color='green')
            ax2_4_twin.set_ylabel('Average Confidence', fontsize=12, color='orange')
            ax2_4.set_xticks(range(len(types)))
            ax2_4.set_xticklabels(types, rotation=45)
            
            # 图例
            lines1, labels1 = ax2_4.get_legend_handles_labels()
            lines2, labels2 = ax2_4_twin.get_legend_handles_labels()
            ax2_4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            plt.tight_layout()
            if save_plots:
                plot_file = plots_dir / "02_detection_type_analysis.png"
                plt.savefig(plot_file, dpi=dpi, bbox_inches='tight')
                plot_files['detection_type_analysis'] = plot_file
                logger.info(f"检测类型分析已保存: {plot_file}")
            plt.show()
            plt.close()
        
        # 3. 阈值分析图
        if 'threshold_analysis' in metrics and 'performance_curve' in metrics['threshold_analysis']:
            logger.info("生成阈值分析图...")
            fig3, ((ax3_1, ax3_2), (ax3_3, ax3_4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            threshold_data = metrics['threshold_analysis']['performance_curve']
            thresholds = [d['threshold'] for d in threshold_data]
            precisions = [d['precision'] for d in threshold_data]
            recalls = [d['recall'] for d in threshold_data]
            f1_scores = [d['f1_score'] for d in threshold_data]
            accuracies = [d['accuracy'] for d in threshold_data]
            
            # 子图1：精确率-召回率曲线
            ax3_1.plot(thresholds, precisions, 'b-', linewidth=2, label='Precision', marker='o')
            ax3_1.plot(thresholds, recalls, 'r-', linewidth=2, label='Recall', marker='s')
            ax3_1.set_title('Precision-Recall vs Threshold', fontsize=14, fontweight='bold')
            ax3_1.set_xlabel('Threshold', fontsize=12)
            ax3_1.set_ylabel('Score', fontsize=12)
            ax3_1.legend()
            ax3_1.grid(True, alpha=0.3)
            ax3_1.set_ylim(0, 1.1)
            
            # 标记当前阈值
            current_threshold = self.config['detection']['thresholds']['risk_score']
            ax3_1.axvline(x=current_threshold, color='green', linestyle='--', 
                         label=f'Current ({current_threshold})')
            ax3_1.legend()
            
            # 子图2：F1分数曲线
            ax3_2.plot(thresholds, f1_scores, 'g-', linewidth=3, marker='D')
            ax3_2.set_title('F1-Score vs Threshold', fontsize=14, fontweight='bold')
            ax3_2.set_xlabel('Threshold', fontsize=12)
            ax3_2.set_ylabel('F1-Score', fontsize=12)
            ax3_2.grid(True, alpha=0.3)
            ax3_2.set_ylim(0, 1.1)
            
            # 标记最佳阈值
            best_f1_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_f1_idx]
            best_f1 = f1_scores[best_f1_idx]
            ax3_2.scatter([best_threshold], [best_f1], color='red', s=100, zorder=5)
            ax3_2.annotate(f'Best: {best_threshold:.3f}\nF1: {best_f1:.3f}',
                          xy=(best_threshold, best_f1), xytext=(10, 10),
                          textcoords='offset points', fontsize=10,
                          bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))
            
            # 子图3：准确率曲线
            ax3_3.plot(thresholds, accuracies, 'm-', linewidth=2, marker='^')
            ax3_3.set_title('Accuracy vs Threshold', fontsize=14, fontweight='bold')
            ax3_3.set_xlabel('Threshold', fontsize=12)
            ax3_3.set_ylabel('Accuracy', fontsize=12)
            ax3_3.grid(True, alpha=0.3)
            ax3_3.set_ylim(0, 1.1)
            
            # 子图4：综合性能雷达图
            # 选择几个关键阈值的性能
            key_thresholds = [0.3, 0.5, 0.7]
            radar_data = []
            
            for kt in key_thresholds:
                # 找到最接近的阈值
                closest_idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - kt))
                radar_data.append({
                    'threshold': kt,
                    'precision': precisions[closest_idx],
                    'recall': recalls[closest_idx],
                    'f1_score': f1_scores[closest_idx],
                    'accuracy': accuracies[closest_idx]
                })
            
            # 简化为条形图显示几个关键阈值的性能
            metrics_names = ['Precision', 'Recall', 'F1-Score', 'Accuracy']
            x_pos = np.arange(len(metrics_names))
            width = 0.25
            
            for i, data in enumerate(radar_data):
                values = [data['precision'], data['recall'], data['f1_score'], data['accuracy']]
                ax3_4.bar(x_pos + i * width, values, width, 
                         label=f"Threshold {data['threshold']}", alpha=0.8)
            
            ax3_4.set_title('Performance Comparison at Key Thresholds', fontsize=14, fontweight='bold')
            ax3_4.set_ylabel('Score', fontsize=12)
            ax3_4.set_xticks(x_pos + width)
            ax3_4.set_xticklabels(metrics_names)
            ax3_4.legend()
            ax3_4.grid(True, alpha=0.3, axis='y')
            ax3_4.set_ylim(0, 1.1)
            
            plt.tight_layout()
            if save_plots:
                plot_file = plots_dir / "03_threshold_analysis.png"
                plt.savefig(plot_file, dpi=dpi, bbox_inches='tight')
                plot_files['threshold_analysis'] = plot_file
                logger.info(f"阈值分析已保存: {plot_file}")
            plt.show()
            plt.close()
        
        # 4. 误报漏报详细分析
        if 'error_analysis' in metrics:
            logger.info("生成误报漏报分析...")
            fig4, ((ax4_1, ax4_2), (ax4_3, ax4_4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            error_analysis = metrics['error_analysis']
            
            # 子图1：误报分析 - 按检测类型
            fp_detection_types = error_analysis['false_positives']['common_detection_types']
            if fp_detection_types:
                fp_counts = {}
                false_positives = df_results[(df_results['label'] == 0) & (df_results['predicted'] == 1)]
                
                for dt in fp_detection_types:
                    count = sum(1 for _, row in false_positives.iterrows() 
                              if dt in row.get('detection_types', ''))
                    fp_counts[dt] = count
                
                ax4_1.bar(fp_counts.keys(), fp_counts.values(), color='lightcoral', alpha=0.8)
                ax4_1.set_title(f"False Positives by Detection Type (Total: {error_analysis['false_positives']['count']})", 
                               fontsize=14, fontweight='bold')
                ax4_1.set_ylabel('Count', fontsize=12)
                ax4_1.tick_params(axis='x', rotation=45)
                ax4_1.grid(True, alpha=0.3, axis='y')
            else:
                ax4_1.text(0.5, 0.5, 'No False Positives', ha='center', va='center', 
                          transform=ax4_1.transAxes, fontsize=16)
                ax4_1.set_title('False Positives Analysis', fontsize=14, fontweight='bold')
            
            # 子图2：漏报分析 - 按攻击类型
            fn_by_attack_type = error_analysis['false_negatives']['by_attack_type']
            if fn_by_attack_type:
                ax4_2.bar(fn_by_attack_type.keys(), fn_by_attack_type.values(), 
                         color='orange', alpha=0.8)
                ax4_2.set_title(f"False Negatives by Attack Type (Total: {error_analysis['false_negatives']['count']})", 
                               fontsize=14, fontweight='bold')
                ax4_2.set_ylabel('Count', fontsize=12)
                ax4_2.tick_params(axis='x', rotation=45)
                ax4_2.grid(True, alpha=0.3, axis='y')
            else:
                ax4_2.text(0.5, 0.5, 'No False Negatives', ha='center', va='center', 
                          transform=ax4_2.transAxes, fontsize=16)
                ax4_2.set_title('False Negatives by Attack Type', fontsize=14, fontweight='bold')
            
            # 子图3：漏报分析 - 按语言
            fn_by_language = error_analysis['false_negatives']['by_language']
            if fn_by_language:
                ax4_3.bar(fn_by_language.keys(), fn_by_language.values(), 
                         color='gold', alpha=0.8)
                ax4_3.set_title('False Negatives by Language', fontsize=14, fontweight='bold')
                ax4_3.set_ylabel('Count', fontsize=12)
                ax4_3.grid(True, alpha=0.3, axis='y')
            else:
                ax4_3.text(0.5, 0.5, 'No Language-specific FN', ha='center', va='center', 
                          transform=ax4_3.transAxes, fontsize=16)
                ax4_3.set_title('False Negatives by Language', fontsize=14, fontweight='bold')
            
            # 子图4：错误类型风险分数分布
            fp_scores = df_results[(df_results['label'] == 0) & (df_results['predicted'] == 1)]['risk_score']
            fn_scores = df_results[(df_results['label'] == 1) & (df_results['predicted'] == 0)]['risk_score']
            
            if len(fp_scores) > 0:
                ax4_4.hist(fp_scores, alpha=0.7, label=f'False Positives (n={len(fp_scores)})', 
                          bins=15, color='red', density=True)
            
            if len(fn_scores) > 0:
                ax4_4.hist(fn_scores, alpha=0.7, label=f'False Negatives (n={len(fn_scores)})', 
                          bins=15, color='orange', density=True)
            
            # 添加阈值线
            threshold = self.config['detection']['thresholds']['risk_score']
            ax4_4.axvline(x=threshold, color='green', linestyle='--', linewidth=2, 
                         label=f'Threshold ({threshold})')
            
            ax4_4.set_title('Risk Score Distribution of Errors', fontsize=14, fontweight='bold')
            ax4_4.set_xlabel('Risk Score', fontsize=12)
            ax4_4.set_ylabel('Density', fontsize=12)
            ax4_4.legend()
            ax4_4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            if save_plots:
                plot_file = plots_dir / "04_error_analysis.png"
                plt.savefig(plot_file, dpi=dpi, bbox_inches='tight')
                plot_files['error_analysis'] = plot_file
                logger.info(f"误报漏报分析已保存: {plot_file}")
            plt.show()
            plt.close()
        
        # 5-9. 保持原有的其他图表...
        # (这里可以继续添加其他图表，如风险分数分布、语言性能等)
        
        # 生成改进的HTML报告
        if save_plots:
            self._generate_enhanced_html_report(plots_dir, plot_files, metrics, df_results)
            logger.info(f"所有图表已保存到: {plots_dir}")
        
        return {'plots_directory': plots_dir if save_plots else None, 'plot_files': plot_files}
    
    def _generate_enhanced_html_report(self, plots_dir: Path, plot_files: Dict[str, Path], 
                                     metrics: Dict, df_results: pd.DataFrame):
        """生成增强的HTML报告"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Enhanced Detection Performance Analysis Report</title>
    <meta charset="UTF-8">
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        .timestamp {{
            opacity: 0.9;
            font-style: italic;
            margin-top: 10px;
        }}
        .content {{
            padding: 30px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .metric-card {{
            background: linear-gradient(145deg, #f8f9ff, #e8ecff);
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            border: 1px solid #e1e8ff;
            transition: transform 0.3s ease;
        }}
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }}
        .metric-value {{
            font-size: 2.2em;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .metric-value.excellent {{ color: #27ae60; }}
        .metric-value.good {{ color: #f39c12; }}
        .metric-value.poor {{ color: #e74c3c; }}
        .metric-label {{
            font-size: 1.1em;
            color: #666;
            font-weight: 500;
        }}
        .section {{
            margin: 40px 0;
            padding: 30px;
            background: #fafbfc;
            border-radius: 15px;
            border-left: 5px solid #667eea;
        }}
        .section h2 {{
            color: #2c3e50;
            margin-top: 0;
            font-size: 1.8em;
        }}
        .plot-container {{ 
            margin: 30px 0; 
            padding: 25px; 
            background: white;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            border: 1px solid #eef2f7;
        }}
        .plot-title {{ 
            font-size: 1.4em; 
            font-weight: bold; 
            margin-bottom: 15px; 
            color: #2c3e50;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        img {{ 
            max-width: 100%; 
            height: auto; 
            border-radius: 10px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }}
        .summary-stats {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin: 30px 0;
        }}
        .stat-box {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }}
        .stat-box h3 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        .recommendation {{
            background: linear-gradient(135deg, #fff3cd, #ffeaa7);
            border: 1px solid #ffeaa7;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
        }}
        .recommendation h3 {{
            color: #856404;
            margin-top: 0;
        }}
        .footer {{
            text-align: center;
            padding: 30px;
            background: #f8f9fa;
            color: #6c757d;
            border-top: 1px solid #e9ecef;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ Enhanced Detection Performance Analysis</h1>
            <p class="timestamp">Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>
        
        <div class="content">
            <div class="section">
                <h2>📊 Performance Overview</h2>
                <div class="metrics-grid">
                    {self._generate_metric_card('Accuracy', metrics.get('accuracy', 0))}
                    {self._generate_metric_card('Precision', metrics.get('precision', 0))}
                    {self._generate_metric_card('Recall', metrics.get('recall', 0))}
                    {self._generate_metric_card('F1-Score', metrics.get('f1_score', 0))}
                    {self._generate_metric_card('ROC AUC', metrics.get('roc_auc', 0))}
                    {self._generate_metric_card('PR AUC', metrics.get('pr_auc', 0))}
                </div>
            </div>
            
            <div class="section">
                <h2>📈 Dataset Summary</h2>
                <div class="summary-stats">
                    <div class="stat-box">
                        <h3>Dataset Composition</h3>
                        <p><strong>Total Files:</strong> {metrics.get('support', {}).get('total', 0)}</p>
                        <p><strong>Normal Files:</strong> {metrics.get('support', {}).get('clean', 0)}</p>
                        <p><strong>Attack Files:</strong> {metrics.get('support', {}).get('attack', 0)}</p>
                        <p><strong>Attack Ratio:</strong> {(metrics.get('support', {}).get('attack', 0) / metrics.get('support', {}).get('total', 1) * 100):.1f}%</p>
                    </div>
                    <div class="stat-box">
                        <h3>Error Analysis</h3>
                        <p><strong>False Positives:</strong> {metrics.get('error_analysis', {}).get('false_positives', {}).get('count', 0)}</p>
                        <p><strong>False Negatives:</strong> {metrics.get('error_analysis', {}).get('false_negatives', {}).get('count', 0)}</p>
                        <p><strong>False Positive Rate:</strong> {metrics.get('false_positive_rate', 0):.3f}</p>
                        <p><strong>False Negative Rate:</strong> {metrics.get('false_negative_rate', 0):.3f}</p>
                    </div>
                </div>
            </div>
            
            {self._generate_recommendations_section(metrics)}
"""
        
        # 添加图表部分
        plot_descriptions = {
            'confusion_matrix': '🎯 Enhanced Confusion Matrix - 显示分类准确性，包含数量和百分比',
            'detection_type_analysis': '🔍 Detection Type Analysis - 各种检测机制的效果分析',
            'threshold_analysis': '⚖️ Threshold Analysis - 阈值对性能的影响分析',
            'error_analysis': '❌ Error Analysis - 误报和漏报详细分析',
            'risk_distribution': '📊 Risk Score Distribution - 风险分数分布对比',
            'performance_metrics': '📈 Performance Metrics - 关键性能指标',
            'roc_pr_curves': '📉 ROC and PR Curves - 分类器性能曲线',
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
        </div>
        
        <div class="footer">
            <p>📧 Generated by Enhanced Paper Review Attack Detection System</p>
            <p>🔬 Advanced ML-based Detection with Adaptive Thresholding</p>
        </div>
    </div>
</body>
</html>
"""
        
        html_file = plots_dir / "enhanced_report.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"📋 增强HTML报告已生成: {html_file}")
    
    def _generate_metric_card(self, label: str, value: float) -> str:
        """生成指标卡片HTML"""
        # 根据数值确定颜色类别
        if value >= 0.8:
            css_class = "excellent"
        elif value >= 0.6:
            css_class = "good"
        else:
            css_class = "poor"
        
        return f"""
                    <div class="metric-card">
                        <div class="metric-value {css_class}">{value:.3f}</div>
                        <div class="metric-label">{label}</div>
                    </div>"""
    
    def _generate_recommendations_section(self, metrics: Dict) -> str:
        """生成建议部分"""
        recommendations = []
        
        # 基于性能指标生成建议
        if metrics.get('f1_score', 0) < 0.7:
            recommendations.append("🎯 F1分数偏低，建议优化检测算法或调整阈值")
        
        if metrics.get('false_positive_rate', 0) > 0.1:
            recommendations.append("⚠️ 误报率较高，建议提高检测精度或调整权重")
        
        if metrics.get('false_negative_rate', 0) > 0.1:
            recommendations.append("🔍 漏报率较高，建议增强检测覆盖度")
        
        if metrics.get('roc_auc', 0) < 0.8:
            recommendations.append("📈 AUC偏低，建议改进特征工程或模型算法")
        
        # 检测类型相关建议
        if 'detection_type_analysis' in metrics:
            type_analysis = metrics['detection_type_analysis']
            high_fp_types = [t for t, data in type_analysis.items() 
                           if data.get('clean_false_positive_rate', 0) > 0.1]
            if high_fp_types:
                recommendations.append(f"🎭 以下检测类型误报率较高，建议调整: {', '.join(high_fp_types[:3])}")
        
        if not recommendations:
            recommendations.append("✅ 系统性能良好，建议继续监控和维护")
        
        recommendations_html = ""
        if recommendations:
            recommendations_html = """
            <div class="section">
                <h2>💡 Performance Recommendations</h2>"""
            
            for rec in recommendations:
                recommendations_html += f"""
                <div class="recommendation">
                    <p>{rec}</p>
                </div>"""
            
            recommendations_html += "</div>"
        
        return recommendations_html
    
    def generate_enhanced_report(self, df_results: pd.DataFrame, metrics: Dict) -> str:
        """生成增强的详细报告"""
        
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""
# 📊 Enhanced Paper Review Attack Detection Report

🕒 **Generated**: {timestamp}

## 🎯 Executive Summary

本报告详细分析了论文审稿攻击检测系统的性能表现，涵盖了准确性、效率和鲁棒性等多个维度。

### 🏆 Key Performance Indicators
- **Overall Accuracy**: {metrics['accuracy']:.3f}
- **Detection Precision**: {metrics['precision']:.3f}  
- **Detection Recall**: {metrics['recall']:.3f}
- **F1-Score**: {metrics['f1_score']:.3f}
- **ROC AUC**: {metrics['roc_auc']:.3f}

## 📈 Dataset Overview

### 📁 Data Composition
| Category | Count | Percentage |
|----------|--------|------------|
| Normal Files | {metrics['support']['clean']} | {metrics['support']['clean']/metrics['support']['total']*100:.1f}% |
| Attack Files | {metrics['support']['attack']} | {metrics['support']['attack']/metrics['support']['total']*100:.1f}% |
| **Total** | **{metrics['support']['total']}** | **100.0%** |

## 🎭 Detection Performance Analysis

### 🎯 Confusion Matrix Analysis
"""
        
        if 'confusion_matrix_details' in metrics:
            cm_details = metrics['confusion_matrix_details']
            report += f"""
| Metric | Value | Description |
|--------|--------|-------------|
| True Negatives | {cm_details['true_negative']} | Correctly identified normal files |
| False Positives | {cm_details['false_positive']} | Normal files flagged as attacks |
| False Negatives | {cm_details['false_negative']} | Missed attack files |
| True Positives | {cm_details['true_positive']} | Correctly detected attacks |

### 📊 Error Rates
- **False Positive Rate**: {metrics['false_positive_rate']:.3f}
- **False Negative Rate**: {metrics['false_negative_rate']:.3f}
- **Specificity**: {metrics.get('specificity', 0):.3f}
- **Sensitivity**: {metrics.get('sensitivity', 0):.3f}
"""
        
        # 按攻击类型的性能分析
        if 'performance_by_attack_type' in metrics:
            report += "\n## 🏹 Attack Type Performance Analysis\n\n"
            report += "| Attack Type | Samples | Detection Rate | Avg Risk Score | Top Detection Methods |\n"
            report += "|-------------|---------|----------------|----------------|----------------------|\n"
            
            for attack_type, perf in metrics['performance_by_attack_type'].items():
                detection_methods = ', '.join(perf.get('detection_types', [])[:2])
                report += f"| {attack_type} | {perf['count']} | {perf['detection_rate']:.3f} | {perf['avg_risk_score']:.3f} | {detection_methods} |\n"
        
        # 按语言的性能分析
        if 'performance_by_language' in metrics:
            report += "\n## 🌍 Language-specific Performance\n\n"
            report += "| Language | Samples | Detection Rate | Avg Risk Score | FN Rate |\n"
            report += "|----------|---------|----------------|----------------|---------|\n"
            
            for language, perf in metrics['performance_by_language'].items():
                report += f"| {language} | {perf['count']} | {perf['detection_rate']:.3f} | {perf['avg_risk_score']:.3f} | {perf.get('false_negative_rate', 0):.3f} |\n"
        
        # 检测类型效果分析
        if 'detection_type_analysis' in metrics:
            report += "\n## 🔍 Detection Mechanism Analysis\n\n"
            report += "| Detection Type | Total Count | Precision | Malicious Detection Rate | FP Rate |\n"
            report += "|----------------|-------------|-----------|--------------------------|----------|\n"
            
            type_analysis = metrics['detection_type_analysis']
            for det_type, analysis in type_analysis.items():
                report += f"| {det_type} | {analysis['total_count']} | {analysis['precision']:.3f} | {analysis['malicious_detection_rate']:.3f} | {analysis['clean_false_positive_rate']:.4f} |\n"
        
        # 风险分数分析
        if 'risk_score_analysis' in metrics:
            rsa = metrics['risk_score_analysis']
            report += f"""
## 📊 Risk Score Distribution Analysis

### 📋 Normal Files Risk Scores
- **Mean**: {rsa['clean_files']['mean']:.3f} ± {rsa['clean_files']['std']:.3f}
- **Range**: [{rsa['clean_files']['min']:.3f}, {rsa['clean_files']['max']:.3f}]
- **Median**: {rsa['clean_files']['median']:.3f}
- **IQR**: [{rsa['clean_files']['q25']:.3f}, {rsa['clean_files']['q75']:.3f}]

### 🚨 Attack Files Risk Scores  
- **Mean**: {rsa['attack_files']['mean']:.3f} ± {rsa['attack_files']['std']:.3f}
- **Range**: [{rsa['attack_files']['min']:.3f}, {rsa['attack_files']['max']:.3f}]
- **Median**: {rsa['attack_files']['median']:.3f}
- **IQR**: [{rsa['attack_files']['q25']:.3f}, {rsa['attack_files']['q75']:.3f}]

### 🎯 Separation Analysis
"""
            
            if 'separation' in rsa:
                sep = rsa['separation']
                report += f"""- **Mean Difference**: {sep['mean_difference']:.3f}
- **Cohen's d (Effect Size)**: {sep['cohens_d']:.3f}
- **Distribution Overlap**: {sep['overlap_ratio']:.3f}
"""
        
        # 阈值分析
        if 'threshold_analysis' in metrics and 'optimal_thresholds' in metrics['threshold_analysis']:
            ot = metrics['threshold_analysis']['optimal_thresholds']
            current_threshold = self.config['detection']['thresholds']['risk_score']
            
            report += f"""
## ⚖️ Threshold Optimization Analysis

### 🎯 Current vs Optimal Thresholds
- **Current Threshold**: {current_threshold}
"""
            
            if 'f1_optimal' in ot:
                f1_opt = ot['f1_optimal']
                report += f"""- **F1-Optimal Threshold**: {f1_opt['threshold']:.3f}
  - Precision: {f1_opt['precision']:.3f}
  - Recall: {f1_opt['recall']:.3f}  
  - F1-Score: {f1_opt['f1_score']:.3f}
"""
        
        # 错误分析
        if 'error_analysis' in metrics:
            ea = metrics['error_analysis']
            report += f"""
## ❌ Error Analysis

### 🚫 False Positives Analysis
- **Count**: {ea['false_positives']['count']}
- **Average Risk Score**: {ea['false_positives']['avg_risk_score']:.3f}
- **Average Detection Count**: {ea['false_positives']['avg_detection_count']:.1f}
- **Common Detection Types**: {', '.join(ea['false_positives']['common_detection_types'])}

### 🎯 False Negatives Analysis  
- **Count**: {ea['false_negatives']['count']}
- **Average Risk Score**: {ea['false_negatives']['avg_risk_score']:.3f}
- **Average Detection Count**: {ea['false_negatives']['avg_detection_count']:.1f}

#### By Attack Type:
"""
            for attack_type, count in ea['false_negatives']['by_attack_type'].items():
                report += f"- **{attack_type}**: {count} missed detections\n"
            
            report += "\n#### By Language:\n"
            for language, count in ea['false_negatives']['by_language'].items():
                report += f"- **{language}**: {count} missed detections\n"
        
        # 建议和改进方向
        report += "\n## 💡 Recommendations and Improvements\n\n"
        
        recommendations = []
        
        # 基于性能生成建议
        if metrics['f1_score'] < 0.7:
            recommendations.append("🎯 **Low F1-Score**: Consider algorithm optimization or threshold adjustment")
        
        if metrics.get('false_positive_rate', 0) > 0.1:
            recommendations.append("⚠️ **High FPR**: Improve detection precision by refining keyword lists or adjusting weights")
        
        if metrics.get('false_negative_rate', 0) > 0.1:
            recommendations.append("🔍 **High FNR**: Enhance detection coverage by adding new detection dimensions")
        
        if metrics.get('roc_auc', 0) < 0.8:
            recommendations.append("📈 **Low AUC**: Improve feature engineering or risk score calculation method")
        
        # 语言特定建议
        if 'performance_by_language' in metrics:
            lang_perf = metrics['performance_by_language']
            weak_languages = [lang for lang, perf in lang_perf.items() 
                            if perf['detection_rate'] < 0.8]
            if weak_languages:
                recommendations.append(f"🌍 **Language Issues**: Improve detection for: {', '.join(weak_languages)}")
        
        # 检测类型建议
        if 'detection_type_analysis' in metrics:
            type_analysis = metrics['detection_type_analysis']
            high_fp_types = [t for t, data in type_analysis.items() 
                           if data.get('clean_false_positive_rate', 0) > 0.1]
            if high_fp_types:
                recommendations.append(f"🎭 **High FP Types**: Adjust weights for: {', '.join(high_fp_types[:3])}")
        
        if not recommendations:
            recommendations.append("✅ **System Performance Good**: Continue monitoring and maintenance")
        
        for i, rec in enumerate(recommendations, 1):
            report += f"{i}. {rec}\n"
        
        # 保存报告
        timestamp_file = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        report_file = Path(self.output_dir) / f"enhanced_experiment_report_{timestamp_file}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"增强实验报告已保存: {report_file}")
        
        return report

    def compare_experiments(self, experiment_results: List[Dict]) -> Dict:
        """比较多个实验结果 - 改进版"""
        if len(experiment_results) < 2:
            logger.warning("需要至少2个实验结果进行比较")
            return {}
        
        comparison = {
            'experiment_count': len(experiment_results),
            'metrics_comparison': {},
            'best_experiment': {},
            'improvement_analysis': {},
            'trend_analysis': {}
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
                'improvement': max(values) - min(values),
                'std_dev': np.std(values),
                'trend': 'improving' if values[-1] > values[0] else 'declining'
            }
        
        # 确定最佳实验
        f1_scores = [exp.get('f1_score', 0) for exp in experiment_results]
        best_idx = np.argmax(f1_scores)
        comparison['best_experiment'] = {
            'index': best_idx,
            'f1_score': f1_scores[best_idx],
            'all_metrics': experiment_results[best_idx]
        }
        
        # 改进分析
        if len(experiment_results) >= 2:
            latest = experiment_results[-1]
            previous = experiment_results[-2]
            
            comparison['improvement_analysis'] = {
                'accuracy_change': latest.get('accuracy', 0) - previous.get('accuracy', 0),
                'precision_change': latest.get('precision', 0) - previous.get('precision', 0),
                'recall_change': latest.get('recall', 0) - previous.get('recall', 0),
                'f1_change': latest.get('f1_score', 0) - previous.get('f1_score', 0)
            }
        
        return comparison

    def generate_report(self, df_results: pd.DataFrame, metrics: Dict) -> str:
        """生成详细报告 - 兼容性方法，调用增强版报告"""
        logger.info("调用兼容性报告生成方法，转向增强版报告")
        return self.generate_enhanced_report(df_results, metrics)
