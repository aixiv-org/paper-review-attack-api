import pdfplumber
import fitz
import re
import base64
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score
import nltk
from collections import Counter, defaultdict, deque
import warnings
import sys
import os
import io
import time
from contextlib import redirect_stderr, contextmanager
from .utils import (
    setup_logging, detect_language, clean_text, 
    PerformanceMonitor, CacheManager, check_model_availability
)

logger = setup_logging()

# 全局错误抑制
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

@contextmanager
def suppress_stderr():
    """抑制标准错误输出的上下文管理器"""
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

@contextmanager
def suppress_all_output():
    """抑制所有输出的上下文管理器"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

class ModelManager:
    """智能模型管理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_config = config.get('detection', {}).get('model_management', {})
        self.models = {}
        self.model_status = {}
        self.fallback_level = 0
        
        # 初始化缓存
        self.cache_manager = CacheManager(config)
        
        # 模型级联配置
        self.model_cascade = self.model_config.get('model_cascade', {})
        
        logger.info("模型管理器初始化中...")
        self._initialize_models()
    
    def _initialize_models(self):
        """级联初始化模型"""
        self.sentiment_analyzer = self._init_sentiment_model()
        self.multilingual_model, self.tokenizer = self._init_multilingual_model()
        
        # 记录初始化状态
        self.model_status = {
            'sentiment_analyzer': self.sentiment_analyzer is not None,
            'multilingual_model': self.multilingual_model is not None,
            'fallback_level': self.fallback_level
        }
        
        logger.info(f"模型初始化完成，状态: {self.model_status}")
    
    def _init_sentiment_model(self):
        """级联初始化情感分析模型"""
        cascade_config = self.model_cascade.get('sentiment_analysis', {})
        primary_model = cascade_config.get('primary', 'cardiffnlp/twitter-roberta-base-sentiment-latest')
        fallbacks = cascade_config.get('fallback', ['textblob', 'vader', 'rule_based'])
        
        # 尝试主要模型
        if not self.model_config.get('offline_first', True):
            try:
                logger.info(f"尝试加载主要情感模型: {primary_model}")
                
                if primary_model.startswith('local://'):
                    model_path = primary_model[8:]  # 移除 'local://' 前缀
                    if os.path.exists(model_path):
                        from transformers import pipeline
                        analyzer = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
                        logger.info(f"✅ 本地模型加载成功: {model_path}")
                        return analyzer
                else:
                    if check_model_availability(primary_model, 'huggingface'):
                        from transformers import pipeline
                        with suppress_stderr():
                            analyzer = pipeline(
                                "sentiment-analysis",
                                model=primary_model,
                                return_all_scores=True,
                                device=-1  # 使用CPU
                            )
                        logger.info(f"✅ HuggingFace模型加载成功: {primary_model}")
                        return analyzer
            
            except Exception as e:
                logger.warning(f"主要模型加载失败: {e}")
        
        # 尝试备用方案
        for fallback in fallbacks:
            try:
                logger.info(f"尝试备用情感分析: {fallback}")
                
                if fallback == 'textblob':
                    if check_model_availability('textblob', 'textblob'):
                        analyzer = self._create_textblob_analyzer()
                        logger.info("✅ TextBlob情感分析器就绪")
                        self.fallback_level = 1
                        return analyzer
                
                elif fallback == 'vader':
                    if check_model_availability('vader', 'vader'):
                        analyzer = self._create_vader_analyzer()
                        logger.info("✅ VADER情感分析器就绪")
                        self.fallback_level = 2
                        return analyzer
                
                elif fallback == 'rule_based':
                    analyzer = self._create_rule_based_analyzer()
                    logger.info("✅ 基于规则的情感分析器就绪")
                    self.fallback_level = 3
                    return analyzer
                    
            except Exception as e:
                logger.warning(f"备用方案 {fallback} 失败: {e}")
                continue
        
        logger.error("所有情感分析模型都加载失败")
        return None
    
    def _create_textblob_analyzer(self):
        """创建TextBlob分析器"""
        from textblob import TextBlob
        
        class TextBlobAnalyzer:
            def __call__(self, text: str):
                try:
                    blob = TextBlob(text)
                    polarity = blob.sentiment.polarity
                    
                    if polarity > 0.1:
                        return [{'label': 'POSITIVE', 'score': min(0.9, 0.5 + polarity * 0.4)}]
                    elif polarity < -0.1:
                        return [{'label': 'NEGATIVE', 'score': min(0.9, 0.5 - polarity * 0.4)}]
                    else:
                        return [{'label': 'NEUTRAL', 'score': 0.6}]
                except Exception:
                    return [{'label': 'NEUTRAL', 'score': 0.5}]
        
        return TextBlobAnalyzer()
    
    def _create_vader_analyzer(self):
        """创建VADER分析器"""
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        
        class VaderAnalyzer:
            def __init__(self):
                self.analyzer = SentimentIntensityAnalyzer()
            
            def __call__(self, text: str):
                try:
                    scores = self.analyzer.polarity_scores(text)
                    compound = scores['compound']
                    
                    if compound >= 0.05:
                        return [{'label': 'POSITIVE', 'score': min(0.9, 0.5 + compound * 0.4)}]
                    elif compound <= -0.05:
                        return [{'label': 'NEGATIVE', 'score': min(0.9, 0.5 - compound * 0.4)}]
                    else:
                        return [{'label': 'NEUTRAL', 'score': 0.6}]
                except Exception:
                    return [{'label': 'NEUTRAL', 'score': 0.5}]
        
        return VaderAnalyzer()
    
    def _create_rule_based_analyzer(self):
        """创建基于规则的分析器"""
        
        class RuleBasedAnalyzer:
            def __init__(self):
                self.positive_words = {
                    'excellent', 'outstanding', 'superb', 'brilliant', 'amazing',
                    'fantastic', 'wonderful', 'great', 'good', 'perfect',
                    'innovative', 'groundbreaking', 'revolutionary', 'exceptional',
                    '优秀', '卓越', '杰出', '完美', '出色', '创新', '突破性'
                }
                
                self.negative_words = {
                    'terrible', 'awful', 'horrible', 'bad', 'poor', 'weak',
                    'inadequate', 'insufficient', 'flawed', 'problematic',
                    '糟糕', '差劲', '不足', '缺陷', '问题'
                }
            
            def __call__(self, text: str):
                try:
                    text_lower = text.lower()
                    words = set(text_lower.split())
                    
                    positive_count = len(words.intersection(self.positive_words))
                    negative_count = len(words.intersection(self.negative_words))
                    
                    if positive_count > negative_count:
                        score = min(0.85, 0.6 + positive_count * 0.05)
                        return [{'label': 'POSITIVE', 'score': score}]
                    elif negative_count > positive_count:
                        score = min(0.85, 0.6 + negative_count * 0.05)
                        return [{'label': 'NEGATIVE', 'score': score}]
                    else:
                        return [{'label': 'NEUTRAL', 'score': 0.5}]
                        
                except Exception:
                    return [{'label': 'NEUTRAL', 'score': 0.5}]
        
        return RuleBasedAnalyzer()
    
    def _init_multilingual_model(self):
        """初始化多语言模型"""
        cascade_config = self.model_cascade.get('multilingual', {})
        primary_model = cascade_config.get('primary', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        fallbacks = cascade_config.get('fallback', ['language_detection', 'rule_based'])
        
        # 尝试主要模型
        if not self.model_config.get('offline_first', True):
            try:
                logger.info(f"尝试加载多语言模型: {primary_model}")
                
                if primary_model.startswith('local://'):
                    model_path = primary_model[8:]
                    if os.path.exists(model_path):
                        from transformers import AutoTokenizer, AutoModel
                        tokenizer = AutoTokenizer.from_pretrained(model_path)
                        model = AutoModel.from_pretrained(model_path)
                        logger.info(f"✅ 本地多语言模型加载成功: {model_path}")
                        return model, tokenizer
                else:
                    if check_model_availability(primary_model, 'huggingface'):
                        from transformers import AutoTokenizer, AutoModel
                        with suppress_stderr():
                            tokenizer = AutoTokenizer.from_pretrained(primary_model)
                            model = AutoModel.from_pretrained(primary_model)
                        logger.info(f"✅ HuggingFace多语言模型加载成功: {primary_model}")
                        return model, tokenizer
            
            except Exception as e:
                logger.warning(f"多语言模型加载失败: {e}")
        
        # 备用方案
        logger.info("使用多语言检测备用方案")
        return None, None
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            'status': self.model_status,
            'fallback_level': self.fallback_level,
            'available_models': {
                'sentiment': self.sentiment_analyzer is not None,
                'multilingual': self.multilingual_model is not None
            }
        }

class IntelligentThresholdManager:
    """智能阈值管理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.threshold_config = config.get('detection', {}).get('intelligent_thresholds', {})
        self.enabled = self.threshold_config.get('enabled', True)
        
        # 基础阈值
        self.base_thresholds = config.get('detection', {}).get('thresholds', {})
        self.current_thresholds = self.base_thresholds.copy()
        
        # 自适应配置
        self.adaptive_config = self.threshold_config.get('adaptive_system', {})
        self.learning_rate = self.adaptive_config.get('learning_rate', 0.01)
        self.momentum = self.adaptive_config.get('momentum', 0.9)
        
        # 性能历史
        self.performance_history = deque(maxlen=100)
        self.adjustment_history = deque(maxlen=50)
        
        # 分层阈值
        self.layered_thresholds = self.threshold_config.get('layered_thresholds', {})
        
        logger.info(f"智能阈值管理器初始化，启用状态: {self.enabled}")
    
    def get_threshold(self, detection_type: str, context: Dict = None) -> float:
        """获取智能阈值"""
        if not self.enabled:
            return self.base_thresholds.get(detection_type, 0.5)
        
        # 基础阈值
        base_threshold = self.current_thresholds.get(detection_type, 0.5)
        
        # 上下文调整
        if context:
            adjustment = self._calculate_context_adjustment(detection_type, context)
            adjusted_threshold = base_threshold + adjustment
            
            # 限制在合理范围内
            min_threshold = self.adaptive_config.get('min_threshold', 0.1)
            max_threshold = self.adaptive_config.get('max_threshold', 0.9)
            
            return max(min_threshold, min(max_threshold, adjusted_threshold))
        
        return base_threshold
    
    def _calculate_context_adjustment(self, detection_type: str, context: Dict) -> float:
        """计算上下文调整"""
        adjustment = 0.0
        
        # 文档长度调整
        text_length = context.get('text_length', 1000)
        if text_length < 500:
            adjustment += 0.05  # 短文档提高阈值
        elif text_length > 5000:
            adjustment -= 0.02  # 长文档降低阈值
        
        # 检测密度调整
        detection_count = context.get('detection_count', 0)
        detection_density = detection_count / (text_length / 1000) if text_length > 0 else 0
        
        if detection_density > 2.0:
            adjustment += min(0.1, detection_density * 0.03)  # 高密度提高阈值
        
        # 特定检测类型调整
        if detection_type == 'small_text_injection':
            small_text_ratio = context.get('small_text_ratio', 0)
            if small_text_ratio > 0.05:
                adjustment += min(0.15, small_text_ratio * 2)
        
        elif detection_type == 'contextual_anomaly':
            adjustment += 0.1  # 上下文异常本身就容易误报
        
        return adjustment
    
    def update_performance(self, performance_metrics: Dict):
        """更新性能指标"""
        if not self.enabled:
            return
        
        self.performance_history.append({
            'timestamp': time.time(),
            'metrics': performance_metrics
        })
        
        # 自适应调整
        if self.adaptive_config.get('enabled', False):
            self._adaptive_threshold_adjustment(performance_metrics)
    
    def _adaptive_threshold_adjustment(self, metrics: Dict):
        """自适应阈值调整"""
        target_f1 = self.adaptive_config.get('performance_targets', {}).get('target_value', 0.75)
        current_f1 = metrics.get('f1_score', 0.5)
        
        # 计算调整
        f1_diff = target_f1 - current_f1
        
        if abs(f1_diff) > 0.05:  # 只在差异较大时调整
            # 基于性能差异调整主要阈值
            risk_score_adjustment = f1_diff * self.learning_rate
            
            # 应用动量
            if self.adjustment_history:
                last_adjustment = self.adjustment_history[-1].get('risk_score_adjustment', 0)
                risk_score_adjustment = (1 - self.momentum) * risk_score_adjustment + self.momentum * last_adjustment
            
            # 更新阈值
            old_threshold = self.current_thresholds.get('risk_score', 0.5)
            new_threshold = old_threshold - risk_score_adjustment  # 注意方向
            
            # 限制调整幅度
            min_threshold = self.adaptive_config.get('min_threshold', 0.1)
            max_threshold = self.adaptive_config.get('max_threshold', 0.9)
            new_threshold = max(min_threshold, min(max_threshold, new_threshold))
            
            self.current_thresholds['risk_score'] = new_threshold
            
            # 记录调整
            self.adjustment_history.append({
                'timestamp': time.time(),
                'old_threshold': old_threshold,
                'new_threshold': new_threshold,
                'risk_score_adjustment': risk_score_adjustment,
                'trigger_metric': f1_diff
            })
            
            logger.info(f"自适应阈值调整: {old_threshold:.3f} -> {new_threshold:.3f} (F1差异: {f1_diff:.3f})")
    
    def get_layered_decision(self, risk_score: float, detection_count: int, 
                           confidence_sum: float, detection_types: set) -> Dict:
        """分层决策"""
        if not self.layered_thresholds.get('enabled', False):
            # 传统单一阈值决策
            threshold = self.get_threshold('risk_score')
            return {
                'is_malicious': risk_score > threshold,
                'confidence_level': 'standard',
                'decision_layer': 1,
                'threshold_used': threshold
            }
        
        # 第一层：快速筛选
        quick_filter = self.layered_thresholds.get('quick_filter', {})
        if (risk_score > quick_filter.get('risk_score', 0.2) and 
            detection_count >= quick_filter.get('detection_count', 1)):
            
            # 第二层：详细分析
            detailed_analysis = self.layered_thresholds.get('detailed_analysis', {})
            if (risk_score > detailed_analysis.get('risk_score', 0.35) and
                detection_count >= detailed_analysis.get('detection_count', 2) and
                confidence_sum > detailed_analysis.get('confidence_sum', 1.5)):
                
                # 第三层：高置信度确认
                high_confidence = self.layered_thresholds.get('high_confidence', {})
                if (risk_score > high_confidence.get('risk_score', 0.6) and
                    detection_count >= high_confidence.get('detection_count', 3) and
                    (not high_confidence.get('multiple_types', False) or len(detection_types) > 1)):
                    
                    return {
                        'is_malicious': True,
                        'confidence_level': 'high',
                        'decision_layer': 3,
                        'threshold_used': high_confidence.get('risk_score', 0.6)
                    }
                
                return {
                    'is_malicious': True,
                    'confidence_level': 'medium',
                    'decision_layer': 2,
                    'threshold_used': detailed_analysis.get('risk_score', 0.35)
                }
        
        return {
            'is_malicious': False,
            'confidence_level': 'low',
            'decision_layer': 1,
            'threshold_used': quick_filter.get('risk_score', 0.2)
        }

class FalsePositiveSupressor:
    """误报抑制器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.suppression_config = config.get('detection', {}).get('false_positive_suppression', {})
        self.enabled = self.suppression_config.get('enabled', True)
        
        # 规则配置
        self.rule_config = self.suppression_config.get('rule_based_suppression', {})
        self.doc_rules = self.rule_config.get('document_level', {})
        self.detection_rules = self.rule_config.get('detection_level', {})
        
        logger.info(f"误报抑制器初始化，启用状态: {self.enabled}")
    
    def should_suppress(self, detection: Dict, context: Dict) -> Tuple[bool, str]:
        """判断是否应该抑制检测"""
        if not self.enabled:
            return False, ""
        
        detection_type = detection['type']
        
        # 文档级别抑制
        suppress, reason = self._check_document_level_suppression(detection, context)
        if suppress:
            return True, f"文档级别抑制: {reason}"
        
        # 检测级别抑制
        suppress, reason = self._check_detection_level_suppression(detection, context)
        if suppress:
            return True, f"检测级别抑制: {reason}"
        
        # 特定类型抑制
        suppress, reason = self._check_type_specific_suppression(detection, context)
        if suppress:
            return True, f"类型特定抑制: {reason}"
        
        return False, ""
    
    def _check_document_level_suppression(self, detection: Dict, context: Dict) -> Tuple[bool, str]:
        """文档级别抑制检查"""
        text_length = context.get('text_length', 0)
        detection_count = context.get('detection_count', 0)
        
        # 小文档过多检测抑制
        if text_length > 0 and text_length < 1000:
            detection_density = detection_count / (text_length / 100)
            if detection_density > 5:  # 每100字符超过5个检测
                return True, "小文档检测密度过高"
        
        # 小字体比例检查
        small_text_ratio = context.get('small_text_ratio', 0)
        max_ratio = self.doc_rules.get('max_small_text_ratio', 0.03)
        if small_text_ratio > max_ratio and detection['type'] == 'small_text_injection':
            return True, f"小字体比例过高: {small_text_ratio:.3f} > {max_ratio}"
        
        # 有意义内容比例检查
        if text_length > 0:
            meaningful_content = context.get('meaningful_content_length', text_length)
            meaningful_ratio = meaningful_content / text_length
            min_ratio = self.doc_rules.get('min_meaningful_content', 0.8)
            
            if meaningful_ratio < min_ratio:
                return True, f"有意义内容比例过低: {meaningful_ratio:.3f} < {min_ratio}"
        
        return False, ""
    
    def _check_detection_level_suppression(self, detection: Dict, context: Dict) -> Tuple[bool, str]:
        """检测级别抑制检查"""
        detection_type = detection['type']
        confidence = detection.get('confidence', 0.5)
        
        # 最小置信度检查
        min_confidence = self.detection_rules.get('min_keyword_confidence', 0.75)
        if detection_type.startswith('keyword') and confidence < min_confidence:
            return True, f"关键词置信度过低: {confidence:.3f} < {min_confidence}"
        
        # 上下文相关性检查
        if self.detection_rules.get('require_context_relevance', True):
            if detection_type == 'contextual_anomaly':
                avg_similarity = detection.get('avg_similarity', 1.0)
                if avg_similarity > 0.1:  # 不够异常
                    return True, f"上下文相关性过高: {avg_similarity:.3f}"
        
        return False, ""
    
    def _check_type_specific_suppression(self, detection: Dict, context: Dict) -> Tuple[bool, str]:
        """特定类型抑制检查"""
        detection_type = detection['type']
        specific_rules = self.detection_rules.get('specific_rules', {})
        
        if detection_type == 'small_text_injection':
            rules = specific_rules.get('small_text_injection', {})
            
            # 频次检查
            if 'occurrences' in detection:
                max_freq = rules.get('max_frequency', 5)
                if detection['occurrences'] > max_freq:
                    return True, f"小字体检测频次过高: {detection['occurrences']} > {max_freq}"
            
            # 可疑比例检查
            text_content = detection.get('content', '')
            if text_content:
                suspicious_chars = len([c for c in text_content if c.isalnum()])
                total_chars = len(text_content)
                if total_chars > 0:
                    suspicious_ratio = suspicious_chars / total_chars
                    min_ratio = rules.get('min_suspicious_ratio', 0.6)
                    if suspicious_ratio < min_ratio:
                        return True, f"可疑字符比例过低: {suspicious_ratio:.3f} < {min_ratio}"
            
            # 排除纯数字
            if rules.get('exclude_numbers', True) and text_content.isdigit():
                return True, "排除纯数字内容"
            
            # 排除纯标点
            if rules.get('exclude_punctuation', True) and all(not c.isalnum() for c in text_content):
                return True, "排除纯标点内容"
        
        elif detection_type == 'contextual_anomaly':
            rules = specific_rules.get('contextual_anomaly', {})
            
            # 最小句子长度
            sentence = detection.get('sentence', '')
            min_length = rules.get('min_sentence_length', 5)
            if len(sentence.split()) < min_length:
                return True, f"句子过短: {len(sentence.split())} < {min_length}"
            
            # 排除引用
            if rules.get('exclude_citations', True):
                if re.search(r'\[\d+\]|\(\d{4}\)', sentence):
                    return True, "疑似引用内容"
        
        return False, ""

class IntelligentWeightManager:
    """智能权重管理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.weight_config = config.get('detection', {}).get('intelligent_weights', {})
        self.enabled = self.weight_config.get('enabled', True)
        
        # 基础权重
        self.base_weights = self.weight_config.get('base_weights', {})
        self.current_weights = self.base_weights.copy()
        
        # 动态调整配置
        self.dynamic_config = self.weight_config.get('dynamic_adjustment', {})
        self.contextual_config = self.weight_config.get('contextual_weights', {})
        
        # 权重历史
        self.weight_history = deque(maxlen=100)
        
        logger.info(f"智能权重管理器初始化，启用状态: {self.enabled}")
    
    def get_weight(self, detection_type: str, context: Dict = None) -> float:
        """获取智能权重"""
        if not self.enabled:
            return self.base_weights.get(detection_type, 1.0)
        
        # 基础权重
        base_weight = self.current_weights.get(detection_type, 1.0)
        
        # 上下文调整
        if context:
            context_multiplier = self._calculate_context_multiplier(detection_type, context)
            return base_weight * context_multiplier
        
        return base_weight
    
    def _calculate_context_multiplier(self, detection_type: str, context: Dict) -> float:
        """计算上下文权重乘数"""
        multiplier = 1.0
        
        # 文档特征影响
        doc_features = self.contextual_config.get('document_features', {})
        
        text_length = context.get('text_length', 1000)
        if text_length < 500:
            multiplier *= doc_features.get('short_document', 0.8)
        elif text_length > 5000:
            multiplier *= doc_features.get('long_document', 1.2)
        
        # 扫描质量影响
        if context.get('is_scan_pdf', False):
            multiplier *= doc_features.get('scan_quality', 0.7)
        
        # 语言特征影响
        lang_features = self.contextual_config.get('language_features', {})
        detected_language = context.get('detected_language', 'english')
        
        if detected_language == 'chinese':
            multiplier *= lang_features.get('chinese_content', 1.1)
        elif detected_language == 'japanese':
            multiplier *= lang_features.get('japanese_content', 1.1)
        elif detected_language == 'mixed':
            multiplier *= lang_features.get('mixed_language', 1.2)
        
        # 检测密度影响
        detection_count = context.get('detection_count', 0)
        if detection_count > 5:
            multiplier *= 0.9  # 检测过多时降低权重
        
        return max(0.1, min(2.0, multiplier))  # 限制在合理范围
    
    def update_weights_based_on_performance(self, performance_data: Dict):
        """基于性能更新权重"""
        if not self.dynamic_config.get('enabled', False):
            return
        
        # 简单的权重学习逻辑
        type_performance = performance_data.get('detection_type_performance', {})
        
        for det_type, perf in type_performance.items():
            if det_type in self.current_weights:
                precision = perf.get('precision', 0.5)
                recall = perf.get('recall', 0.5)
                f1 = perf.get('f1_score', 0.5)
                
                # 基于F1分数调整权重
                if f1 < 0.5:
                    # 性能差，降低权重
                    adjustment = -0.05
                elif f1 > 0.8:
                    # 性能好，略微提高权重
                    adjustment = 0.02
                else:
                    adjustment = 0.0
                
                old_weight = self.current_weights[det_type]
                new_weight = max(0.1, min(2.0, old_weight + adjustment))
                self.current_weights[det_type] = new_weight
                
                if abs(adjustment) > 0.01:
                    logger.info(f"权重调整 {det_type}: {old_weight:.3f} -> {new_weight:.3f}")
        
        # 记录权重历史
        self.weight_history.append({
            'timestamp': time.time(),
            'weights': self.current_weights.copy(),
            'trigger': 'performance_update'
        })

class EnhancedPromptInjectionDetector:
    """增强的提示词注入检测器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.detection_config = config['detection']
        
        # 初始化各个管理器
        self.model_manager = ModelManager(config)
        self.threshold_manager = IntelligentThresholdManager(config)
        self.suppressor = FalsePositiveSupressor(config)
        self.weight_manager = IntelligentWeightManager(config)
        
        # 缓存管理
        self.cache_manager = CacheManager(config)
        
        # 性能监控
        self.performance_monitor = PerformanceMonitor(config)
        
        # 检测统计
        self.detection_stats = {
            'total_processed': 0,
            'total_detections': 0,
            'suppressed_detections': 0,
            'processing_times': deque(maxlen=100)
        }
        
        # 初始化其他组件
        self.suspicious_keywords = self.detection_config['suspicious_keywords']
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        logger.info("增强型提示词注入检测器初始化完成")
        
        # 获取模型信息
        model_info = self.model_manager.get_model_info()
        logger.info(f"模型状态: {model_info}")
    
    def extract_pdf_content(self, pdf_path: str) -> Dict[str, Any]:
        """提取PDF内容和格式信息 - 增强版"""
        start_time = time.time()
        
        # 检查缓存
        cache_key = f"pdf_content_{pdf_path}_{os.path.getmtime(pdf_path)}"
        cached_content = self.cache_manager.get(cache_key)
        if cached_content:
            logger.debug(f"使用缓存内容: {pdf_path}")
            return cached_content
        
        content = {
            'text': '',
            'metadata': {},
            'white_text': [],
            'small_text': [],
            'invisible_chars': [],
            'font_analysis': {},
            'page_count': 0,
            'file_size': 0,
            'extraction_method': 'unknown',
            'extraction_time': 0,
            'detected_language': 'unknown'
        }
        
        try:
            # 获取文件大小
            content['file_size'] = os.path.getsize(pdf_path)
            
            # 方法1: 优先使用 pdfplumber
            success = self._extract_with_pdfplumber_enhanced(pdf_path, content)
            
            # 方法2: 备用 PyMuPDF
            if not success or len(content['text']) < 100:
                self._extract_with_pymupdf_enhanced(pdf_path, content)
            
            # 后处理
            content = self._post_process_content_enhanced(content)
            
            # 缓存结果
            self.cache_manager.set(cache_key, content)
            
        except Exception as e:
            logger.error(f"PDF内容提取失败 {pdf_path}: {e}")
            content['error'] = str(e)
        
        content['extraction_time'] = time.time() - start_time
        return content
    
    def _extract_with_pdfplumber_enhanced(self, pdf_path: str, content: Dict) -> bool:
        """增强的pdfplumber提取"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                content['page_count'] = len(pdf.pages)
                content['extraction_method'] = 'pdfplumber'
                
                full_text = ""
                font_sizes = []
                font_colors = []
                
                for page_num, page in enumerate(pdf.pages):
                    try:
                        # 提取文本
                        page_text = page.extract_text() or ""
                        full_text += page_text + "\n"
                        
                        # 安全地分析字符格式
                        chars = getattr(page, 'chars', [])
                        for char in chars:
                            try:
                                self._safe_analyze_char_enhanced(char, content, font_sizes, font_colors)
                            except Exception:
                                continue
                                
                    except Exception as e:
                        logger.debug(f"页面 {page_num} 处理失败: {e}")
                        continue
                
                content['text'] = full_text
                
                # 字体统计
                if font_sizes:
                    content['font_analysis'] = self._safe_font_analysis_enhanced(font_sizes, font_colors)
                
                return len(full_text) > 50
            
        except Exception as e:
            logger.debug(f"pdfplumber 提取失败: {e}")
            return False
    
    def _safe_analyze_char_enhanced(self, char: Dict, content: Dict, 
                                   font_sizes: List, font_colors: List):
        """增强的字符分析"""
        try:
            # 字体大小分析
            size = char.get('size', 12)
            if isinstance(size, (int, float)) and size > 0:
                font_sizes.append(float(size))
                
                # 更智能的小字体检测
                threshold = self.threshold_manager.get_threshold('small_font_size', 
                                                               {'current_size': size})
                if size < threshold:
                    text_char = char.get('text', '')
                    if text_char and text_char.strip() and not text_char.isspace():
                        content['small_text'].append(text_char)
            
            # 更安全的颜色分析
            color = char.get('non_stroking_color') or char.get('color')
            if color is not None:
                try:
                    if self._is_white_color_enhanced(color):
                        text_char = char.get('text', '')
                        if text_char and text_char.strip():
                            # 添加上下文信息
                            char_info = {
                                'text': text_char,
                                'size': size,
                                'color': color
                            }
                            content['white_text'].append(char_info)
                    
                    # 记录颜色统计
                    if isinstance(color, (list, tuple)) and len(color) >= 3:
                        font_colors.append(color)
                        
                except Exception:
                    pass
                    
        except Exception:
            pass
    
    def _is_white_color_enhanced(self, color) -> bool:
        """增强的白色检测"""
        try:
            if color is None:
                return False
            
            # 动态阈值
            white_threshold = self.threshold_manager.get_threshold('white_text_threshold')
            
            if isinstance(color, (int, float)):
                return float(color) > white_threshold
            elif isinstance(color, (list, tuple)):
                if len(color) >= 3:
                    return all(float(c) > white_threshold for c in color[:3])
                elif len(color) == 1:
                    return float(color[0]) > white_threshold
            elif isinstance(color, str):
                if color.startswith('#'):
                    try:
                        rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                        threshold_255 = int(white_threshold * 255)
                        return all(c > threshold_255 for c in rgb)
                    except Exception:
                        return False
            
            return False
            
        except Exception:
            return False
    
    def _post_process_content_enhanced(self, content: Dict) -> Dict:
        """增强的内容后处理"""
        try:
            text = content.get('text', '')
            
            # 语言检测
            if text:
                content['detected_language'] = detect_language(text)
                content['invisible_chars'] = self._detect_invisible_chars_enhanced(text)
            
            # 清理和去重
            content['small_text'] = self._clean_text_list(content.get('small_text', []))
            content['white_text'] = self._clean_white_text_list(content.get('white_text', []))
            
            # 统计信息
            content['content_statistics'] = {
                'text_length': len(text),
                'unique_small_text_count': len(set(content['small_text'])),
                'unique_white_text_count': len(set(str(item) for item in content['white_text'])),
                'invisible_char_count': len(content['invisible_chars']),
                'small_text_ratio': len(''.join(content['small_text'])) / max(len(text), 1)
            }
            
        except Exception as e:
            logger.debug(f"内容后处理失败: {e}")
        
        return content
    
    def _clean_text_list(self, text_list: List[str], max_items: int = 100) -> List[str]:
        """清理文本列表"""
        if not text_list:
            return []
        
        # 去重并过滤
        cleaned = []
        seen = set()
        
        for item in text_list:
            if isinstance(item, str):
                item = item.strip()
                if (item and 
                    item not in seen and 
                    len(item) > 0 and 
                    not item.isspace()):
                    cleaned.append(item)
                    seen.add(item)
                    
                    if len(cleaned) >= max_items:
                        break
        
        return cleaned
    
    def _clean_white_text_list(self, white_text_list: List, max_items: int = 100) -> List[str]:
        """清理白色文本列表"""
        if not white_text_list:
            return []
        
        cleaned = []
        seen = set()
        
        for item in white_text_list:
            # 处理不同格式
            if isinstance(item, dict):
                text = item.get('text', '')
            elif isinstance(item, str):
                text = item
            else:
                continue
            
            text = text.strip()
            if (text and 
                text not in seen and 
                len(text) > 0 and 
                not text.isspace()):
                cleaned.append(text)
                seen.add(text)
                
                if len(cleaned) >= max_items:
                    break
        
        return cleaned
    
    def detect_injection_enhanced(self, pdf_path: str) -> Dict[str, Any]:
        """增强的注入检测"""
        start_time = time.time()
        
        logger.info(f"开始增强检测: {pdf_path}")
        
        # 提取内容
        content = self.extract_pdf_content(pdf_path)
        
        if not content['text'] and not content['metadata']:
            logger.warning(f"无法提取PDF内容: {pdf_path}")
            return self._create_error_result(pdf_path, 'Content extraction failed')
        
        # 创建检测上下文
        context = self._create_detection_context(content)
        
        # 执行检测
        all_detections = []
        detection_errors = []
        
        detection_methods = [
            ('keyword', self.detect_keyword_injection_enhanced),
            ('semantic', self.detect_semantic_injection_enhanced),
            ('format', self.detect_format_injection_enhanced),
            ('encoding', self.detect_encoding_injection_enhanced),
            ('multilingual', self.detect_multilingual_injection_enhanced),
            ('contextual', self.detect_contextual_anomalies_enhanced)
        ]
        
        for method_name, method_func in detection_methods:
            try:
                detections = method_func(content['text'], context)
                # 应用误报抑制
                filtered_detections = self._apply_false_positive_suppression(detections, context)
                all_detections.extend(filtered_detections)
                
            except Exception as e:
                error_msg = f"{method_name}检测失败: {e}"
                detection_errors.append(error_msg)
                logger.warning(error_msg)
        
        # 计算增强风险分数
        risk_score = self.calculate_enhanced_risk_score(all_detections, context)
        
        # 智能决策
        decision = self.threshold_manager.get_layered_decision(
            risk_score, 
            len(all_detections),
            sum(d.get('confidence', 0.5) for d in all_detections),
            set(d['type'] for d in all_detections)
        )
        
        # 构建结果
        result = {
            'file': pdf_path,
            'detections': all_detections,
            'detection_count': len(all_detections),
            'suppressed_count': context.get('suppressed_count', 0),
            'risk_score': risk_score,
            'is_malicious': decision['is_malicious'],
            'confidence_level': decision['confidence_level'],
            'decision_layer': decision['decision_layer'],
            'threshold_used': decision['threshold_used'],
            'content_stats': context['content_stats'],
            'model_info': self.model_manager.get_model_info(),
            'detection_errors': detection_errors,
            'processing_time': time.time() - start_time
        }
        
        # 更新统计
        self._update_detection_stats(result)
        
        logger.info(f"增强检测完成: {pdf_path}, 风险分数: {risk_score:.3f}, "
                   f"检测数: {len(all_detections)}, 决策: {decision['is_malicious']} "
                   f"(层级{decision['decision_layer']})")
        
        return result
    
    def _create_detection_context(self, content: Dict) -> Dict:
        """创建检测上下文"""
        text = content.get('text', '')
        content_stats = content.get('content_statistics', {})
        
        context = {
            'text_length': len(text),
            'page_count': content.get('page_count', 0),
            'file_size': content.get('file_size', 0),
            'detected_language': content.get('detected_language', 'unknown'),
            'extraction_method': content.get('extraction_method', 'unknown'),
            'small_text_ratio': content_stats.get('small_text_ratio', 0),
            'content_stats': content_stats,
            'is_scan_pdf': self._is_scan_pdf(content),
            'suppressed_count': 0
        }
        
        return context
    
    def _is_scan_pdf(self, content: Dict) -> bool:
        """判断是否为扫描PDF"""
        # 简单启发式判断
        text_length = len(content.get('text', ''))
        file_size = content.get('file_size', 0)
        page_count = content.get('page_count', 1)
        
        if page_count == 0:
            return True
        
        # 平均每页文本很少可能是扫描文档
        avg_text_per_page = text_length / page_count
        if avg_text_per_page < 100:
            return True
        
        # 文件大小相对于文本长度过大
        if file_size > 0 and text_length > 0:
            size_text_ratio = file_size / text_length
            if size_text_ratio > 1000:  # 每字符超过1KB
                return True
        
        return False
    
    def _apply_false_positive_suppression(self, detections: List[Dict], 
                                        context: Dict) -> List[Dict]:
        """应用误报抑制"""
        if not detections:
            return detections
        
        filtered_detections = []
        suppressed_count = 0
        
        for detection in detections:
            should_suppress, reason = self.suppressor.should_suppress(detection, context)
            
            if should_suppress:
                suppressed_count += 1
                logger.debug(f"抑制检测 {detection['type']}: {reason}")
            else:
                filtered_detections.append(detection)
        
        context['suppressed_count'] = suppressed_count
        
        if suppressed_count > 0:
            logger.info(f"抑制了 {suppressed_count} 个可能的误报")
        
        return filtered_detections
    
    def calculate_enhanced_risk_score(self, detections: List[Dict], 
                                    context: Dict) -> float:
        """计算增强风险分数"""
        if not detections:
            return 0.0
        
        weighted_score = 0.0
        total_weight = 0.0
        confidence_sum = 0.0
        
        for detection in detections:
            detection_type = detection['type']
            confidence = detection.get('confidence', 0.5)
            
            # 获取智能权重
            weight = self.weight_manager.get_weight(detection_type, context)
            
            # 累计分数
            weighted_score += confidence * weight
            total_weight += weight
            confidence_sum += confidence
        
        if total_weight == 0:
            return 0.0
        
        # 基础分数
        base_score = weighted_score / total_weight
        
        # 上下文调整
        context_adjustment = self._calculate_context_score_adjustment(detections, context)
        
        # 检测多样性奖励（减少）
        unique_types = len(set(d['type'] for d in detections))
        diversity_bonus = min(0.05, unique_types * 0.01)  # 最多5%奖励
        
        # 最终分数
        final_score = min(1.0, base_score + context_adjustment + diversity_bonus)
        
        return final_score
    
    def _calculate_context_score_adjustment(self, detections: List[Dict], 
                                          context: Dict) -> float:
        """计算上下文分数调整"""
        adjustment = 0.0
        
        text_length = context.get('text_length', 1000)
        detection_count = len(detections)
        
        # 检测密度调整
        if text_length > 0:
            detection_density = detection_count / (text_length / 1000)
            
            # 密度过高可能是误报
            if detection_density > 2.0:
                density_penalty = min(0.2, (detection_density - 2.0) * 0.05)
                adjustment -= density_penalty
        
        # 小文档惩罚
        if text_length < 1000:
            adjustment -= 0.03
        
        # 扫描PDF调整
        if context.get('is_scan_pdf', False):
            adjustment -= 0.05  # 扫描文档更容易误报
        
        # 语言特定调整
        detected_language = context.get('detected_language', 'english')
        if detected_language in ['chinese', 'japanese', 'mixed']:
            adjustment += 0.02  # 非英文内容稍微提高风险
        
        return max(-0.3, min(0.2, adjustment))  # 限制调整范围
    
    def detect_keyword_injection_enhanced(self, text: str, context: Dict) -> List[Dict]:
        """增强的关键词检测"""
        detections = []
        
        if not text:
            return detections
        
        text_lower = text.lower()
        detected_language = context.get('detected_language', 'english')
        
        # 选择合适的关键词集合
        if detected_language in self.suspicious_keywords:
            keyword_sets = [detected_language]
        else:
            keyword_sets = list(self.suspicious_keywords.keys())
        
        for lang in keyword_sets:
            keywords = self.suspicious_keywords.get(lang, [])
            
            for keyword in keywords:
                keyword_lower = keyword.lower()
                
                # 精确匹配
                occurrences = len(re.findall(re.escape(keyword_lower), text_lower))
                if occurrences > 0:
                    # 上下文相关性检查
                    contexts = self._extract_keyword_contexts(text, keyword, max_contexts=3)
                    
                    # 计算置信度
                    base_confidence = min(0.95, 0.7 + occurrences * 0.05)
                    context_boost = self._calculate_keyword_context_boost(contexts)
                    final_confidence = min(0.98, base_confidence + context_boost)
                    
                    # 智能阈值检查
                    threshold = self.threshold_manager.get_threshold('keyword_match', context)
                    if final_confidence > threshold:
                        detection = {
                            'type': 'keyword_injection',
                            'language': lang,
                            'keyword': keyword,
                            'occurrences': occurrences,
                            'confidence': final_confidence,
                            'contexts': contexts[:2]  # 保存前2个上下文
                        }
                        detections.append(detection)
                
                # 模糊匹配（更智能）
                if lang == detected_language:  # 只对检测到的语言进行模糊匹配
                    fuzzy_matches = self._enhanced_fuzzy_keyword_match(text_lower, keyword_lower)
                    for match in fuzzy_matches:
                        detection = {
                            'type': 'keyword_injection_fuzzy',
                            'language': lang,
                            'keyword': keyword,
                            'matched_text': match['text'],
                            'similarity': match['similarity'],
                            'confidence': match['confidence']
                        }
                        detections.append(detection)
        
        return detections
    
    def _extract_keyword_contexts(self, text: str, keyword: str, 
                                window_size: int = 50, max_contexts: int = 3) -> List[str]:
        """提取关键词上下文"""
        contexts = []
        keyword_lower = keyword.lower()
        text_lower = text.lower()
        
        start = 0
        while len(contexts) < max_contexts:
            pos = text_lower.find(keyword_lower, start)
            if pos == -1:
                break
            
            # 提取上下文
            context_start = max(0, pos - window_size)
            context_end = min(len(text), pos + len(keyword) + window_size)
            context = text[context_start:context_end].strip()
            
            if context and context not in contexts:
                contexts.append(context)
            
            start = pos + 1
        
        return contexts
    
    def _calculate_keyword_context_boost(self, contexts: List[str]) -> float:
        """计算关键词上下文提升"""
        if not contexts:
            return 0.0
        
        boost = 0.0
        review_indicators = [
            'review', 'paper', 'manuscript', 'publication', 'journal',
            'conference', 'peer', 'editor', 'reviewer', 'submission'
        ]
        
        for context in contexts:
            context_lower = context.lower()
            
            # 检查审稿相关词汇
            review_count = sum(1 for indicator in review_indicators 
                             if indicator in context_lower)
            if review_count > 0:
                boost += min(0.1, review_count * 0.03)
            
            # 检查句子完整性
            if '.' in context and len(context.split()) > 5:
                boost += 0.02
        
        return min(0.2, boost)
    
    def _enhanced_fuzzy_keyword_match(self, text: str, keyword: str, 
                                    threshold: float = 0.85) -> List[Dict]:
        """增强的模糊匹配"""
        matches = []
        words = re.findall(r'\b\w+\b', text)
        
        for word in words:
            if len(word) < 3 or len(keyword) < 3:
                continue
            
            similarity = self._calculate_enhanced_similarity(word, keyword)
            if similarity > threshold:
                confidence = similarity * 0.8  # 模糊匹配降低置信度
                matches.append({
                    'text': word,
                    'similarity': similarity,
                    'confidence': confidence
                })
        
        # 限制返回数量并按相似度排序
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches[:5]
    
    def _calculate_enhanced_similarity(self, str1: str, str2: str) -> float:
        """增强的相似度计算"""
        from difflib import SequenceMatcher
        
        # 基础相似度
        base_similarity = SequenceMatcher(None, str1, str2).ratio()
        
        # 长度惩罚（长度差异过大降低相似度）
        len_diff = abs(len(str1) - len(str2))
        max_len = max(len(str1), len(str2))
        if max_len > 0:
            len_penalty = len_diff / max_len
            base_similarity *= (1 - len_penalty * 0.3)
        
        return base_similarity
    
    def detect_semantic_injection_enhanced(self, text: str, context: Dict) -> List[Dict]:
        """增强的语义检测"""
        detections = []
        
        if not self.model_manager.sentiment_analyzer:
            # 使用备用语义检测
            return self._fallback_semantic_detection(text, context)
        
        # 分句处理
        sentences = self._split_sentences_enhanced(text)
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            try:
                # 情感分析（限制长度）
                truncated_sentence = sentence[:512]
                results = self.model_manager.sentiment_analyzer(truncated_sentence)
                
                # 处理不同模型的输出格式
                if isinstance(results, list) and len(results) > 0:
                    if isinstance(results[0], dict):
                        # 标准transformers输出
                        for result in results:
                            if (result.get('label') == 'POSITIVE' and 
                                result.get('score', 0) > self.threshold_manager.get_threshold('sentiment_confidence', context)):
                                
                                # 增强的审稿相关性检查
                                relevance_score = self._calculate_review_relevance(sentence)
                                
                                if relevance_score > 0.3:
                                    final_confidence = min(0.95, result['score'] * (1 + relevance_score))
                                    
                                    detection = {
                                        'type': 'semantic_injection',
                                        'sentence': sentence[:200],
                                        'sentence_index': i,
                                        'sentiment_label': result['label'],
                                        'sentiment_score': result['score'],
                                        'review_relevance': relevance_score,
                                        'confidence': final_confidence,
                                        'model_type': 'transformer'
                                    }
                                    detections.append(detection)
                    else:
                        # 备用模型输出
                        result = results[0] if isinstance(results, list) else results
                        if (result.get('label') == 'POSITIVE' and 
                            result.get('score', 0) > 0.7):
                            
                            relevance_score = self._calculate_review_relevance(sentence)
                            if relevance_score > 0.3:
                                detection = {
                                    'type': 'semantic_injection',
                                    'sentence': sentence[:200],
                                    'sentence_index': i,
                                    'sentiment_label': result['label'],
                                    'sentiment_score': result['score'],
                                    'review_relevance': relevance_score,
                                    'confidence': min(0.9, result['score'] * 0.9),
                                    'model_type': 'fallback'
                                }
                                detections.append(detection)
                
            except Exception as e:
                logger.debug(f"句子语义分析失败: {e}")
                continue
        
        return detections
    
    def _fallback_semantic_detection(self, text: str, context: Dict) -> List[Dict]:
        """备用语义检测"""
        detections = []
        
        # 基于规则的情感检测
        sentences = self._split_sentences_enhanced(text)
        
        positive_patterns = [
            r'\b(excellent|outstanding|remarkable|exceptional|superb|brilliant)\b',
            r'\b(highly\s+recommend|strongly\s+recommend|definitely\s+accept)\b',
            r'\b(top\s+quality|high\s+quality|superior\s+quality|best\s+quality)\b',
            r'\b(groundbreaking|innovative|novel|cutting-edge|revolutionary)\b',
            r'\b(accept|approve|recommend|endorse)\s+(immediately|without\s+question|strongly)\b'
        ]
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            # 计算积极模式匹配分数
            positive_score = 0
            matched_patterns = []
            
            for pattern in positive_patterns:
                matches = re.findall(pattern, sentence_lower)
                if matches:
                    positive_score += len(matches) * 0.3
                    matched_patterns.extend(matches)
            
            # 检查审稿相关性
            relevance_score = self._calculate_review_relevance(sentence)
            
            if positive_score > 0.6 and relevance_score > 0.3:
                final_confidence = min(0.85, (positive_score + relevance_score) * 0.8)
                
                detection = {
                    'type': 'semantic_injection',
                    'sentence': sentence[:200],
                    'sentence_index': i,
                    'sentiment_label': 'POSITIVE',
                    'sentiment_score': positive_score,
                    'review_relevance': relevance_score,
                    'confidence': final_confidence,
                    'model_type': 'rule_based',
                    'matched_patterns': matched_patterns[:3]
                }
                detections.append(detection)
        
        return detections
    
    def _calculate_review_relevance(self, sentence: str) -> float:
        """计算审稿相关性分数"""
        sentence_lower = sentence.lower()
        
        # 审稿相关词汇权重
        review_terms = {
            'review': 0.3, 'paper': 0.2, 'manuscript': 0.3, 'publication': 0.25,
            'journal': 0.2, 'conference': 0.2, 'peer': 0.15, 'editor': 0.2,
            'reviewer': 0.3, 'submission': 0.25, 'accept': 0.3, 'reject': 0.3,
            'recommend': 0.25, 'approve': 0.25, 'rating': 0.2, 'evaluation': 0.2,
            'assessment': 0.2, 'feedback': 0.15, 'comment': 0.1, 'quality': 0.15
        }
        
        relevance_score = 0.0
        for term, weight in review_terms.items():
            if term in sentence_lower:
                relevance_score += weight
        
        # 学术短语
        academic_phrases = [
            'peer review', 'review process', 'manuscript evaluation',
            'publication decision', 'editorial decision', 'review comments',
            'reviewer feedback', 'accept for publication', 'recommend acceptance'
        ]
        
        for phrase in academic_phrases:
            if phrase in sentence_lower:
                relevance_score += 0.4
        
        return min(1.0, relevance_score)
    
    def _split_sentences_enhanced(self, text: str) -> List[str]:
        """增强的分句方法"""
        # 更智能的分句，考虑不同语言
        sentences = []
        
        # 基于标点的分句
        basic_sentences = re.split(r'[.!?。！？]+', text)
        
        for sentence in basic_sentences:
            sentence = sentence.strip()
            if len(sentence) < 5:
                continue
            
            # 进一步分割过长的句子
            if len(sentence) > 500:
                sub_sentences = re.split(r'[;,，；]+', sentence)
                for sub_sentence in sub_sentences:
                    sub_sentence = sub_sentence.strip()
                    if len(sub_sentence) > 10:
                        sentences.append(sub_sentence)
            else:
                sentences.append(sentence)
        
        return sentences
    
    def detect_format_injection_enhanced(self, content: Dict, context: Dict) -> List[Dict]:
        """增强的格式检测"""
        detections = []
        
        # 智能白色文本检测
        white_text_detections = self._detect_white_text_enhanced(content, context)
        detections.extend(white_text_detections)
        
        # 智能小字体检测
        small_text_detections = self._detect_small_text_enhanced(content, context)
        detections.extend(small_text_detections)
        
        # 元数据检测
        metadata_detections = self._detect_metadata_injection_enhanced(content, context)
        detections.extend(metadata_detections)
        
        # 不可见字符检测
        invisible_detections = self._detect_invisible_chars_enhanced(content, context)
        detections.extend(invisible_detections)
        
        return detections
    
    def _detect_white_text_enhanced(self, content: Dict, context: Dict) -> List[Dict]:
        """增强的白色文本检测"""
        detections = []
        white_text_list = content.get('white_text', [])
        
        if not white_text_list:
            return detections
        
        # 合并白色文本
        if isinstance(white_text_list[0], dict):
            white_text = ''.join([item.get('text', '') for item in white_text_list])
        else:
            white_text = ''.join(white_text_list)
        
        white_text = white_text.strip()
        
        if len(white_text) > 15:  # 提高最小长度要求
            # 检查可疑性
            suspicion_score = self._calculate_white_text_suspicion(white_text, context)
            
            if suspicion_score > 0.5:
                # 动态置信度计算
                base_confidence = min(0.9, suspicion_score)
                length_bonus = min(0.05, len(white_text) / 1000)
                final_confidence = min(0.95, base_confidence + length_bonus)
                
                detection = {
                    'type': 'white_text_injection',
                    'content': white_text[:300],
                    'length': len(white_text),
                    'suspicion_score': suspicion_score,
                    'confidence': final_confidence,
                    'char_count': len(white_text_list)
                }
                detections.append(detection)
        
        return detections
    
    def _calculate_white_text_suspicion(self, text: str, context: Dict) -> float:
        """计算白色文本可疑度"""
        suspicion = 0.0
        
        # 包含可疑关键词
        if self._contains_suspicious_keywords_enhanced(text):
            suspicion += 0.6
        
        # 文本长度
        if len(text) > 50:
            suspicion += 0.3
        
        # 包含完整句子
        if '.' in text and len(text.split()) > 3:
            suspicion += 0.2
        
        # 语言一致性
        detected_language = context.get('detected_language', 'english')
        text_language = detect_language(text)
        if text_language == detected_language:
            suspicion += 0.1
        
        # 排除可能的格式化内容
        if self._is_likely_formatting_artifact_enhanced(text):
            suspicion -= 0.4
        
        return max(0.0, min(1.0, suspicion))
    
    def _contains_suspicious_keywords_enhanced(self, text: str) -> bool:
        """增强的可疑关键词检查"""
        text_lower = text.lower()
        
        # 检查所有语言的关键词
        for lang_keywords in self.suspicious_keywords.values():
            for keyword in lang_keywords:
                if keyword.lower() in text_lower:
                    return True
        
        # 检查模糊匹配关键词
        fuzzy_keywords = self.detection_config.get('suspicious_keywords', {}).get('fuzzy_keywords', {})
        for lang, patterns in fuzzy_keywords.items():
            for pattern in patterns:
                if '*' in pattern:
                    # 简单的通配符匹配
                    regex_pattern = pattern.replace('*', '.*')
                    if re.search(regex_pattern, text_lower):
                        return True
        
        return False
    
    def _is_likely_formatting_artifact_enhanced(self, text: str) -> bool:
        """增强的格式化伪影判断"""
        # 原有的模式
        formatting_patterns = [
            r'^\s*\d+\s*$',  # 纯数字
            r'^\s*[a-zA-Z]\s*$',  # 单字母
            r'^\s*[.]{2,}\s*$',  # 点号序列
            r'^\s*[-_=]{2,}\s*$',  # 分隔符
            r'^\s*\([^)]*\)\s*$',  # 括号内容
            r'^\s*[IVXLCDMivxlcdm]+\s*$',  # 罗马数字
            r'^\s*\w{1,3}\.\s*$',  # 简短编号
            r'^\s*page\s+\d+\s*$',  # 页码
            r'^\s*\d+\s*/\s*\d+\s*$',  # 分数格式
        ]
        
        for pattern in formatting_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        # 检查是否主要由非字母数字字符组成
        if len(text) > 0:
            alnum_ratio = len([c for c in text if c.isalnum()]) / len(text)
            if alnum_ratio < 0.3:  # 少于30%的字母数字字符
                return True
        
        # 检查重复字符
        if len(set(text.lower())) < len(text) * 0.3:  # 唯一字符太少
            return True
        
        return False
    
    def _detect_small_text_enhanced(self, content: Dict, context: Dict) -> List[Dict]:
        """增强的小字体检测"""
        detections = []
        small_text_list = content.get('small_text', [])
        
        if not small_text_list:
            return detections
        
        small_text = ''.join(small_text_list).strip()
        
        if len(small_text) > 10:
            # 计算可疑程度
            suspicion_score = self._calculate_small_text_suspicion_enhanced(small_text, context)
            
            # 动态阈值
            threshold = self.threshold_manager.get_threshold('small_text_suspicion', context)
            
            if suspicion_score > threshold:
                detection = {
                    'type': 'small_text_injection',
                    'content': small_text[:200],
                    'length': len(small_text),
                    'char_count': len(small_text_list),
                    'suspicion_score': suspicion_score,
                    'confidence': min(0.85, suspicion_score * 0.9)
                }
                detections.append(detection)
        
        return detections
    
    def _calculate_small_text_suspicion_enhanced(self, text: str, context: Dict) -> float:
        """增强的小字体可疑度计算"""
        suspicion = 0.0
        
        # 基础检查
        if self._contains_suspicious_keywords_enhanced(text):
            suspicion += 0.7
        
        # 文本特征
        if len(text) > 50:
            suspicion += 0.2
        
        if '.' in text and len(text.split()) > 3:
            suspicion += 0.2
        
        # 与主文档语言的一致性
        detected_language = context.get('detected_language', 'english')
        text_language = detect_language(text)
        if text_language == detected_language or text_language == 'mixed':
            suspicion += 0.15
        
        # 文档上下文调整
        text_length = context.get('text_length', 1000)
        if text_length > 0:
            small_text_ratio = len(text) / text_length
            if small_text_ratio > 0.01:  # 超过1%的文档内容
                suspicion += min(0.3, small_text_ratio * 20)
        
        # 排除可能的正常内容
        if self._is_likely_formatting_artifact_enhanced(text):
            suspicion -= 0.5
        
        # 检查数字和标点比例
        if len(text) > 0:
            digit_ratio = len([c for c in text if c.isdigit()]) / len(text)
            punct_ratio = len([c for c in text if c in '.,;:!?()[]{}"\'-']) / len(text)
            
            if digit_ratio > 0.7:  # 主要是数字
                suspicion -= 0.3
            if punct_ratio > 0.5:  # 主要是标点
                suspicion -= 0.3
        
        return max(0.0, min(1.0, suspicion))
    
    def _detect_metadata_injection_enhanced(self, content: Dict, context: Dict) -> List[Dict]:
        """增强的元数据检测"""
        detections = []
        metadata = content.get('metadata', {})
        
        if not metadata:
            return detections
        
        # 检查各个元数据字段
        for field, value in metadata.items():
            if isinstance(value, str) and value.strip():
                if self._contains_suspicious_keywords_enhanced(value):
                    # 计算置信度
                    confidence = 0.8
                    
                    # 字段重要性调整
                    if field in ['subject', 'keywords', 'title']:
                        confidence += 0.1
                    elif field in ['creator', 'producer']:
                        confidence -= 0.1
                    
                    detection = {
                        'type': 'metadata_injection',
                        'field': field,
                        'content': value[:200],
                        'confidence': min(0.95, confidence)
                    }
                    detections.append(detection)
        
        return detections
    
    def _detect_invisible_chars_enhanced(self, content: Dict, context: Dict) -> List[Dict]:
        """增强的不可见字符检测"""
        detections = []
        invisible_chars = content.get('invisible_chars', [])
        
        if not invisible_chars:
            return detections
        
        total_invisible = sum(len(chars) for chars in invisible_chars)
        
        if total_invisible > 20:  # 降低阈值，更敏感
            # 分析不可见字符类型
            char_types = self._analyze_invisible_char_types(invisible_chars)
            
            # 计算风险分数
            risk_score = min(1.0, total_invisible / 100)
            
            # 类型多样性奖励
            if len(char_types) > 1:
                risk_score += 0.2
            
            detection = {
                'type': 'invisible_chars_injection',
                'count': total_invisible,
                'char_types': list(char_types.keys()),
                'samples': invisible_chars[:3],
                'confidence': min(0.95, risk_score)
            }
            detections.append(detection)
        
        return detections
    
    def _analyze_invisible_char_types(self, invisible_chars: List[str]) -> Dict[str, int]:
        """分析不可见字符类型"""
        char_types = defaultdict(int)
        
        for chars in invisible_chars:
            for char in chars:
                code = ord(char)
                
                if code in [0x200b, 0x200c, 0x200d, 0xfeff]:
                    char_types['zero_width'] += 1
                elif code in [0x00a0, 0x2007, 0x202f]:
                    char_types['non_breaking_space'] += 1
                elif 0x2000 <= code <= 0x200f:
                    char_types['unicode_space'] += 1
                elif 0x0300 <= code <= 0x036f:
                    char_types['combining_marks'] += 1
                else:
                    char_types['other_invisible'] += 1
        
        return dict(char_types)
    
    def detect_encoding_injection_enhanced(self, text: str, context: Dict) -> List[Dict]:
        """增强的编码检测"""
        detections = []
        
        # Base64检测
        base64_detections = self._detect_base64_enhanced(text, context)
        detections.extend(base64_detections)
        
        # URL编码检测
        url_detections = self._detect_url_encoding_enhanced(text, context)
        detections.extend(url_detections)
        
        # 十六进制编码检测
        hex_detections = self._detect_hex_encoding_enhanced(text, context)
        detections.extend(hex_detections)
        
        return detections
    
    def _detect_base64_enhanced(self, text: str, context: Dict) -> List[Dict]:
        """增强的Base64检测"""
        detections = []
        
        # 更严格的Base64模式
        base64_pattern = r'(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?'
        matches = re.findall(base64_pattern, text)
        
        for match in matches:
            if len(match) < 20:  # 跳过太短的匹配
                continue
            
            try:
                # 验证是否为有效的Base64
                decoded_bytes = base64.b64decode(match, validate=True)
                decoded = decoded_bytes.decode('utf-8', errors='ignore')
                
                if self._contains_suspicious_keywords_enhanced(decoded):
                    # 计算置信度
                    confidence = 0.85
                    if len(decoded) > 50:
                        confidence += 0.05
                    
                    detection = {
                        'type': 'base64_injection',
                        'encoded': match[:100],
                        'decoded': decoded[:150],
                        'confidence': min(0.95, confidence)
                    }
                    detections.append(detection)
                    
            except Exception:
                continue
        
        return detections
    
    def _detect_url_encoding_enhanced(self, text: str, context: Dict) -> List[Dict]:
        """增强的URL编码检测"""
        detections = []
        
        # 检查URL编码模式
        url_pattern = r'%[0-9A-Fa-f]{2}'
        if re.search(url_pattern, text):
            try:
                import urllib.parse
                decoded = urllib.parse.unquote(text)
                
                if (decoded != text and 
                    len(decoded) > len(text) * 0.8 and  # 解码后长度合理
                    self._contains_suspicious_keywords_enhanced(decoded)):
                    
                    detection = {
                        'type': 'url_encoding_injection',
                        'original': text[:150],
                        'decoded': decoded[:150],
                        'confidence': 0.8
                    }
                    detections.append(detection)
                    
            except Exception:
                pass
        
        return detections
    
    def _detect_hex_encoding_enhanced(self, text: str, context: Dict) -> List[Dict]:
        """增强的十六进制编码检测"""
        detections = []
        
        # 检查十六进制模式
        hex_pattern = r'(?:0x)?[0-9A-Fa-f]{20,}'
        matches = re.findall(hex_pattern, text)
        
        for match in matches:
            try:
                # 移除0x前缀
                hex_str = match.replace('0x', '')
                if len(hex_str) % 2 != 0:
                    continue
                
                # 解码
                decoded_bytes = bytes.fromhex(hex_str)
                decoded = decoded_bytes.decode('utf-8', errors='ignore')
                
                if self._contains_suspicious_keywords_enhanced(decoded):
                    detection = {
                        'type': 'hex_encoding_injection',
                        'encoded': match[:100],
                        'decoded': decoded[:150],
                        'confidence': 0.75
                    }
                    detections.append(detection)
                    
            except Exception:
                continue
        
        return detections
    
    def detect_multilingual_injection_enhanced(self, text: str, context: Dict) -> List[Dict]:
        """增强的多语言检测"""
        detections = []
        
        # 分析语言分布
        language_dist = self._analyze_language_distribution_enhanced(text)
        
        # 只有在真正混合语言时才进行检测
        if len(language_dist) > 2:
            sentences = self._split_sentences_enhanced(text)
            
            for i, sentence in enumerate(sentences):
                if len(sentence.split()) < 3:  # 跳过太短的句子
                    continue
                
                sentence_lang = detect_language(sentence)
                
                # 检查句子是否包含可疑内容
                if (sentence_lang in self.suspicious_keywords and
                    self._contains_suspicious_keywords_enhanced(sentence)):
                    
                    # 计算语言异常分数
                    lang_anomaly_score = self._calculate_language_anomaly(
                        sentence_lang, language_dist, context
                    )
                    
                    if lang_anomaly_score > 0.3:
                        confidence = min(0.85, 0.6 + lang_anomaly_score * 0.3)
                        
                        detection = {
                            'type': 'multilingual_injection',
                            'sentence': sentence[:150],
                            'sentence_index': i,
                            'detected_language': sentence_lang,
                            'language_distribution': language_dist,
                            'anomaly_score': lang_anomaly_score,
                            'confidence': confidence
                        }
                        detections.append(detection)
        
        return detections
    
    def _analyze_language_distribution_enhanced(self, text: str) -> Dict[str, float]:
        """增强的语言分布分析"""
        if not text:
            return {}
        
        total_chars = len(text)
        
        # 更详细的语言字符统计
        char_counts = {
            'chinese': len(re.findall(r'[\u4e00-\u9fff]', text)),
            'japanese_hiragana': len(re.findall(r'[\u3040-\u309f]', text)),
            'japanese_katakana': len(re.findall(r'[\u30a0-\u30ff]', text)),
            'korean': len(re.findall(r'[\uac00-\ud7af]', text)),
            'english': len(re.findall(r'[a-zA-Z]', text)),
            'arabic': len(re.findall(r'[\u0600-\u06ff]', text)),
            'cyrillic': len(re.findall(r'[\u0400-\u04ff]', text))
        }
        
        # 合并日文字符
        char_counts['japanese'] = char_counts['japanese_hiragana'] + char_counts['japanese_katakana']
        del char_counts['japanese_hiragana']
        del char_counts['japanese_katakana']
        
        # 计算比例
        distribution = {}
        for lang, count in char_counts.items():
            if count > 0:
                distribution[lang] = count / total_chars
        
        return distribution
    
    def _calculate_language_anomaly(self, sentence_lang: str, 
                                  overall_dist: Dict[str, float], 
                                  context: Dict) -> float:
        """计算语言异常分数"""
        # 如果句子语言在整体分布中占比很低，则异常分数高
        sentence_lang_ratio = overall_dist.get(sentence_lang, 0)
        
        if sentence_lang_ratio < 0.1:  # 占比小于10%
            return 0.8
        elif sentence_lang_ratio < 0.3:  # 占比小于30%
            return 0.5
        else:
            return 0.2
    
    def detect_contextual_anomalies_enhanced(self, text: str, context: Dict) -> List[Dict]:
        """增强的上下文异常检测"""
        detections = []
        
        if not text or len(text) < 1000:  # 提高最小文本要求
            return detections
        
        try:
            sentences = self._split_sentences_enhanced(text)
            
            # 过滤有效句子
            valid_sentences = [s for s in sentences if len(s.split()) > 5]
            
            if len(valid_sentences) < 8:  # 需要足够的句子
                return detections
            
            # 使用TF-IDF分析
            try:
                tfidf_matrix = self.vectorizer.fit_transform(valid_sentences)
                similarities = cosine_similarity(tfidf_matrix)
                
                for i, sentence in enumerate(valid_sentences):
                    # 计算与其他句子的相似度
                    sentence_similarities = similarities[i]
                    
                    # 排除自身
                    other_similarities = np.concatenate([
                        sentence_similarities[:i], 
                        sentence_similarities[i+1:]
                    ])
                    
                    if len(other_similarities) > 0:
                        avg_similarity = np.mean(other_similarities)
                        max_similarity = np.max(other_similarities)
                        
                        # 更严格的异常条件
                        is_anomalous = (
                            avg_similarity < 0.02 and  # 平均相似度很低
                            max_similarity < 0.1 and   # 最大相似度也很低
                            len(sentence.split()) > 8 and  # 句子足够长
                            self._contains_suspicious_keywords_enhanced(sentence)  # 包含可疑关键词
                        )
                        
                        if is_anomalous:
                            # 计算异常分数
                            anomaly_score = 1.0 - avg_similarity
                            confidence = min(0.7, anomaly_score * 0.6)  # 降低置信度
                            
                            detection = {
                                'type': 'contextual_anomaly',
                                'sentence': sentence[:200],
                                'sentence_index': i,
                                'avg_similarity': float(avg_similarity),
                                'max_similarity': float(max_similarity),
                                'anomaly_score': float(anomaly_score),
                                'confidence': confidence
                            }
                            detections.append(detection)
                            
                            # 限制检测数量，避免过多误报
                            if len(detections) >= 3:
                                break
                                
            except Exception as e:
                logger.debug(f"TF-IDF分析失败: {e}")
                
        except Exception as e:
            logger.debug(f"上下文异常检测失败: {e}")
        
        return detections
    
    def _create_error_result(self, pdf_path: str, error_msg: str) -> Dict[str, Any]:
        """创建错误结果"""
        return {
            'file': pdf_path,
            'detections': [],
            'detection_count': 0,
            'risk_score': 0.0,
            'is_malicious': False,
            'confidence_level': 'unknown',
            'error': error_msg,
            'processing_time': 0
        }
    
    def _update_detection_stats(self, result: Dict):
        """更新检测统计"""
        self.detection_stats['total_processed'] += 1
        self.detection_stats['total_detections'] += result.get('detection_count', 0)
        
        processing_time = result.get('processing_time', 0)
        self.detection_stats['processing_times'].append(processing_time)
        
        # 更新性能监控
        if hasattr(self, 'performance_monitor') and self.performance_monitor.enabled:
            performance_data = {
                'processing_time': processing_time,
                'detection_count': result.get('detection_count', 0),
                'risk_score': result.get('risk_score', 0),
                'is_malicious': result.get('is_malicious', False)
            }
            # 这里可以添加性能数据到监控器
    
    def get_detection_stats(self) -> Dict:
        """获取检测统计信息"""
        stats = self.detection_stats.copy()
        
        if self.detection_stats['processing_times']:
            times = list(self.detection_stats['processing_times'])
            stats['avg_processing_time'] = np.mean(times)
            stats['total_processing_time'] = np.sum(times)
        
        stats['model_info'] = self.model_manager.get_model_info()
        
        return stats
    
    # 为了兼容性，保留原方法名
    def detect_injection(self, pdf_path: str) -> Dict[str, Any]:
        """检测注入攻击（兼容性方法）"""
        return self.detect_injection_enhanced(pdf_path)


# 为了向后兼容，保留原类名的别名
PromptInjectionDetector = EnhancedPromptInjectionDetector
AdaptivePromptInjectionDetector = EnhancedPromptInjectionDetector

# 导出主要类
__all__ = [
    'EnhancedPromptInjectionDetector',
    'PromptInjectionDetector', 
    'AdaptivePromptInjectionDetector',
    'ModelManager',
    'IntelligentThresholdManager',
    'FalsePositiveSupressor',
    'IntelligentWeightManager'
]
