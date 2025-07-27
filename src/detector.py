import pdfplumber
import fitz
import re
import base64
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from collections import Counter
from .utils import setup_logging, detect_language, clean_text

logger = setup_logging()

class PromptInjectionDetector:
    """提示词注入检测器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.detection_config = config['detection']
        self.thresholds = self.detection_config['thresholds']
        self.suspicious_keywords = self.detection_config['suspicious_keywords']
        
        # 初始化模型
        self._initialize_models()
        
        # 初始化检测器
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        logger.info("提示词注入检测器初始化完成")
    
    def _initialize_models(self):
        """初始化AI模型"""
        try:
            # 情感分析模型
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=self.detection_config['models']['sentiment_model'],
                return_all_scores=True
            )
            
            # 多语言模型
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.detection_config['models']['multilingual_model']
            )
            self.multilingual_model = AutoModel.from_pretrained(
                self.detection_config['models']['multilingual_model']
            )
            
            logger.info("AI模型加载成功")
            
        except Exception as e:
            logger.error(f"AI模型加载失败: {e}")
            # 使用备用方案
            self.sentiment_analyzer = None
            self.tokenizer = None
            self.multilingual_model = None
    
    def extract_pdf_content(self, pdf_path: str) -> Dict[str, Any]:
        """提取PDF内容和格式信息"""
        content = {
            'text': '',
            'metadata': {},
            'white_text': [],
            'small_text': [],
            'invisible_chars': [],
            'font_analysis': {},
            'page_count': 0,
            'file_size': 0
        }
        
        try:
            # 获取文件大小
            import os
            content['file_size'] = os.path.getsize(pdf_path)
            
            # 使用pdfplumber进行详细分析
            with pdfplumber.open(pdf_path) as pdf:
                content['page_count'] = len(pdf.pages)
                full_text = ""
                font_sizes = []
                font_colors = []
                
                for page_num, page in enumerate(pdf.pages):
                    # 提取文本
                    page_text = page.extract_text() or ""
                    full_text += page_text + "\n"
                    
                    # 分析字符格式
                    chars = page.chars if hasattr(page, 'chars') else []
                    
                    for char in chars:
                        try:
                            # 字体大小分析
                            size = char.get('size', 12)
                            font_sizes.append(size)
                            
                            # 颜色分析
                            color = char.get('color', (0, 0, 0))
                            if isinstance(color, (list, tuple)) and len(color) >= 3:
                                font_colors.append(color)
                                
                                # 检查白色或接近白色的文本
                                if self._is_white_color(color):
                                    content['white_text'].append(char.get('text', ''))
                            
                            # 检查极小字体
                            if size < self.thresholds['small_font_size']:
                                content['small_text'].append(char.get('text', ''))
                                
                        except Exception as e:
                            logger.debug(f"字符分析失败: {e}")
                            continue
                
                content['text'] = full_text
                
                # 字体统计
                if font_sizes:
                    content['font_analysis'] = {
                        'avg_font_size': np.mean(font_sizes),
                        'min_font_size': np.min(font_sizes),
                        'max_font_size': np.max(font_sizes),
                        'font_size_std': np.std(font_sizes),
                        'small_font_ratio': len([s for s in font_sizes if s < 4]) / len(font_sizes)
                    }
            
            # 使用PyMuPDF提取元数据
            doc = fitz.open(pdf_path)
            content['metadata'] = doc.metadata or {}
            doc.close()
            
            # 检查不可见字符
            content['invisible_chars'] = self._detect_invisible_chars(content['text'])
            
        except Exception as e:
            logger.error(f"PDF内容提取失败 {pdf_path}: {e}")
        
        return content
    
    def _is_white_color(self, color: Tuple) -> bool:
        """判断是否为白色或接近白色"""
        if len(color) < 3:
            return False
        
        # RGB值接近(1,1,1)或(255,255,255)
        if all(c > 0.95 for c in color[:3]):
            return True
        
        return False
    
    def _detect_invisible_chars(self, text: str) -> List[str]:
        """检测不可见字符"""
        invisible_patterns = [
            r'[\u200b\u200c\u200d\ufeff\u2060\u180e]+',  # 零宽字符
            r'[\u00a0\u2007\u202f]+',  # 非断行空格
            r'[\u034f\u061c\u115f\u1160\u17b4\u17b5]+',  # 其他不可见字符
        ]
        
        invisible_chars = []
        for pattern in invisible_patterns:
            matches = re.findall(pattern, text)
            invisible_chars.extend(matches)
        
        return invisible_chars
    
    def detect_keyword_injection(self, text: str) -> List[Dict]:
        """检测关键词注入"""
        detections = []
        text_lower = text.lower()
        
        for lang, keywords in self.suspicious_keywords.items():
            for keyword in keywords:
                keyword_lower = keyword.lower()
                
                # 直接匹配
                if keyword_lower in text_lower:
                    # 计算出现次数和位置
                    occurrences = len(re.findall(re.escape(keyword_lower), text_lower))
                    positions = [m.start() for m in re.finditer(re.escape(keyword_lower), text_lower)]
                    
                    detection = {
                        'type': 'keyword_injection',
                        'language': lang,
                        'keyword': keyword,
                        'occurrences': occurrences,
                        'positions': positions,
                        'confidence': min(0.95, 0.7 + occurrences * 0.1)
                    }
                    detections.append(detection)
                
                # 模糊匹配（处理变形）
                fuzzy_matches = self._fuzzy_keyword_match(text_lower, keyword_lower)
                for match in fuzzy_matches:
                    detection = {
                        'type': 'keyword_injection_fuzzy',
                        'language': lang,
                        'keyword': keyword,
                        'matched_text': match['text'],
                        'position': match['position'],
                        'confidence': match['confidence']
                    }
                    detections.append(detection)
        
        return detections
    
    def _fuzzy_keyword_match(self, text: str, keyword: str, threshold: float = 0.8) -> List[Dict]:
        """模糊关键词匹配"""
        matches = []
        words = text.split()
        
        for i, word in enumerate(words):
            # 计算编辑距离相似度
            similarity = self._calculate_similarity(word, keyword)
            if similarity > threshold:
                matches.append({
                    'text': word,
                    'position': i,
                    'confidence': similarity * 0.8  # 降低模糊匹配的置信度
                })
        
        return matches
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """计算字符串相似度"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, str1, str2).ratio()
    
    def detect_semantic_injection(self, text: str) -> List[Dict]:
        """检测语义注入"""
        detections = []
        
        if not self.sentiment_analyzer:
            logger.warning("情感分析模型未加载，跳过语义检测")
            return detections
        
        # 分句处理
        sentences = self._split_sentences(text)
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            try:
                # 情感分析
                results = self.sentiment_analyzer(sentence[:512])
                
                for result in results:
                    if (result['label'] == 'POSITIVE' and 
                        result['score'] > self.thresholds['sentiment_confidence']):
                        
                        # 检查是否包含审稿相关词汇
                        review_keywords = [
                            'review', 'accept', 'recommend', 'excellent', 'outstanding',
                            'publication', 'approve', 'positive', 'high quality'
                        ]
                        
                        if any(keyword in sentence.lower() for keyword in review_keywords):
                            detection = {
                                'type': 'semantic_injection',
                                'sentence': sentence,
                                'sentence_index': i,
                                'sentiment_label': result['label'],
                                'sentiment_score': result['score'],
                                'confidence': min(0.9, result['score'])
                            }
                            detections.append(detection)
                            
            except Exception as e:
                logger.debug(f"语义分析失败: {e}")
                continue
        
        return detections
    
    def _split_sentences(self, text: str) -> List[str]:
        """分句"""
        # 简单的分句方法
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def detect_format_injection(self, content: Dict) -> List[Dict]:
        """检测格式注入"""
        detections = []
        
        # 检查白色字体
        if content['white_text']:
            white_text = ''.join(content['white_text']).strip()
            if len(white_text) > 10:  # 有足够长度的白色文本
                # 检查是否包含可疑关键词
                if self._contains_suspicious_keywords(white_text):
                    detection = {
                        'type': 'white_text_injection',
                        'content': white_text[:200],  # 限制长度
                        'length': len(white_text),
                        'confidence': self.thresholds['white_text_threshold']
                    }
                    detections.append(detection)
        
        # 检查极小字体
        if content['small_text']:
            small_text = ''.join(content['small_text']).strip()
            if len(small_text) > 20:
                detection = {
                    'type': 'small_text_injection',
                    'content': small_text[:200],
                    'length': len(small_text),
                    'confidence': 0.8
                }
                detections.append(detection)
        
        # 检查字体分析异常
        font_analysis = content.get('font_analysis', {})
        if font_analysis:
            small_font_ratio = font_analysis.get('small_font_ratio', 0)
            if small_font_ratio > 0.1:  # 超过10%的文本使用小字体
                detection = {
                    'type': 'suspicious_font_pattern',
                    'small_font_ratio': small_font_ratio,
                    'min_font_size': font_analysis.get('min_font_size', 0),
                    'confidence': min(0.9, small_font_ratio * 5)
                }
                detections.append(detection)
        
        # 检查元数据
        metadata = content.get('metadata', {})
        for field, value in metadata.items():
            if isinstance(value, str) and value:
                if self._contains_suspicious_keywords(value):
                    detection = {
                        'type': 'metadata_injection',
                        'field': field,
                        'content': value[:200],
                        'confidence': 0.9
                    }
                    detections.append(detection)
        
        # 检查不可见字符
        invisible_chars = content.get('invisible_chars', [])
        if invisible_chars:
            total_invisible = sum(len(chars) for chars in invisible_chars)
            if total_invisible > 50:  # 超过50个不可见字符
                detection = {
                    'type': 'invisible_chars_injection',
                    'count': total_invisible,
                    'samples': invisible_chars[:3],
                    'confidence': min(0.95, total_invisible / 100)
                }
                detections.append(detection)
        
        return detections
    
    def _contains_suspicious_keywords(self, text: str) -> bool:
        """检查文本是否包含可疑关键词"""
        text_lower = text.lower()
        
        for keywords in self.suspicious_keywords.values():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    return True
        
        return False
    
    def detect_encoding_injection(self, text: str) -> List[Dict]:
        """检测编码注入攻击"""
        detections = []
        
        # 检查Base64编码
        base64_pattern = r'[A-Za-z0-9+/]{20,}={0,2}'
        base64_matches = re.findall(base64_pattern, text)
        
        for match in base64_matches:
            try:
                decoded = base64.b64decode(match).decode('utf-8')
                if self._contains_suspicious_keywords(decoded):
                    detection = {
                        'type': 'base64_injection',
                        'encoded': match[:50],
                        'decoded': decoded[:100],
                        'confidence': 0.85
                    }
                    detections.append(detection)
            except Exception:
                continue
        
        # 检查URL编码
        url_encoded_pattern = r'%[0-9A-Fa-f]{2}'
        if re.search(url_encoded_pattern, text):
            try:
                import urllib.parse
                decoded = urllib.parse.unquote(text)
                if decoded != text and self._contains_suspicious_keywords(decoded):
                    detection = {
                        'type': 'url_encoding_injection',
                        'original': text[:100],
                        'decoded': decoded[:100],
                        'confidence': 0.8
                    }
                    detections.append(detection)
            except Exception:
                pass
        
        return detections
    
    def detect_multilingual_injection(self, text: str) -> List[Dict]:
        """检测多语言注入"""
        detections = []
        
        # 检测语言分布
        language_dist = self._analyze_language_distribution(text)
        
        # 如果存在多种语言混合且包含可疑内容
        if len(language_dist) > 2:
            sentences = self._split_sentences(text)
            
            for sentence in sentences:
                lang = detect_language(sentence)
                if lang in self.suspicious_keywords:
                    if self._contains_suspicious_keywords(sentence):
                        detection = {
                            'type': 'multilingual_injection',
                            'sentence': sentence[:100],
                            'detected_language': lang,
                            'language_distribution': language_dist,
                            'confidence': 0.75
                        }
                        detections.append(detection)
        
        return detections
    
    def _analyze_language_distribution(self, text: str) -> Dict[str, float]:
        """分析文本的语言分布"""
        # 统计不同语言字符的比例
        total_chars = len(text)
        if total_chars == 0:
            return {}
        
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        japanese_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        distribution = {}
        if chinese_chars > 0:
            distribution['chinese'] = chinese_chars / total_chars
        if japanese_chars > 0:
            distribution['japanese'] = japanese_chars / total_chars
        if english_chars > 0:
            distribution['english'] = english_chars / total_chars
        
        return distribution
    
    def detect_contextual_anomalies(self, content: Dict) -> List[Dict]:
        """检测上下文异常"""
        detections = []
        text = content.get('text', '')
        
        if not text:
            return detections
        
        # 检查文本连贯性
        sentences = self._split_sentences(text)
        
        if len(sentences) < 10:
            return detections
        
        try:
            # 使用TF-IDF检测异常句子
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            
            # 计算每个句子与其他句子的相似度
            similarities = cosine_similarity(tfidf_matrix)
            
            for i, sentence in enumerate(sentences):
                # 计算该句子与其他句子的平均相似度
                avg_similarity = np.mean(similarities[i])
                
                # 如果相似度过低且包含可疑关键词
                if (avg_similarity < 0.1 and 
                    self._contains_suspicious_keywords(sentence)):
                    
                    detection = {
                        'type': 'contextual_anomaly',
                        'sentence': sentence[:100],
                        'sentence_index': i,
                        'avg_similarity': float(avg_similarity),
                        'confidence': 0.7
                    }
                    detections.append(detection)
                    
        except Exception as e:
            logger.debug(f"上下文分析失败: {e}")
        
        return detections
    
    def calculate_risk_score(self, detections: List[Dict]) -> float:
        """计算总体风险分数"""
        if not detections:
            return 0.0
        
        # 按检测类型给予不同权重
        type_weights = {
            'keyword_injection': 1.0,
            'semantic_injection': 0.9,
            'white_text_injection': 1.0,
            'metadata_injection': 0.8,
            'invisible_chars_injection': 0.9,
            'base64_injection': 0.8,
            'multilingual_injection': 0.7,
            'contextual_anomaly': 0.6,
            'small_text_injection': 0.7,
            'suspicious_font_pattern': 0.5
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for detection in detections:
            detection_type = detection['type']
            confidence = detection['confidence']
            weight = type_weights.get(detection_type, 0.5)
            
            total_score += confidence * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        # 归一化到0-1范围
        risk_score = min(1.0, total_score / total_weight)
        
        # 多个检测类型会增加风险
        type_bonus = min(0.2, len(set(d['type'] for d in detections)) * 0.05)
        risk_score = min(1.0, risk_score + type_bonus)
        
        return risk_score
    
    def detect_injection(self, pdf_path: str) -> Dict[str, Any]:
        """综合检测注入攻击"""
        logger.info(f"开始检测: {pdf_path}")
        
        # 提取内容
        content = self.extract_pdf_content(pdf_path)
        
        if not content['text'] and not content['metadata']:
            logger.warning(f"无法提取PDF内容: {pdf_path}")
            return {
                'file': pdf_path,
                'detections': [],
                'risk_score': 0.0,
                'is_malicious': False,
                'error': 'Content extraction failed'
            }
        
        # 执行各种检测
        all_detections = []
        
        try:
            # 关键词检测
            keyword_detections = self.detect_keyword_injection(content['text'])
            all_detections.extend(keyword_detections)
            
            # 语义检测
            semantic_detections = self.detect_semantic_injection(content['text'])
            all_detections.extend(semantic_detections)
            
            # 格式检测
            format_detections = self.detect_format_injection(content)
            all_detections.extend(format_detections)
            
            # 编码检测
            encoding_detections = self.detect_encoding_injection(content['text'])
            all_detections.extend(encoding_detections)
            
            # 多语言检测
            multilingual_detections = self.detect_multilingual_injection(content['text'])
            all_detections.extend(multilingual_detections)
            
            # 上下文异常检测
            contextual_detections = self.detect_contextual_anomalies(content)
            all_detections.extend(contextual_detections)
            
        except Exception as e:
            logger.error(f"检测过程中出错: {e}")
        
        # 计算风险分数
        risk_score = self.calculate_risk_score(all_detections)
        is_malicious = risk_score > self.thresholds['risk_score']
        
        result = {
            'file': pdf_path,
            'detections': all_detections,
            'detection_count': len(all_detections),
            'risk_score': risk_score,
            'is_malicious': is_malicious,
            'content_stats': {
                'text_length': len(content['text']),
                'page_count': content['page_count'],
                'file_size': content['file_size'],
                'white_text_count': len(content['white_text']),
                'small_text_count': len(content['small_text']),
                'invisible_chars_count': len(content['invisible_chars'])
            }
        }
        
        logger.info(f"检测完成: {pdf_path}, 风险分数: {risk_score:.3f}, "
                   f"检测数: {len(all_detections)}, 恶意: {is_malicious}")
        
        return result

class EnsembleDetector:
    """集成检测器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.detectors = []
        
        # 创建多个检测器实例
        self.primary_detector = PromptInjectionDetector(config)
        
        logger.info("集成检测器初始化完成")
    
    def detect_injection(self, pdf_path: str) -> Dict[str, Any]:
        """使用多个检测器进行检测"""
        results = []
        
        # 主检测器
        primary_result = self.primary_detector.detect_injection(pdf_path)
        results.append(primary_result)
        
        # 这里可以添加更多检测器
        
        # 合并结果
        return self._merge_results(results)
    
    def _merge_results(self, results: List[Dict]) -> Dict[str, Any]:
        """合并多个检测器的结果"""
        if not results:
            return {}
        
        if len(results) == 1:
            return results[0]
        
        # 合并检测结果
        merged_detections = []
        risk_scores = []
        
        for result in results:
            merged_detections.extend(result.get('detections', []))
            risk_scores.append(result.get('risk_score', 0))
        
        # 计算平均风险分数
        avg_risk_score = np.mean(risk_scores) if risk_scores else 0
        max_risk_score = max(risk_scores) if risk_scores else 0
        
        # 使用更保守的方法：取最大值和平均值的加权平均
        final_risk_score = 0.7 * max_risk_score + 0.3 * avg_risk_score
        
        merged_result = results[0].copy()
        merged_result.update({
            'detections': merged_detections,
            'detection_count': len(merged_detections),
            'risk_score': final_risk_score,
            'is_malicious': final_risk_score > self.config['detection']['thresholds']['risk_score'],
            'ensemble_scores': risk_scores
        })
        
        return merged_result