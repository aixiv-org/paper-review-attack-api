import pdfplumber
import fitz
import re
import base64
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
import os
from .utils import setup_logging, detect_language, clean_text

logger = setup_logging()

class LightweightPromptInjectionDetector:
    """轻量级提示词注入检测器（不依赖大型AI模型）"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.detection_config = config['detection']
        self.thresholds = self.detection_config['thresholds']
        self.suspicious_keywords = self.detection_config['suspicious_keywords']
        
        logger.info("轻量级检测器初始化完成（无需加载AI模型）")
    
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
    
    def detect_simple_sentiment(self, text: str) -> List[Dict]:
        """简单的情感检测（基于关键词）"""
        detections = []
        
        # 积极情感关键词
        positive_keywords = [
            'excellent', 'outstanding', 'brilliant', 'exceptional', 'superb',
            'amazing', 'fantastic', 'wonderful', 'perfect', 'great',
            '优秀', '杰出', '出色', '卓越', '完美', '很好', '非常好'
        ]
        
        # 审稿相关词汇
        review_keywords = [
            'review', 'accept', 'recommend', 'publication', 'approve',
            'positive', 'high quality', 'acceptance', 'publish',
            '评审', '接受', '推荐', '发表', '批准', '同意', '高质量'
        ]
        
        sentences = self._split_sentences(text)
        
        for i, sentence in enumerate(sentences):
            sentence_lower = sentence.lower()
            
            # 计算积极词汇密度
            positive_count = sum(1 for word in positive_keywords if word in sentence_lower)
            review_count = sum(1 for word in review_keywords if word in sentence_lower)
            
            # 如果句子同时包含积极词汇和审稿词汇
            if positive_count > 0 and review_count > 0:
                confidence = min(0.9, (positive_count + review_count) * 0.2)
                
                detection = {
                    'type': 'simple_sentiment_injection',
                    'sentence': sentence,
                    'sentence_index': i,
                    'positive_words': positive_count,
                    'review_words': review_count,
                    'confidence': confidence
                }
                detections.append(detection)
        
        return detections
    
    def _split_sentences(self, text: str) -> List[str]:
        """分句"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def detect_format_injection(self, content: Dict) -> List[Dict]:
        """检测格式注入"""
        detections = []
        
        # 检查白色字体
        if content['white_text']:
            white_text = ''.join(content['white_text']).strip()
            if len(white_text) > 10:
                if self._contains_suspicious_keywords(white_text):
                    detection = {
                        'type': 'white_text_injection',
                        'content': white_text[:200],
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
            if small_font_ratio > 0.1:
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
            if total_invisible > 50:
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
    
    def calculate_risk_score(self, detections: List[Dict]) -> float:
        """计算总体风险分数"""
        if not detections:
            return 0.0
        
        # 按检测类型给予不同权重
        type_weights = {
            'keyword_injection': 1.0,
            'simple_sentiment_injection': 0.8,
            'white_text_injection': 1.0,
            'metadata_injection': 0.8,
            'invisible_chars_injection': 0.9,
            'base64_injection': 0.8,
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
        logger.info(f"开始轻量级检测: {pdf_path}")
        
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
            
            # 简单情感检测
            sentiment_detections = self.detect_simple_sentiment(content['text'])
            all_detections.extend(sentiment_detections)
            
            # 格式检测
            format_detections = self.detect_format_injection(content)
            all_detections.extend(format_detections)
            
            # 编码检测
            encoding_detections = self.detect_encoding_injection(content['text'])
            all_detections.extend(encoding_detections)
            
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
        
        logger.info(f"轻量级检测完成: {pdf_path}, 风险分数: {risk_score:.3f}, "
                   f"检测数: {len(all_detections)}, 恶意: {is_malicious}")
        
        return result