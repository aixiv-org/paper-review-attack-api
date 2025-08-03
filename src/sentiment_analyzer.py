import re
import warnings
from typing import Dict, Optional
from .utils import setup_logging

logger = setup_logging()
warnings.filterwarnings('ignore')

class LocalSentimentAnalyzer:
    """本地化情感分析器 - 不依赖外部模型"""
    
    def __init__(self):
        self.positive_patterns = self._build_positive_patterns()
        self.negative_patterns = self._build_negative_patterns()
        self.intensity_modifiers = self._build_intensity_modifiers()
        
    def _build_positive_patterns(self) -> Dict[str, float]:
        """构建正面情感模式"""
        return {
            # 强烈正面词汇
            'excellent': 0.9, 'outstanding': 0.9, 'brilliant': 0.9, 'exceptional': 0.9,
            'superb': 0.85, 'magnificent': 0.85, 'remarkable': 0.85, 'extraordinary': 0.85,
            'perfect': 0.8, 'fantastic': 0.8, 'wonderful': 0.8, 'amazing': 0.8,
            
            # 中等正面词汇
            'good': 0.6, 'great': 0.7, 'nice': 0.5, 'fine': 0.5, 'decent': 0.4,
            'solid': 0.6, 'strong': 0.6, 'effective': 0.6, 'successful': 0.7,
            
            # 学术正面词汇
            'innovative': 0.7, 'groundbreaking': 0.9, 'novel': 0.7, 'significant': 0.7,
            'valuable': 0.6, 'important': 0.6, 'contribution': 0.6, 'advance': 0.7,
            
            # 推荐性词汇
            'recommend': 0.8, 'accept': 0.8, 'approve': 0.8, 'endorse': 0.7,
            'support': 0.6, 'favor': 0.6, 'praise': 0.7, 'commend': 0.7,
            
            # 中文正面词汇
            '优秀': 0.9, '杰出': 0.9, '卓越': 0.9, '出色': 0.8, '良好': 0.6,
            '推荐': 0.8, '接受': 0.8, '批准': 0.8, '赞扬': 0.7, '支持': 0.6,
            
            # 日文正面词汇
            '優秀': 0.9, '素晴らしい': 0.8, '良い': 0.6, '推奨': 0.8, '受理': 0.8
        }
    
    def _build_negative_patterns(self) -> Dict[str, float]:
        """构建负面情感模式"""
        return {
            # 强烈负面词汇
            'terrible': 0.9, 'awful': 0.9, 'horrible': 0.9, 'dreadful': 0.9,
            'appalling': 0.85, 'atrocious': 0.85, 'abysmal': 0.85,
            
            # 中等负面词汇
            'bad': 0.7, 'poor': 0.7, 'weak': 0.6, 'inadequate': 0.7,
            'insufficient': 0.6, 'unsatisfactory': 0.7, 'disappointing': 0.7,
            
            # 学术负面词汇
            'flawed': 0.7, 'problematic': 0.7, 'questionable': 0.6, 'unclear': 0.5,
            'limited': 0.4, 'lacking': 0.6, 'deficient': 0.7, 'incomplete': 0.5,
            
            # 拒绝性词汇
            'reject': 0.8, 'decline': 0.7, 'refuse': 0.8, 'oppose': 0.7,
            'disapprove': 0.7, 'criticize': 0.6, 'condemn': 0.8,
            
            # 中文负面词汇
            '糟糕': 0.8, '差': 0.7, '不好': 0.6, '拒绝': 0.8, '反对': 0.7,
            '批评': 0.6, '问题': 0.5, '缺陷': 0.7, '不足': 0.6,
            
            # 日文负面词汇
            '悪い': 0.7, 'ダメ': 0.8, '問題': 0.6, '拒否': 0.8, '反対': 0.7
        }
    
    def _build_intensity_modifiers(self) -> Dict[str, float]:
        """构建强度修饰词"""
        return {
            # 增强词
            'very': 1.5, 'extremely': 2.0, 'incredibly': 1.8, 'absolutely': 1.8,
            'really': 1.3, 'quite': 1.2, 'highly': 1.4, 'tremendously': 1.7,
            'exceptionally': 1.6, 'remarkably': 1.5, 'particularly': 1.3,
            
            # 减弱词
            'somewhat': 0.7, 'rather': 0.8, 'fairly': 0.8, 'moderately': 0.7,
            'slightly': 0.5, 'a little': 0.6, 'kind of': 0.6, 'sort of': 0.6,
            
            # 中文修饰词
            '非常': 1.5, '极其': 2.0, '特别': 1.4, '相当': 1.2, '很': 1.3,
            '有点': 0.6, '稍微': 0.5, '比较': 0.8, '还算': 0.7,
            
            # 日文修饰词
            'とても': 1.5, '非常に': 1.8, 'かなり': 1.2, 'ちょっと': 0.6, '少し': 0.5
        }
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """分析文本情感"""
        if not text:
            return {'label': 'NEUTRAL', 'score': 0.5, 'confidence': 0.0}
        
        text_lower = text.lower()
        words = re.findall(r'\w+', text_lower)
        
        if not words:
            return {'label': 'NEUTRAL', 'score': 0.5, 'confidence': 0.0}
        
        # 计算情感分数
        positive_score = 0.0
        negative_score = 0.0
        total_sentiment_words = 0
        
        for i, word in enumerate(words):
            # 检查正面词汇
            if word in self.positive_patterns:
                score = self.positive_patterns[word]
                # 检查强度修饰词
                if i > 0 and words[i-1] in self.intensity_modifiers:
                    score *= self.intensity_modifiers[words[i-1]]
                positive_score += score
                total_sentiment_words += 1
            
            # 检查负面词汇
            elif word in self.negative_patterns:
                score = self.negative_patterns[word]
                # 检查强度修饰词
                if i > 0 and words[i-1] in self.intensity_modifiers:
                    score *= self.intensity_modifiers[words[i-1]]
                negative_score += score
                total_sentiment_words += 1
        
        # 计算最终分数
        if total_sentiment_words == 0:
            return {'label': 'NEUTRAL', 'score': 0.5, 'confidence': 0.0}
        
        # 归一化分数
        net_score = (positive_score - negative_score) / total_sentiment_words
        confidence = min(1.0, total_sentiment_words / len(words) * 2)  # 基于情感词汇密度
        
        # 转换为0-1分数
        normalized_score = max(0.0, min(1.0, (net_score + 1.0) / 2.0))
        
        # 确定标签
        if normalized_score > 0.6:
            label = 'POSITIVE'
        elif normalized_score < 0.4:
            label = 'NEGATIVE'
        else:
            label = 'NEUTRAL'
        
        return {
            'label': label,
            'score': normalized_score,
            'confidence': confidence,
            'positive_score': positive_score,
            'negative_score': negative_score,
            'sentiment_words_count': total_sentiment_words
        }
    
    def analyze_review_sentiment(self, text: str) -> Dict[str, float]:
        """专门针对学术评审的情感分析"""
        result = self.analyze_sentiment(text)
        
        # 检查评审特有的模式
        review_positive_patterns = [
            r'should\s+be\s+accepted', r'recommend\s+acceptance', r'approve\s+immediately',
            r'excellent\s+(?:work|paper|research)', r'outstanding\s+(?:contribution|work)',
            r'highly\s+recommend', r'strong\s+accept', r'clear\s+accept',
            r'推荐\s*接受', r'应该\s*接受', r'立即\s*批准', r'强烈\s*推荐'
        ]
        
        review_boost = 0.0
        for pattern in review_positive_patterns:
            if re.search(pattern, text.lower()):
                review_boost += 0.2
        
        # 调整分数
        if review_boost > 0:
            result['score'] = min(1.0, result['score'] + review_boost)
            result['confidence'] = min(1.0, result['confidence'] + 0.2)
            if result['score'] > 0.7:
                result['label'] = 'POSITIVE'
        
        return result

class FallbackSentimentAnalyzer:
    """后备情感分析器"""
    
    def __init__(self):
        self.local_analyzer = LocalSentimentAnalyzer()
        logger.info("使用本地情感分析器")
    
    def analyze(self, text: str) -> Dict[str, float]:
        """统一的分析接口"""
        try:
            # 使用本地分析器
            result = self.local_analyzer.analyze_review_sentiment(text)
            return {
                'label': result['label'],
                'score': result['score'],
                'confidence': result.get('confidence', 0.5)
            }
        except Exception as e:
            logger.error(f"情感分析失败: {e}")
            return {'label': 'NEUTRAL', 'score': 0.5, 'confidence': 0.0}
