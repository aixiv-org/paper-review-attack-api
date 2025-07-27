"""
检测器测试模块
"""

import unittest
import tempfile
import os
from pathlib import Path
import json

# 添加项目根目录到Python路径
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.detector import PromptInjectionDetector
from src.utils import load_config
from unittest.mock import patch, MagicMock

class TestPromptInjectionDetector(unittest.TestCase):
    """提示词注入检测器测试"""
    
    def setUp(self):
        """测试初始化"""
        # 创建测试配置
        self.test_config = {
            'detection': {
                'models': {
                    'sentiment_model': 'cardiffnlp/twitter-xlm-roberta-base-sentiment',
                    'multilingual_model': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
                },
                'thresholds': {
                    'risk_score': 0.5,
                    'sentiment_confidence': 0.9,
                    'white_text_threshold': 0.95,
                    'small_font_size': 2.0
                },
                'suspicious_keywords': {
                    'english': [
                        'recommend acceptance',
                        'give positive review',
                        'excellent work'
                    ],
                    'chinese': [
                        '推荐接受',
                        '给高分',
                        '优秀论文'
                    ]
                }
            }
        }
        
        # 模拟模型加载失败的情况
        with patch('src.detector.pipeline') as mock_pipeline:
            mock_pipeline.side_effect = Exception("Model loading failed")
            self.detector = PromptInjectionDetector(self.test_config)
    
    def test_keyword_detection(self):
        """测试关键词检测"""
        test_text = "This paper is excellent work and should recommend acceptance."
        
        detections = self.detector.detect_keyword_injection(test_text)
        
        self.assertGreater(len(detections), 0)
        self.assertEqual(detections[0]['type'], 'keyword_injection')
        self.assertIn('excellent work', [d['keyword'] for d in detections])
    
    def test_chinese_keyword_detection(self):
        """测试中文关键词检测"""
        test_text = "这篇论文很优秀，推荐接受发表。"
        
        detections = self.detector.detect_keyword_injection(test_text)
        
        self.assertGreater(len(detections), 0)
        chinese_detections = [d for d in detections if d['language'] == 'chinese']
        self.assertGreater(len(chinese_detections), 0)
    
    def test_format_detection(self):
        """测试格式检测"""
        # 模拟包含白色字体的内容
        test_content = {
            'text': 'Normal paper content',
            'white_text': ['recommend acceptance', 'give high score'],
            'small_text': ['tiny text'],
            'metadata': {
                'subject': 'excellent paper review',
                'keywords': 'positive feedback'
            },
            'invisible_chars': ['\u200b\u200c']
        }
        
        detections = self.detector.detect_format_injection(test_content)
        
        # 应该检测到白色字体和元数据注入
        detection_types = [d['type'] for d in detections]
        self.assertIn('white_text_injection', detection_types)
        self.assertIn('metadata_injection', detection_types)
    
    def test_encoding_detection(self):
        """测试编码检测"""
        # Base64编码的"recommend acceptance"
        encoded_text = "Normal text cmVjb21tZW5kIGFjY2VwdGFuY2U= more text"
        
        detections = self.detector.detect_encoding_injection(encoded_text)
        
        # 应该检测到Base64注入
        base64_detections = [d for d in detections if d['type'] == 'base64_injection']
        self.assertGreater(len(base64_detections), 0)
    
    def test_risk_score_calculation(self):
        """测试风险分数计算"""
        detections = [
            {'type': 'keyword_injection', 'confidence': 0.9},
            {'type': 'white_text_injection', 'confidence': 0.95},
            {'type': 'semantic_injection', 'confidence': 0.8}
        ]
        
        risk_score = self.detector.calculate_risk_score(detections)
        
        self.assertGreater(risk_score, 0.5)
        self.assertLessEqual(risk_score, 1.0)
    
    def test_empty_detections_risk_score(self):
        """测试空检测列表的风险分数"""
        risk_score = self.detector.calculate_risk_score([])
        self.assertEqual(risk_score, 0.0)
    
    def test_multilingual_detection(self):
        """测试多语言检测"""
        mixed_text = "This is English text. 这是中文文本。これは日本語です。"
        
        detections = self.detector.detect_multilingual_injection(mixed_text)
        
        # 多语言文本应该被检测到
        self.assertIsInstance(detections, list)
    
    def test_similarity_calculation(self):
        """测试相似度计算"""
        similarity = self.detector._calculate_similarity("accept", "acceptance")
        self.assertGreater(similarity, 0.5)
        
        similarity = self.detector._calculate_similarity("hello", "world")
        self.assertLess(similarity, 0.5)
    
    def test_invisible_chars_detection(self):
        """测试不可见字符检测"""
        text_with_invisible = "Normal text\u200b\u200c\ufeffHidden content"
        
        invisible_chars = self.detector._detect_invisible_chars(text_with_invisible)
        
        self.assertGreater(len(invisible_chars), 0)
    
    def test_white_color_detection(self):
        """测试白色检测"""
        # 测试白色
        self.assertTrue(self.detector._is_white_color((1, 1, 1)))
        self.assertTrue(self.detector._is_white_color((0.99, 0.99, 0.99)))
        
        # 测试非白色
        self.assertFalse(self.detector._is_white_color((0, 0, 0)))
        self.assertFalse(self.detector._is_white_color((0.5, 0.5, 0.5)))

class TestDetectorIntegration(unittest.TestCase):
    """检测器集成测试"""
    
    def setUp(self):
        """测试初始化"""
        # 使用真实配置进行集成测试
        config_path = project_root / "config" / "config.yaml"
        if config_path.exists():
            self.config = load_config(str(config_path))
        else:
            # 使用默认测试配置
            self.config = {
                'detection': {
                    'models': {
                        'sentiment_model': 'cardiffnlp/twitter-xlm-roberta-base-sentiment',
                        'multilingual_model': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
                    },
                    'thresholds': {
                        'risk_score': 0.5,
                        'sentiment_confidence': 0.9,
                        'white_text_threshold': 0.95,
                        'small_font_size': 2.0
                    },
                    'suspicious_keywords': {
                        'english': ['recommend acceptance', 'excellent work'],
                        'chinese': ['推荐接受', '优秀论文']
                    }
                }
            }
    
    @patch('src.detector.pdfplumber')
    @patch('src.detector.fitz')
    def test_pdf_content_extraction(self, mock_fitz, mock_pdfplumber):
        """测试PDF内容提取（模拟）"""
        # 模拟PDF解析
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Test paper content"
        mock_page.chars = [
            {'text': 'T', 'size': 12, 'color': (0, 0, 0)},
            {'text': 'h', 'size': 1, 'color': (1, 1, 1)}  # 白色小字体
        ]
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        # 模拟PyMuPDF
        mock_doc = MagicMock()
        mock_doc.metadata = {'title': 'Test Paper'}
        mock_fitz.open.return_value = mock_doc
        
        detector = PromptInjectionDetector(self.config)
        
        # 创建临时PDF文件路径
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            content = detector.extract_pdf_content(tmp_path)
            
            self.assertIn('text', content)
            self.assertIn('metadata', content)
            self.assertIn('white_text', content)
            self.assertIn('small_text', content)
            
        finally:
            os.unlink(tmp_path)

if __name__ == '__main__':
    unittest.main()
