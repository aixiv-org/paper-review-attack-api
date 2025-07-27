import os
import yaml
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger
import hashlib
import re
from datetime import datetime

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """设置日志配置"""
    logger.remove()  # 移除默认handler
    
    # 控制台输出
    logger.add(
        lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # 文件输出
    if log_file:
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_level,
            rotation="10 MB",
            retention="30 days"
        )
    
    return logger

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"配置文件加载成功: {config_path}")
        return config
    except Exception as e:
        logger.error(f"配置文件加载失败: {e}")
        raise

def ensure_dir(dir_path: str) -> str:
    """确保目录存在"""
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dir_path

def calculate_file_hash(file_path: str) -> str:
    """计算文件MD5哈希"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"计算文件哈希失败 {file_path}: {e}")
        return ""

def clean_text(text: str) -> str:
    """清理文本"""
    if not text:
        return ""
    
    # 移除多余空白
    text = re.sub(r'\s+', ' ', text)
    # 移除特殊字符
    text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:()-]', '', text)
    return text.strip()

def extract_metadata_info(metadata: Dict) -> Dict:
    """提取有用的元数据信息"""
    useful_fields = ['title', 'author', 'subject', 'keywords', 'creator', 'producer']
    result = {}
    
    for field in useful_fields:
        if field in metadata and metadata[field]:
            result[field] = str(metadata[field])
    
    return result

def detect_language(text: str) -> str:
    """简单的语言检测"""
    if not text:
        return "unknown"
    
    # 中文字符
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    # 日文字符
    japanese_chars = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text))
    # 英文字符
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    
    total_chars = chinese_chars + japanese_chars + english_chars
    
    if total_chars == 0:
        return "unknown"
    
    chinese_ratio = chinese_chars / total_chars
    japanese_ratio = japanese_chars / total_chars
    english_ratio = english_chars / total_chars
    
    if chinese_ratio > 0.3:
        return "chinese"
    elif japanese_ratio > 0.2:
        return "japanese"
    elif english_ratio > 0.7:
        return "english"
    else:
        return "mixed"

def save_results(results: Dict, output_path: str):
    """保存结果到JSON文件"""
    try:
        ensure_dir(os.path.dirname(output_path))
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"结果已保存到: {output_path}")
    except Exception as e:
        logger.error(f"保存结果失败: {e}")

def load_results(input_path: str) -> Dict:
    """从JSON文件加载结果"""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        logger.info(f"结果已加载: {input_path}")
        return results
    except Exception as e:
        logger.error(f"加载结果失败: {e}")
        return {}

class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()
    
    def update(self, step: int = 1):
        """更新进度"""
        self.current += step
        percentage = (self.current / self.total) * 100
        elapsed = datetime.now() - self.start_time
        
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
            logger.info(f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%) - ETA: {eta}")
        else:
            logger.info(f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%)")
    
    def finish(self):
        """完成进度"""
        elapsed = datetime.now() - self.start_time
        logger.info(f"{self.description} 完成! 用时: {elapsed}")

def validate_pdf(file_path: str) -> bool:
    """验证PDF文件是否有效"""
    try:
        import fitz
        doc = fitz.open(file_path)
        page_count = len(doc)
        doc.close()
        return page_count > 0
    except Exception:
        return False

def get_file_size(file_path: str) -> int:
    """获取文件大小（字节）"""
    try:
        return os.path.getsize(file_path)
    except Exception:
        return 0

def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"