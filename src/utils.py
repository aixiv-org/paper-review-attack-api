import os
import yaml
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from loguru import logger
import hashlib
import re
import warnings
import fitz
import psutil
import time
import threading
import queue
import pickle
import gzip
import sqlite3
from datetime import datetime, timedelta
from functools import wraps, lru_cache
from collections import defaultdict, deque
import numpy as np

# ============================================================================
# 🔧 基础工具函数 (保持你的原有代码)
# ============================================================================

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None, config: Dict = None):
    """设置增强的日志配置"""
    logger.remove()  # 移除默认handler
    
    # 如果提供了配置，使用配置中的设置
    if config:
        log_config = config.get('logging', {})
        log_level = log_config.get('levels', {}).get('console', log_level)
        if not log_file:
            log_file = log_config.get('files', {}).get('main_log')
    
    # 控制台输出
    console_format = ("<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                     "<level>{level: <8}</level> | "
                     "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                     "<level>{message}</level>")
    
    logger.add(
        lambda msg: print(msg, end=""),
        format=console_format,
        level=log_level,
        colorize=True
    )
    
    # 文件输出
    if log_file:
        ensure_dir(os.path.dirname(log_file))
        file_format = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
        
        logger.add(
            log_file,
            format=file_format,
            level="DEBUG",
            rotation="10 MB",
            retention="30 days",
            compression="zip"
        )
        
        # 错误日志单独文件
        if config and config.get('logging', {}).get('files', {}).get('error_log'):
            error_log = config['logging']['files']['error_log']
            ensure_dir(os.path.dirname(error_log))
            logger.add(
                error_log,
                format=file_format,
                level="ERROR",
                rotation="5 MB",
                retention="30 days"
            )
    
    return logger

@lru_cache(maxsize=1)
def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """加载配置文件（带缓存）"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 验证配置
        config = validate_and_fill_config(config)
        
        logger.info(f"配置文件加载成功: {config_path}")
        return config
    except Exception as e:
        logger.error(f"配置文件加载失败: {e}")
        raise

def validate_and_fill_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """验证并填充配置默认值"""
    # 确保必要的配置存在
    default_config = {
        'detection': {
            'offline_mode': True,
            'thresholds': {
                'risk_score': 0.35,
                'sentiment_confidence': 0.85,
                'keyword_match': 0.7,
                'detection_count': 2
            },
            'detection_weights': {
                'semantic_injection': 1.8,
                'contextual_anomaly': 1.6,
                'keyword_injection': 1.4,
                'small_text_injection': 0.4
            },
            'false_positive_suppression': {
                'enabled': True,
                'max_small_text_ratio': 0.03
            }
        },
        'experiment': {
            'output_dir': './data/results',
            'visualization': {
                'figsize': [12, 8],
                'dpi': 300
            }
        },
        'logging': {
            'level': 'INFO',
            'files': {
                'main_log': './logs/experiment.log'
            }
        }
    }
    
    # 递归合并配置
    def merge_dict(base, override):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                merge_dict(base[key], value)
            else:
                base[key] = value
    
    merge_dict(default_config, config)
    return default_config

# ============================================================================
# 🚀 新增：性能监控和资源管理
# ============================================================================

class PerformanceMonitor:
    """增强的性能监控器"""
    
    def __init__(self, config: Dict = None):
        self.config = config.get('logging', {}).get('monitoring', {}) if config else {}
        self.enabled = self.config.get('enabled', False)
        self.metrics = deque(maxlen=1000)  # 保留最近1000条记录
        self.start_time = time.time()
        self.alerts = []
        self._monitoring = False
        self._monitor_thread = None
        
    def start_monitoring(self):
        """开始监控"""
        if not self.enabled or self._monitoring:
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("性能监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("性能监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self._monitoring:
            try:
                # 收集系统指标
                metrics = {
                    'timestamp': time.time(),
                    'memory_usage': psutil.virtual_memory().percent / 100,
                    'cpu_usage': psutil.cpu_percent(interval=1) / 100,
                    'disk_usage': psutil.disk_usage('.').percent / 100,
                    'process_memory': psutil.Process().memory_info().rss / (1024**3)  # GB
                }
                
                self.metrics.append(metrics)
                self._check_alerts(metrics)
                
                time.sleep(30)  # 每30秒监控一次
                
            except Exception as e:
                logger.error(f"监控错误: {e}")
                time.sleep(60)  # 出错后等待更长时间
    
    def _check_alerts(self, metrics: Dict):
        """检查告警"""
        alerts = self.config.get('alerts', {})
        
        # 内存告警
        if metrics['memory_usage'] > alerts.get('memory_threshold', 0.8):
            alert = f"内存使用率过高: {metrics['memory_usage']:.1%}"
            if alert not in self.alerts:
                self.alerts.append(alert)
                logger.warning(alert)
        
        # CPU告警
        if metrics['cpu_usage'] > alerts.get('cpu_threshold', 0.9):
            alert = f"CPU使用率过高: {metrics['cpu_usage']:.1%}"
            if alert not in self.alerts:
                self.alerts.append(alert)
                logger.warning(alert)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        if not self.metrics:
            return {}
        
        recent_metrics = list(self.metrics)[-10:]  # 最近10条记录
        
        return {
            'avg_memory_usage': np.mean([m['memory_usage'] for m in recent_metrics]),
            'avg_cpu_usage': np.mean([m['cpu_usage'] for m in recent_metrics]),
            'peak_memory': max([m['memory_usage'] for m in recent_metrics]),
            'alerts_count': len(self.alerts),
            'uptime': time.time() - self.start_time
        }

class CacheManager:
    """智能缓存管理器"""
    
    def __init__(self, config: Dict = None):
        self.config = config.get('resource_management', {}).get('cache', {}) if config else {}
        self.enabled = self.config.get('enabled', True)
        self.max_size = self._parse_size(self.config.get('max_size', '1GB'))
        self.cache_dir = Path(self.config.get('directory', './cache'))
        self.ttl = self._parse_duration(self.config.get('ttl', '7d'))
        
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._init_cache_db()
    
    def _parse_size(self, size_str: str) -> int:
        """解析大小字符串为字节数"""
        units = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
        if isinstance(size_str, int):
            return size_str
        
        size_str = size_str.upper().strip()
        for unit, multiplier in units.items():
            if size_str.endswith(unit):
                return int(float(size_str[:-len(unit)]) * multiplier)
        return int(size_str)
    
    def _parse_duration(self, duration_str: str) -> int:
        """解析时间字符串为秒数"""
        units = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}
        if isinstance(duration_str, int):
            return duration_str
        
        duration_str = duration_str.lower().strip()
        for unit, multiplier in units.items():
            if duration_str.endswith(unit):
                return int(float(duration_str[:-1]) * multiplier)
        return int(duration_str)
    
    def _init_cache_db(self):
        """初始化缓存数据库"""
        self.db_path = self.cache_dir / 'cache.db'
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    file_path TEXT,
                    created_at REAL,
                    accessed_at REAL,
                    size INTEGER
                )
            ''')
            conn.commit()
    
    def get(self, key: str):
        """获取缓存"""
        if not self.enabled:
            return None
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute(
                    'SELECT file_path, created_at FROM cache_entries WHERE key = ?',
                    (key,)
                )
                row = cursor.fetchone()
                
                if row:
                    file_path, created_at = row
                    
                    # 检查是否过期
                    if time.time() - created_at > self.ttl:
                        self.delete(key)
                        return None
                    
                    # 更新访问时间
                    conn.execute(
                        'UPDATE cache_entries SET accessed_at = ? WHERE key = ?',
                        (time.time(), key)
                    )
                    conn.commit()
                    
                    # 读取缓存文件
                    cache_file = Path(file_path)
                    if cache_file.exists():
                        if self.config.get('compression', True):
                            with gzip.open(cache_file, 'rb') as f:
                                return pickle.load(f)
                        else:
                            with open(cache_file, 'rb') as f:
                                return pickle.load(f)
        
        except Exception as e:
            logger.debug(f"缓存读取失败 {key}: {e}")
        
        return None
    
    def set(self, key: str, value: Any):
        """设置缓存"""
        if not self.enabled:
            return
        
        try:
            # 生成缓存文件路径
            cache_file = self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.cache"
            
            # 保存数据
            if self.config.get('compression', True):
                with gzip.open(cache_file, 'wb') as f:
                    pickle.dump(value, f)
            else:
                with open(cache_file, 'wb') as f:
                    pickle.dump(value, f)
            
            file_size = cache_file.stat().st_size
            current_time = time.time()
            
            # 更新数据库
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO cache_entries 
                    (key, file_path, created_at, accessed_at, size)
                    VALUES (?, ?, ?, ?, ?)
                ''', (key, str(cache_file), current_time, current_time, file_size))
                conn.commit()
            
            # 检查缓存大小限制
            self._cleanup_if_needed()
        
        except Exception as e:
            logger.debug(f"缓存保存失败 {key}: {e}")
    
    def delete(self, key: str):
        """删除缓存"""
        if not self.enabled:
            return
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute(
                    'SELECT file_path FROM cache_entries WHERE key = ?',
                    (key,)
                )
                row = cursor.fetchone()
                
                if row:
                    file_path = Path(row[0])
                    if file_path.exists():
                        file_path.unlink()
                    
                    conn.execute('DELETE FROM cache_entries WHERE key = ?', (key,))
                    conn.commit()
        
        except Exception as e:
            logger.debug(f"缓存删除失败 {key}: {e}")
    
    def _cleanup_if_needed(self):
        """如果需要，清理缓存"""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # 获取总大小
                cursor = conn.execute('SELECT SUM(size) FROM cache_entries')
                total_size = cursor.fetchone()[0] or 0
                
                if total_size > self.max_size:
                    # 按LRU策略删除
                    cursor = conn.execute('''
                        SELECT key, file_path FROM cache_entries 
                        ORDER BY accessed_at ASC
                    ''')
                    
                    for key, file_path in cursor.fetchall():
                        self.delete(key)
                        
                        # 重新检查大小
                        cursor2 = conn.execute('SELECT SUM(size) FROM cache_entries')
                        current_size = cursor2.fetchone()[0] or 0
                        
                        if current_size <= self.max_size * 0.8:  # 清理到80%
                            break
        
        except Exception as e:
            logger.debug(f"缓存清理失败: {e}")
    
    def clear(self):
        """清空所有缓存"""
        if not self.enabled:
            return
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute('SELECT file_path FROM cache_entries')
                for (file_path,) in cursor.fetchall():
                    cache_file = Path(file_path)
                    if cache_file.exists():
                        cache_file.unlink()
                
                conn.execute('DELETE FROM cache_entries')
                conn.commit()
            
            logger.info("缓存已清空")
        
        except Exception as e:
            logger.error(f"清空缓存失败: {e}")
    
    def get_stats(self) -> Dict:
        """获取缓存统计"""
        if not self.enabled:
            return {'enabled': False}
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute('''
                    SELECT COUNT(*), SUM(size), MAX(accessed_at), MIN(created_at)
                    FROM cache_entries
                ''')
                row = cursor.fetchone()
                
                if row and row[0]:
                    count, total_size, last_access, first_created = row
                    return {
                        'enabled': True,
                        'entry_count': count,
                        'total_size': total_size,
                        'total_size_formatted': format_file_size(total_size),
                        'last_access': datetime.fromtimestamp(last_access).isoformat() if last_access else None,
                        'oldest_entry': datetime.fromtimestamp(first_created).isoformat() if first_created else None,
                        'hit_rate': getattr(self, '_hit_count', 0) / max(getattr(self, '_total_requests', 1), 1)
                    }
        
        except Exception as e:
            logger.debug(f"获取缓存统计失败: {e}")
        
        return {'enabled': True, 'entry_count': 0, 'total_size': 0}

class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self, config: Dict = None):
        self.config = config.get('resource_management', {}) if config else {}
        self.limits = {
            'max_memory': self._parse_size(self.config.get('memory', {}).get('max_usage', '8GB')),
            'max_cpu_cores': self.config.get('cpu', {}).get('max_cores', 4)
        }
        self.warnings_sent = set()
    
    def _parse_size(self, size_str: str) -> int:
        """解析大小字符串"""
        units = {'B': 1, 'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
        if isinstance(size_str, int):
            return size_str
        
        size_str = size_str.upper().strip()
        for unit, multiplier in units.items():
            if size_str.endswith(unit):
                return int(float(size_str[:-len(unit)]) * multiplier)
        return int(size_str)
    
    def check_memory_usage(self) -> bool:
        """检查内存使用情况"""
        current_usage = psutil.virtual_memory().used
        
        if current_usage > self.limits['max_memory']:
            warning_key = f"memory_{int(time.time() // 300)}"  # 每5分钟最多警告一次
            if warning_key not in self.warnings_sent:
                logger.warning(f"内存使用超限: {format_file_size(current_usage)} > {format_file_size(self.limits['max_memory'])}")
                self.warnings_sent.add(warning_key)
            return False
        
        return True
    
    def get_available_cores(self) -> int:
        """获取可用CPU核心数"""
        return min(psutil.cpu_count(), self.limits['max_cpu_cores'])
    
    def get_system_info(self) -> Dict:
        """获取系统信息"""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        
        return {
            'cpu_cores': psutil.cpu_count(),
            'cpu_usage': psutil.cpu_percent(interval=1),
            'memory_total': memory.total,
            'memory_used': memory.used,
            'memory_available': memory.available,
            'memory_percent': memory.percent,
            'disk_total': disk.total,
            'disk_used': disk.used,
            'disk_free': disk.free,
            'disk_percent': (disk.used / disk.total) * 100
        }

# ============================================================================
# 🎯 新增：模型管理相关工具
# ============================================================================

def check_model_availability(model_name: str, model_type: str = "huggingface") -> bool:
    """检查模型是否可用"""
    try:
        if model_type == "huggingface":
            from transformers import AutoConfig
            # 尝试加载配置文件，不下载模型
            AutoConfig.from_pretrained(model_name, trust_remote_code=False)
            return True
        
        elif model_type == "local":
            return os.path.exists(model_name)
        
        elif model_type == "textblob":
            try:
                from textblob import TextBlob
                return True
            except ImportError:
                return False
        
        elif model_type == "vader":
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                return True
            except ImportError:
                return False
    
    except Exception as e:
        logger.debug(f"模型 {model_name} 不可用: {e}")
        return False

def get_model_cache_path(model_name: str) -> Optional[str]:
    """获取模型缓存路径"""
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        
        # 尝试常见的缓存目录
        cache_dirs = [
            os.path.expanduser("~/.cache/huggingface/transformers"),
            os.path.expanduser("~/.cache/huggingface/hub"),
            "./models"
        ]
        
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                # 查找模型文件
                for root, dirs, files in os.walk(cache_dir):
                    if any(model_name.replace('/', '--') in d for d in dirs):
                        return root
                    if 'config.json' in files:
                        with open(os.path.join(root, 'config.json'), 'r') as f:
                            cached_config = json.load(f)
                            if cached_config.get('_name_or_path') == model_name:
                                return root
    
    except Exception as e:
        logger.debug(f"获取模型缓存路径失败: {e}")
    
    return None

def download_and_cache_model(model_name: str, cache_dir: str = "./models") -> bool:
    """下载并缓存模型"""
    try:
        ensure_dir(cache_dir)
        
        from transformers import AutoTokenizer, AutoModel
        
        logger.info(f"下载模型: {model_name}")
        
        # 下载到指定目录
        model_path = os.path.join(cache_dir, model_name.replace('/', '--'))
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)
        
        logger.info(f"模型已缓存到: {model_path}")
        return True
    
    except Exception as e:
        logger.error(f"下载模型失败: {e}")
        return False

# ============================================================================
# 🔍 新增：智能文件处理
# ============================================================================

def smart_file_validator(file_path: str, config: Dict = None) -> Dict[str, Any]:
    """智能文件验证器"""
    result = {
        'is_valid': False,
        'file_type': 'unknown',
        'size': 0,
        'issues': [],
        'quality_score': 0.0,
        'metadata': {}
    }
    
    try:
        if not os.path.exists(file_path):
            result['issues'].append('文件不存在')
            return result
        
        file_size = os.path.getsize(file_path)
        result['size'] = file_size
        
        # 基于配置的质量检查
        quality_config = config.get('data_collection', {}).get('quality_control', {}) if config else {}
        
        # 文件大小检查
        min_size = quality_config.get('min_file_size', 50000)
        max_size = quality_config.get('max_file_size', 10485760)
        
        if file_size < min_size:
            result['issues'].append(f'文件过小: {format_file_size(file_size)} < {format_file_size(min_size)}')
        elif file_size > max_size:
            result['issues'].append(f'文件过大: {format_file_size(file_size)} > {format_file_size(max_size)}')
        
        # PDF特定检查
        if file_path.lower().endswith('.pdf'):
            result['file_type'] = 'pdf'
            pdf_info = get_pdf_info(file_path)
            result['metadata'] = pdf_info
            
            if pdf_info['is_valid']:
                result['is_valid'] = True
                
                # 页数检查
                min_pages = quality_config.get('min_pages', 4)
                max_pages = quality_config.get('max_pages', 50)
                page_count = pdf_info['page_count']
                
                if page_count < min_pages:
                    result['issues'].append(f'页数过少: {page_count} < {min_pages}')
                elif page_count > max_pages:
                    result['issues'].append(f'页数过多: {page_count} > {max_pages}')
                
                # 文本内容检查
                if not pdf_info['has_text']:
                    result['issues'].append('缺少文本内容')
                
                # 计算质量分数
                score = 1.0
                score -= len(result['issues']) * 0.2  # 每个问题扣0.2分
                score = max(0.0, min(1.0, score))
                result['quality_score'] = score
            
            else:
                result['issues'].append('PDF文件损坏或无效')
        
        # 其他文件类型的检查可以在这里添加
        
    except Exception as e:
        result['issues'].append(f'验证过程出错: {str(e)}')
        logger.debug(f"文件验证失败 {file_path}: {e}")
    
    return result

def batch_file_processor(file_paths: List[str], 
                        processor_func: callable,
                        max_workers: int = 4,
                        progress_desc: str = "处理文件") -> List[Any]:
    """批量文件处理器"""
    results = []
    
    if not file_paths:
        return results
    
    # 单线程处理（避免复杂性）
    with ProgressTracker(len(file_paths), progress_desc) as progress:
        for file_path in file_paths:
            try:
                result = processor_func(file_path)
                results.append(result)
            except Exception as e:
                logger.debug(f"处理文件失败 {file_path}: {e}")
                results.append(None)
            
            progress.update()
    
    return results

# ============================================================================
# 🎛️ 新增：配置工具
# ============================================================================

def create_default_config() -> Dict[str, Any]:
    """创建默认配置"""
    return {
        'data_collection': {
            'download_dir': './data/clean_papers',
            'max_papers': 300,
            'quality_control': {
                'min_file_size': 50000,
                'max_file_size': 10485760,
                'min_pages': 4,
                'max_pages': 50
            }
        },
        'detection': {
            'offline_mode': True,
            'thresholds': {
                'risk_score': 0.35,
                'detection_count': 2
            },
            'detection_weights': {
                'semantic_injection': 1.8,
                'small_text_injection': 0.4
            }
        },
        'experiment': {
            'output_dir': './data/results'
        },
        'logging': {
            'level': 'INFO',
            'files': {
                'main_log': './logs/experiment.log'
            }
        }
    }

def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """合并配置"""
    def _merge_dict(base, override):
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = _merge_dict(result[key], value)
            else:
                result[key] = value
        return result
    
    return _merge_dict(base_config, override_config)

# ============================================================================
# 保持你的原有函数（ensure_dir, calculate_file_hash 等）
# ============================================================================

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
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
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
    """进度跟踪器 - 改进版"""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = datetime.now()
        self.last_update_time = datetime.now()
        self.update_interval = 1.0  # 最小更新间隔（秒）
    
    def update(self, step: int = 1):
        """更新进度"""
        self.current += step
        
        # 限制更新频率
        now = datetime.now()
        if (now - self.last_update_time).total_seconds() < self.update_interval and self.current < self.total:
            return
        
        self.last_update_time = now
        percentage = (self.current / self.total) * 100
        elapsed = now - self.start_time
        
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
            eta_str = str(eta).split('.')[0]  # 去掉微秒
            logger.info(f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%) - ETA: {eta_str}")
        else:
            logger.info(f"{self.description}: {self.current}/{self.total} ({percentage:.1f}%)")
    
    def finish(self):
        """完成进度"""
        elapsed = datetime.now() - self.start_time
        elapsed_str = str(elapsed).split('.')[0]  # 去掉微秒
        logger.info(f"{self.description} 完成! 用时: {elapsed_str}")
    
    def __enter__(self):
        """支持上下文管理器"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """自动完成"""
        self.finish()

# ============================================================================
# 保持你的原有PDF处理函数
# ============================================================================

def configure_pdf_error_suppression():
    """配置PDF错误抑制"""
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    try:
        if hasattr(fitz, 'TOOLS'):
            fitz.TOOLS.mupdf_display_errors(False)
    except (AttributeError, Exception):
        pass
    
    os.environ['MUPDF_DISPLAY_ERRORS'] = '0'

def safe_pdf_operation(func):
    """PDF操作的安全装饰器"""
    def wrapper(*args, **kwargs):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return func(*args, **kwargs)
        except Exception as e:
            logger.debug(f"PDF操作失败: {e}")
            return None
    return wrapper

@safe_pdf_operation
def validate_pdf(file_path: str, repair_if_needed: bool = True) -> bool:
    """增强的PDF验证函数"""
    if not os.path.exists(file_path):
        return False
    
    if os.path.getsize(file_path) == 0:
        return False
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            doc = fitz.open(file_path)
            
            if doc.page_count == 0:
                doc.close()
                return False
            
            try:
                page = doc[0]
                rect = page.rect
                if rect.width <= 0 or rect.height <= 0:
                    doc.close()
                    return False
                
                try:
                    text = page.get_text()[:100]
                except:
                    pass
                    
            except Exception as e:
                logger.debug(f"PDF页面访问失败 {file_path}: {e}")
                doc.close()
                return False
            
            doc.close()
            return True
            
    except Exception as e:
        logger.debug(f"PDF验证失败 {file_path}: {e}")
        
        if repair_if_needed:
            return _try_repair_pdf(file_path)
        
        return False

@safe_pdf_operation
def _try_repair_pdf(file_path: str) -> bool:
    """尝试修复PDF文件"""
    try:
        temp_path = file_path + ".repaired.tmp"
        
        doc = fitz.open(file_path)
        
        doc.save(
            temp_path, 
            garbage=4,
            deflate=True,
            clean=True,
            ascii=False,
            linear=False,
            pretty=False,
            encryption=fitz.PDF_ENCRYPT_NONE
        )
        doc.close()
        
        if validate_pdf(temp_path, repair_if_needed=False):
            import shutil
            shutil.move(temp_path, file_path)
            logger.info(f"PDF修复成功: {file_path}")
            return True
        else:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False
            
    except Exception as e:
        logger.debug(f"PDF修复失败 {file_path}: {e}")
        return False

@safe_pdf_operation
def get_pdf_info(file_path: str) -> Dict[str, Any]:
    """安全地获取PDF基本信息"""
    info = {
        'page_count': 0,
        'file_size': 0,
        'is_valid': False,
        'has_text': False,
        'metadata': {},
        'error': None
    }
    
    try:
        info['file_size'] = os.path.getsize(file_path)
        
        if not validate_pdf(file_path):
            info['error'] = "PDF文件无效"
            return info
        
        doc = fitz.open(file_path)
        info['page_count'] = doc.page_count
        info['is_valid'] = True
        
        try:
            info['metadata'] = doc.metadata or {}
        except:
            info['metadata'] = {}
        
        try:
            if doc.page_count > 0:
                first_page = doc[0]
                sample_text = first_page.get_text()[:500]
                info['has_text'] = len(sample_text.strip()) > 0
        except:
            info['has_text'] = False
        
        doc.close()
        
    except Exception as e:
        info['error'] = str(e)
        logger.debug(f"获取PDF信息失败 {file_path}: {e}")
    
    return info

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

def scan_pdf_files(directory: str, recursive: bool = True, 
                   validate_files: bool = True) -> List[Dict[str, Any]]:
    """扫描目录中的PDF文件"""
    pdf_files = []
    
    try:
        directory_path = Path(directory)
        if not directory_path.exists():
            logger.error(f"目录不存在: {directory}")
            return pdf_files
        
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_paths = list(directory_path.glob(pattern))
        
        if not pdf_paths:
            logger.warning(f"在目录中未找到PDF文件: {directory}")
            return pdf_files
        
        logger.info(f"发现 {len(pdf_paths)} 个PDF文件")
        
        with ProgressTracker(len(pdf_paths), "扫描PDF文件") as progress:
            for pdf_path in pdf_paths:
                try:
                    file_info = {
                        'path': str(pdf_path),
                        'name': pdf_path.name,
                        'size': pdf_path.stat().st_size,
                        'size_formatted': format_file_size(pdf_path.stat().st_size)
                    }
                    
                    if validate_files:
                        pdf_info = get_pdf_info(str(pdf_path))
                        file_info.update(pdf_info)
                        
                        if pdf_info['is_valid']:
                            pdf_files.append(file_info)
                        else:
                            logger.debug(f"跳过无效PDF文件: {pdf_path}")
                    else:
                        pdf_files.append(file_info)
                    
                except Exception as e:
                    logger.debug(f"处理PDF文件失败 {pdf_path}: {e}")
                
                progress.update()
        
        logger.info(f"成功扫描 {len(pdf_files)} 个有效PDF文件")
        
    except Exception as e:
        logger.error(f"扫描PDF文件失败: {e}")
    
    return pdf_files

def batch_validate_pdfs(file_paths: List[str], repair_errors: bool = False) -> Dict[str, Any]:
    """批量验证PDF文件"""
    results = {
        'total_files': len(file_paths),
        'valid_files': [],
        'invalid_files': [],
        'repaired_files': [],
        'errors': []
    }
    
    if not file_paths:
        return results
    
    with ProgressTracker(len(file_paths), "验证PDF文件") as progress:
        for file_path in file_paths:
            try:
                is_valid = validate_pdf(file_path, repair_if_needed=repair_errors)
                
                if is_valid:
                    results['valid_files'].append(file_path)
                else:
                    results['invalid_files'].append(file_path)
                    
                    if repair_errors:
                        if _try_repair_pdf(file_path):
                            results['repaired_files'].append(file_path)
                            results['valid_files'].append(file_path)
                            results['invalid_files'].remove(file_path)
                
            except Exception as e:
                error_info = {'file': file_path, 'error': str(e)}
                results['errors'].append(error_info)
                logger.debug(f"验证PDF失败 {file_path}: {e}")
            
            progress.update()
    
    logger.info(f"PDF验证完成:")
    logger.info(f"  总文件数: {results['total_files']}")
    logger.info(f"  有效文件: {len(results['valid_files'])}")
    logger.info(f"  无效文件: {len(results['invalid_files'])}")
    logger.info(f"  修复文件: {len(results['repaired_files'])}")
    logger.info(f"  错误文件: {len(results['errors'])}")
    
    return results

def create_file_backup(file_path: str, backup_dir: str = None) -> Optional[str]:
    """创建文件备份"""
    try:
        if backup_dir is None:
            backup_dir = os.path.dirname(file_path)
        
        ensure_dir(backup_dir)
        
        file_name = os.path.basename(file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{timestamp}_{file_name}"
        backup_path = os.path.join(backup_dir, backup_name)
        
        import shutil
        shutil.copy2(file_path, backup_path)
        
        logger.debug(f"文件备份成功: {file_path} -> {backup_path}")
        return backup_path
        
    except Exception as e:
        logger.error(f"创建文件备份失败 {file_path}: {e}")
        return None

# ============================================================================
# 🚀 新增：全局管理器
# ============================================================================

class GlobalManager:
    """全局管理器，整合所有组件"""
    
    _instance = None
    
    def __new__(cls, config: Dict = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: Dict = None):
        if self._initialized:
            return
        
        self.config = config or load_config()
        self.performance_monitor = PerformanceMonitor(self.config)
        self.cache_manager = CacheManager(self.config)
        self.resource_monitor = ResourceMonitor(self.config)
        
        # 启动监控
        if self.config.get('logging', {}).get('monitoring', {}).get('enabled', False):
            self.performance_monitor.start_monitoring()
        
        self._initialized = True
        logger.info("全局管理器初始化完成")
    
    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'performance_monitor'):
            self.performance_monitor.stop_monitoring()
        logger.info("全局管理器清理完成")
    
    def get_stats(self) -> Dict:
        """获取所有统计信息"""
        return {
            'performance': self.performance_monitor.get_stats(),
            'cache': self.cache_manager.get_stats(),
            'system': self.resource_monitor.get_system_info()
        }

# 初始化
configure_pdf_error_suppression()

# 导出主要类和函数
__all__ = [
    'setup_logging', 'load_config', 'ensure_dir', 'calculate_file_hash',
    'clean_text', 'detect_language', 'save_results', 'load_results',
    'ProgressTracker', 'validate_pdf', 'get_pdf_info', 'scan_pdf_files',
    'PerformanceMonitor', 'CacheManager', 'ResourceMonitor', 'GlobalManager',
    'smart_file_validator', 'batch_file_processor', 'check_model_availability'
]
