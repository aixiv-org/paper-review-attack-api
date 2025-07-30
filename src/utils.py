import os
import yaml
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger
import hashlib
import re
import os
import json
import yaml
import hashlib
import warnings
import fitz
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger

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

# 🔧 改进的PDF处理函数

def configure_pdf_error_suppression():
    """配置PDF错误抑制"""
    # 抑制警告
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    # 抑制PyMuPDF的详细错误信息
    try:
        # 某些版本的PyMuPDF支持这个方法
        if hasattr(fitz, 'TOOLS'):
            fitz.TOOLS.mupdf_display_errors(False)
    except (AttributeError, Exception):
        pass
    
    # 设置环境变量来抑制MuPDF错误
    os.environ['MUPDF_DISPLAY_ERRORS'] = '0'

def safe_pdf_operation(func):
    """PDF操作的安全装饰器"""
    def wrapper(*args, **kwargs):
        try:
            # 抑制PDF处理过程中的警告和错误信息
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return func(*args, **kwargs)
        except Exception as e:
            logger.debug(f"PDF操作失败: {e}")
            return None
    return wrapper

@safe_pdf_operation
def validate_pdf(file_path: str, repair_if_needed: bool = True) -> bool:
    """
    增强的PDF验证函数
    
    Args:
        file_path: PDF文件路径
        repair_if_needed: 是否尝试修复损坏的PDF
    
    Returns:
        bool: PDF是否有效
    """
    if not os.path.exists(file_path):
        return False
    
    if os.path.getsize(file_path) == 0:
        return False
    
    try:
        # 抑制MuPDF的错误输出
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # 尝试打开PDF
            doc = fitz.open(file_path)
            
            # 基本检查
            if doc.page_count == 0:
                doc.close()
                return False
            
            # 尝试访问第一页
            try:
                page = doc[0]
                # 检查页面是否有效
                rect = page.rect
                if rect.width <= 0 or rect.height <= 0:
                    doc.close()
                    return False
                
                # 尝试获取少量文本内容（测试可读性）
                try:
                    text = page.get_text()[:100]  # 只获取前100个字符
                except:
                    pass  # 即使文本提取失败，PDF仍可能有效
                    
            except Exception as e:
                logger.debug(f"PDF页面访问失败 {file_path}: {e}")
                doc.close()
                return False
            
            doc.close()
            return True
            
    except Exception as e:
        logger.debug(f"PDF验证失败 {file_path}: {e}")
        
        # 如果需要修复，尝试使用其他方法
        if repair_if_needed:
            return _try_repair_pdf(file_path)
        
        return False

@safe_pdf_operation
def _try_repair_pdf(file_path: str) -> bool:
    """尝试修复PDF文件"""
    try:
        # 创建修复后的临时文件
        temp_path = file_path + ".repaired.tmp"
        
        # 使用PyMuPDF的修复功能
        doc = fitz.open(file_path)
        
        # 保存时进行清理和修复
        doc.save(
            temp_path, 
            garbage=4,      # 垃圾回收级别
            deflate=True,   # 压缩
            clean=True,     # 清理
            ascii=False,    # 允许非ASCII字符
            linear=False,   # 不线性化
            pretty=False,   # 不美化
            encryption=fitz.PDF_ENCRYPT_NONE  # 不加密
        )
        doc.close()
        
        # 验证修复后的文件
        if validate_pdf(temp_path, repair_if_needed=False):
            # 替换原文件
            import shutil
            shutil.move(temp_path, file_path)
            logger.info(f"PDF修复成功: {file_path}")
            return True
        else:
            # 清理临时文件
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
        
        # 获取元数据
        try:
            info['metadata'] = doc.metadata or {}
        except:
            info['metadata'] = {}
        
        # 检查是否有文本内容
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
    """
    扫描目录中的PDF文件
    
    Args:
        directory: 扫描目录
        recursive: 是否递归扫描子目录
        validate_files: 是否验证PDF文件有效性
    
    Returns:
        List[Dict]: PDF文件信息列表
    """
    pdf_files = []
    
    try:
        directory_path = Path(directory)
        if not directory_path.exists():
            logger.error(f"目录不存在: {directory}")
            return pdf_files
        
        # 查找PDF文件
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_paths = list(directory_path.glob(pattern))
        
        if not pdf_paths:
            logger.warning(f"在目录中未找到PDF文件: {directory}")
            return pdf_files
        
        logger.info(f"发现 {len(pdf_paths)} 个PDF文件")
        
        # 使用进度跟踪器
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
                        # 获取详细信息
                        pdf_info = get_pdf_info(str(pdf_path))
                        file_info.update(pdf_info)
                        
                        # 只保留有效的PDF文件
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
    """
    批量验证PDF文件
    
    Args:
        file_paths: PDF文件路径列表
        repair_errors: 是否尝试修复错误的PDF
    
    Returns:
        Dict: 验证结果统计
    """
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
                        # 尝试修复
                        if _try_repair_pdf(file_path):
                            results['repaired_files'].append(file_path)
                            results['valid_files'].append(file_path)
                            results['invalid_files'].remove(file_path)
                
            except Exception as e:
                error_info = {'file': file_path, 'error': str(e)}
                results['errors'].append(error_info)
                logger.debug(f"验证PDF失败 {file_path}: {e}")
            
            progress.update()
    
    # 打印统计信息
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

# 初始化PDF错误抑制
configure_pdf_error_suppression()
