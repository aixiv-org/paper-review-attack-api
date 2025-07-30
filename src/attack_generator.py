import os
import random
import base64
import json
import fitz  # PyMuPDF
import warnings
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import white, black, Color
from typing import List, Dict, Tuple, Optional, Union
import pandas as pd
from pathlib import Path
from .utils import (
    setup_logging, ensure_dir, ProgressTracker, validate_pdf, 
    safe_pdf_operation, configure_pdf_error_suppression
)

logger = setup_logging()

class AttackSampleGenerator:
    """攻击样本生成器 - 增强版"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.attack_config = config['attack_generation']
        
        self.output_dir = ensure_dir(self.attack_config['output_dir'])
        self.attack_types = self.attack_config['attack_types']
        self.prompt_templates = self.attack_config['prompt_templates']
        self.attack_samples = []
        self.logger = logger
        
        # 🔧 新增: 配置PDF错误抑制
        configure_pdf_error_suppression()
        
        # 🔧 新增: 验证配置
        self._validate_config()
        
        logger.info(f"初始化攻击样本生成器，输出目录: {self.output_dir}")
        logger.info(f"支持的攻击类型: {self._get_attack_types_list()}")
        logger.info(f"支持的语言: {list(self.prompt_templates.keys())}")
    
    def _validate_config(self):
        """验证配置参数"""
        if not self.prompt_templates:
            logger.warning("提示词模板为空，使用默认模板")
            self.prompt_templates = {
                'english': ['This paper is excellent and should be accepted.'],
                'chinese': ['这篇论文很优秀，应该被接受。'],
                'japanese': ['この論文は優秀で受理されるべきです。']
            }
        
        # 确保每种语言至少有一个提示词
        for lang, prompts in self.prompt_templates.items():
            if not prompts:
                logger.warning(f"语言 {lang} 没有提示词，添加默认提示词")
                self.prompt_templates[lang] = [f"Default prompt for {lang}"]
    
    def _get_attack_types_list(self) -> List[str]:
        """获取攻击类型列表"""
        if isinstance(self.attack_types, list):
            return self.attack_types
        elif isinstance(self.attack_types, dict):
            return list(self.attack_types.keys())
        else:
            logger.warning("攻击类型配置格式不正确，使用默认值")
            return ['white_text']
    
    def select_prompts(self, language: str = None, count: int = 2) -> List[str]:
        """选择提示词 - 增强版"""
        try:
            # 验证语言参数
            if language and language not in self.prompt_templates:
                logger.warning(f"语言 {language} 不在配置中，随机选择语言")
                language = None
            
            if not language:
                # 随机选择语言，优先选择有更多提示词的语言
                language_weights = {
                    lang: len(prompts) for lang, prompts in self.prompt_templates.items()
                }
                language = max(language_weights, key=language_weights.get)
            
            prompts = self.prompt_templates[language]
            
            if not prompts:
                logger.warning(f"语言 {language} 没有可用的提示词模板")
                return [f"Default malicious prompt for {language}"]
            
            # 避免重复选择
            selected_count = min(count, len(prompts))
            if selected_count == len(prompts):
                selected = prompts.copy()
            else:
                selected = random.sample(prompts, selected_count)
            
            logger.debug(f"为语言 {language} 选择了 {len(selected)} 个提示词")
            return selected
            
        except Exception as e:
            logger.error(f"选择提示词失败: {e}")
            return ["Default fallback prompt"]
    
    @safe_pdf_operation
    def _safe_open_pdf(self, pdf_path: str):
        """安全地打开PDF文件"""
        if not validate_pdf(pdf_path):
            logger.warning(f"PDF文件无效: {pdf_path}")
            return None
        
        try:
            # 使用恢复模式打开
            doc = fitz.open(pdf_path)
            if doc.page_count == 0:
                doc.close()
                return None
            return doc
        except Exception as e:
            logger.warning(f"无法打开PDF文件 {pdf_path}: {e}")
            return None
    
    @safe_pdf_operation
    def inject_white_text(self, input_pdf: str, output_pdf: str, prompts: List[str]) -> bool:
        """注入白色字体攻击 - 增强版"""
        doc = None
        try:
            doc = self._safe_open_pdf(input_pdf)
            if doc is None:
                return False
            
            # 限制处理页数
            max_pages = min(3, doc.page_count)
            injected_count = 0
            
            for page_num in range(max_pages):
                try:
                    page = doc[page_num]
                    rect = page.rect
                    
                    # 验证页面有效性
                    if rect.width <= 0 or rect.height <= 0:
                        continue
                    
                    # 为每个提示词选择不同的注入策略
                    for i, prompt in enumerate(prompts[:3]):  # 最多3个提示词
                        if not prompt or len(prompt.strip()) == 0:
                            continue
                        
                        try:
                            # 策略1: 随机位置注入
                            if i == 0:
                                x = random.uniform(50, max(51, rect.width - 200))
                                y = random.uniform(50, max(51, rect.height - 50))
                                fontsize = random.uniform(0.5, 1.5)
                                
                                page.insert_text(
                                    (x, y),
                                    prompt[:100],  # 限制长度
                                    fontsize=fontsize,
                                    color=(1, 1, 1),  # 纯白色
                                    fontname="helv"
                                )
                                injected_count += 1
                            
                            # 策略2: 页面边缘注入
                            elif i == 1 and rect.width > 200:
                                edge_positions = [
                                    (rect.width - 150, random.uniform(20, rect.height - 20)),  # 右边缘
                                    (random.uniform(20, rect.width - 20), rect.height - 15),   # 底边缘
                                    (10, random.uniform(20, rect.height - 20))                 # 左边缘
                                ]
                                
                                pos_x, pos_y = random.choice(edge_positions)
                                page.insert_text(
                                    (pos_x, pos_y),
                                    prompt[:50],  # 更短的文本适应边缘
                                    fontsize=0.1,
                                    color=(1, 1, 1),
                                    fontname="helv"
                                )
                                injected_count += 1
                            
                            # 策略3: 近白色注入
                            elif i == 2:
                                x = random.uniform(100, max(101, rect.width - 100))
                                y = random.uniform(100, max(101, rect.height - 100))
                                
                                # 使用非常接近白色的颜色
                                near_white_colors = [
                                    (0.99, 0.99, 0.99),
                                    (1.0, 0.99, 0.99),
                                    (0.99, 1.0, 0.99),
                                    (0.99, 0.99, 1.0)
                                ]
                                
                                page.insert_text(
                                    (x, y),
                                    prompt[:80],
                                    fontsize=random.uniform(0.8, 1.2),
                                    color=random.choice(near_white_colors),
                                    fontname="helv"
                                )
                                injected_count += 1
                                
                        except Exception as e:
                            logger.debug(f"页面 {page_num} 注入失败: {e}")
                            continue
                            
                except Exception as e:
                    logger.debug(f"处理页面 {page_num} 失败: {e}")
                    continue
            
            if injected_count == 0:
                logger.warning("未能成功注入任何白色文本")
                if doc:
                    doc.close()
                return False
            
            # 安全保存
            try:
                doc.save(output_pdf, garbage=4, deflate=True, clean=True)
                doc.close()
                
                if validate_pdf(output_pdf):
                    logger.debug(f"白色字体攻击注入成功: {output_pdf} (注入 {injected_count} 处)")
                    return True
                else:
                    logger.warning(f"生成的PDF无效: {output_pdf}")
                    if os.path.exists(output_pdf):
                        os.remove(output_pdf)
                    return False
                    
            except Exception as e:
                logger.error(f"保存PDF失败 {output_pdf}: {e}")
                if doc:
                    doc.close()
                return False
            
        except Exception as e:
            logger.error(f"白色字体攻击注入失败 {input_pdf}: {e}")
            if doc:
                try:
                    doc.close()
                except:
                    pass
            return False
    
    @safe_pdf_operation
    def inject_metadata_attack(self, input_pdf: str, output_pdf: str, prompts: List[str]) -> bool:
        """注入元数据攻击 - 增强版"""
        doc = None
        try:
            doc = self._safe_open_pdf(input_pdf)
            if doc is None:
                return False
            
            # 安全获取元数据
            try:
                metadata = doc.metadata.copy() if doc.metadata else {}
            except:
                metadata = {}
            
            # 策略性注入元数据
            try:
                if len(prompts) >= 1:
                    # 主题字段
                    metadata['subject'] = prompts[0][:200]
                
                if len(prompts) >= 2:
                    # 关键词字段
                    keywords = [prompt[:50] for prompt in prompts[:3]]
                    metadata['keywords'] = "; ".join(keywords)[:500]
                
                if len(prompts) >= 3:
                    # 创建者字段
                    metadata['creator'] = f"Academic Publisher with {prompts[2][:50]}"
                
                # 标题字段 - 更隐蔽的注入
                original_title = metadata.get('title', 'Research Paper')
                if prompts:
                    # 在标题末尾添加看似正常的内容
                    suffix_templates = [
                        f": {prompts[0][:30]}",
                        f" - {prompts[0][:40]}",
                        f" ({prompts[0][:25]})",
                        f" with {prompts[0][:35]}"
                    ]
                    suffix = random.choice(suffix_templates)
                    metadata['title'] = (original_title + suffix)[:300]
                
                # 添加自定义字段
                if len(prompts) > 1:
                    metadata['custom_field'] = prompts[1][:100]
                
                doc.set_metadata(metadata)
                logger.debug(f"成功注入 {len([k for k, v in metadata.items() if v])} 个元数据字段")
                
            except Exception as e:
                logger.warning(f"设置元数据失败: {e}")
                # 即使元数据注入失败，也尝试保存原文件
            
            # 保存文件
            try:
                doc.save(output_pdf, garbage=4, deflate=True, clean=True)
                doc.close()
                
                if validate_pdf(output_pdf):
                    logger.debug(f"元数据攻击注入成功: {output_pdf}")
                    return True
                else:
                    if os.path.exists(output_pdf):
                        os.remove(output_pdf)
                    return False
                    
            except Exception as e:
                logger.error(f"保存PDF失败 {output_pdf}: {e}")
                if doc:
                    doc.close()
                return False
            
        except Exception as e:
            logger.error(f"元数据攻击注入失败 {input_pdf}: {e}")
            if doc:
                try:
                    doc.close()
                except:
                    pass
            return False
    
    @safe_pdf_operation
    def inject_invisible_chars(self, input_pdf: str, output_pdf: str, prompts: List[str]) -> bool:
        """注入不可见字符攻击 - 增强版"""
        doc = None
        try:
            doc = self._safe_open_pdf(input_pdf)
            if doc is None:
                return False
            
            # 扩展的不可见字符集
            invisible_chars = [
                '\u200b',  # 零宽空格
                '\u200c',  # 零宽非连接符
                '\u200d',  # 零宽连接符
                '\ufeff',  # 字节顺序标记
                '\u2060',  # 单词连接符
                '\u180e',  # 蒙古文元音分隔符
                '\u061c',  # 阿拉伯字母标记
                '\u2061',  # 函数应用
                '\u2062',  # 不可见乘法
                '\u2063',  # 不可见分隔符
                '\u2064'   # 不可见加法
            ]
            
            max_pages = min(2, doc.page_count)
            injected_count = 0
            
            for page_num in range(max_pages):
                try:
                    page = doc[page_num]
                    rect = page.rect
                    
                    if rect.width <= 0 or rect.height <= 0:
                        continue
                    
                    for i, prompt in enumerate(prompts[:2]):  # 每页最多2个提示词
                        if not prompt:
                            continue
                        
                        try:
                            # 策略1: 字符间插入不可见字符
                            if i == 0:
                                encoded_prompt = ""
                                for char in prompt[:50]:  # 限制长度
                                    encoded_prompt += char
                                    if random.random() < 0.4:  # 40%概率插入
                                        encoded_prompt += random.choice(invisible_chars)
                                
                                x = random.uniform(100, max(101, rect.width - 200))
                                y = random.uniform(100, max(101, rect.height - 100))
                                
                                page.insert_text(
                                    (x, y),
                                    encoded_prompt,
                                    fontsize=random.uniform(0.1, 0.8),
                                    color=(0.98, 0.98, 0.98),
                                    fontname="helv"
                                )
                                injected_count += 1
                            
                            # 策略2: 完全不可见字符编码
                            else:
                                # 将提示词转换为不可见字符序列
                                invisible_text = ""
                                for char in prompt[:30]:
                                    # 使用字符的ASCII值选择不可见字符
                                    char_code = ord(char) % len(invisible_chars)
                                    invisible_text += invisible_chars[char_code]
                                    invisible_text += random.choice(invisible_chars)
                                
                                x = random.uniform(50, max(51, rect.width - 100))
                                y = random.uniform(50, max(51, rect.height - 50))
                                
                                page.insert_text(
                                    (x, y),
                                    invisible_text,
                                    fontsize=0.1,
                                    color=(1, 1, 1),  # 白色
                                    fontname="helv"
                                )
                                injected_count += 1
                                
                        except Exception as e:
                            logger.debug(f"页面 {page_num} 不可见字符注入失败: {e}")
                            continue
                            
                except Exception as e:
                    logger.debug(f"处理页面 {page_num} 失败: {e}")
                    continue
            
            if injected_count == 0:
                logger.warning("未能成功注入任何不可见字符")
                if doc:
                    doc.close()
                return False
            
            # 保存文件
            try:
                doc.save(output_pdf, garbage=4, deflate=True, clean=True)
                doc.close()
                
                if validate_pdf(output_pdf):
                    logger.debug(f"不可见字符攻击注入成功: {output_pdf} (注入 {injected_count} 处)")
                    return True
                else:
                    if os.path.exists(output_pdf):
                        os.remove(output_pdf)
                    return False
                    
            except Exception as e:
                logger.error(f"保存PDF失败: {e}")
                if doc:
                    doc.close()
                return False
            
        except Exception as e:
            logger.error(f"不可见字符攻击注入失败 {input_pdf}: {e}")
            if doc:
                try:
                    doc.close()
                except:
                    pass
            return False
    
    @safe_pdf_operation
    def inject_mixed_language_attack(self, input_pdf: str, output_pdf: str, prompts: List[str]) -> bool:
        """注入混合语言攻击 - 增强版"""
        doc = None
        try:
            doc = self._safe_open_pdf(input_pdf)
            if doc is None:
                return False
            
            # 构建混合语言提示词
            mixed_prompts = []
            
            # 从每种语言获取提示词
            for lang in ['english', 'chinese', 'japanese']:
                if lang in self.prompt_templates and self.prompt_templates[lang]:
                    lang_prompts = self.prompt_templates[lang]
                    selected = random.choice(lang_prompts)
                    mixed_prompts.append(selected)
            
            # 如果没有混合语言，使用原始提示词
            if not mixed_prompts:
                mixed_prompts = prompts[:2]
            
            # 创建不同的混合策略
            mixed_strategies = []
            
            if len(mixed_prompts) >= 2:
                # 策略1: 简单连接
                mixed_strategies.append(f"{mixed_prompts[0]} {mixed_prompts[1]}")
                
                # 策略2: 句子内混合
                words1 = mixed_prompts[0].split()[:3]
                words2 = mixed_prompts[1].split()[:3]
                mixed_words = []
                for w1, w2 in zip(words1, words2):
                    mixed_words.extend([w1, w2])
                mixed_strategies.append(" ".join(mixed_words))
                
                # 策略3: 标点符号分隔
                mixed_strategies.append(f"{mixed_prompts[0]}, {mixed_prompts[1]}")
            
            if mixed_strategies:
                selected_strategy = random.choice(mixed_strategies)
            else:
                selected_strategy = " ".join(mixed_prompts) if mixed_prompts else "Mixed language test"
            
            # 注入到PDF
            max_pages = min(2, doc.page_count)
            injected_count = 0
            
            for page_num in range(max_pages):
                try:
                    page = doc[page_num]
                    rect = page.rect
                    
                    if rect.width <= 0 or rect.height <= 0:
                        continue
                    
                    # 选择注入位置和样式
                    positions = [
                        (random.uniform(50, rect.width - 300), random.uniform(50, rect.height - 50)),
                        (random.uniform(rect.width * 0.1, rect.width * 0.9), rect.height - 30),
                        (rect.width - 200, random.uniform(20, rect.height - 20))
                    ]
                    
                    colors = [
                        (0.95, 0.95, 0.95),  # 浅灰色
                        (0.98, 0.98, 0.98),  # 很浅的灰色
                        (0.92, 0.95, 0.92)   # 淡绿色
                    ]
                    
                    for pos, color in zip(positions[:1], colors[:1]):  # 每页一个位置
                        try:
                            page.insert_text(
                                pos,
                                selected_strategy[:120],  # 限制长度
                                fontsize=random.uniform(0.8, 1.5),
                                color=color,
                                fontname="helv"
                            )
                            injected_count += 1
                            break  # 成功注入后跳出
                            
                        except Exception as e:
                            logger.debug(f"位置 {pos} 注入失败: {e}")
                            continue
                            
                except Exception as e:
                    logger.debug(f"页面 {page_num} 处理失败: {e}")
                    continue
            
            if injected_count == 0:
                logger.warning("未能成功注入任何混合语言文本")
                if doc:
                    doc.close()
                return False
            
            # 保存文件
            try:
                doc.save(output_pdf, garbage=4, deflate=True, clean=True)
                doc.close()
                
                if validate_pdf(output_pdf):
                    logger.debug(f"混合语言攻击注入成功: {output_pdf} (注入 {injected_count} 处)")
                    return True
                else:
                    if os.path.exists(output_pdf):
                        os.remove(output_pdf)
                    return False
                    
            except Exception as e:
                logger.error(f"保存PDF失败: {e}")
                if doc:
                    doc.close()
                return False
            
        except Exception as e:
            logger.error(f"混合语言攻击注入失败 {input_pdf}: {e}")
            if doc:
                try:
                    doc.close()
                except:
                    pass
            return False
    
    @safe_pdf_operation
    def inject_steganographic_attack(self, input_pdf: str, output_pdf: str, prompts: List[str]) -> bool:
        """注入隐写术攻击 - 增强版"""
        doc = None
        try:
            doc = self._safe_open_pdf(input_pdf)
            if doc is None:
                return False
            
            # 多种编码策略
            encoded_prompts = []
            
            for prompt in prompts[:3]:  # 最多处理3个提示词
                if not prompt:
                    continue
                
                try:
                    # 策略1: Base64编码
                    base64_encoded = base64.b64encode(prompt.encode('utf-8')).decode('ascii')
                    encoded_prompts.append(('base64', base64_encoded))
                    
                    # 策略2: 十六进制编码
                    hex_encoded = prompt.encode('utf-8').hex()
                    encoded_prompts.append(('hex', hex_encoded))
                    
                    # 策略3: URL编码风格
                    url_encoded = prompt.replace(' ', '%20').replace(',', '%2C')
                    encoded_prompts.append(('url', url_encoded))
                    
                except Exception as e:
                    logger.debug(f"编码提示词失败: {e}")
                    encoded_prompts.append(('plain', prompt))
            
            if not encoded_prompts:
                logger.warning("没有成功编码的提示词")
                if doc:
                    doc.close()
                return False
            
            # 注入编码后的内容
            max_pages = min(2, doc.page_count)
            injected_count = 0
            
            for page_num in range(max_pages):
                try:
                    page = doc[page_num]
                    rect = page.rect
                    
                    if rect.width <= 0 or rect.height <= 0:
                        continue
                    
                    for i, (encoding_type, encoded_content) in enumerate(encoded_prompts[:2]):
                        try:
                            # 根据编码类型选择伪装策略
                            if encoding_type == 'base64':
                                # 伪装成图片引用
                                fake_content = f"data:image/png;base64,{encoded_content[:40]}"
                                fontsize = 6.0
                                color = (0.8, 0.8, 0.8)
                                
                            elif encoding_type == 'hex':
                                # 伪装成校验和
                                fake_content = f"checksum: {encoded_content[:32]}"
                                fontsize = 5.0
                                color = (0.85, 0.85, 0.85)
                                
                            elif encoding_type == 'url':
                                # 伪装成URL
                                fake_content = f"https://api.example.com/v1/{encoded_content[:30]}"
                                fontsize = 7.0
                                color = (0.7, 0.7, 0.9)  # 蓝色调，像链接
                                
                            else:
                                # 直接文本，伪装成注释
                                fake_content = f"<!-- {encoded_content[:50]} -->"
                                fontsize = 4.0
                                color = (0.9, 0.9, 0.9)
                            
                            # 选择合适的位置
                            if i == 0:
                                # 底部区域，像页脚
                                x = random.uniform(50, rect.width - 200)
                                y = random.uniform(10, 30)
                            else:
                                # 边缘区域
                                x = rect.width - 150
                                y = random.uniform(rect.height - 50, rect.height - 10)
                            
                            page.insert_text(
                                (x, y),
                                fake_content,
                                fontsize=fontsize,
                                color=color,
                                fontname="helv"
                            )
                            injected_count += 1
                            
                        except Exception as e:
                            logger.debug(f"注入编码内容失败: {e}")
                            continue
                            
                except Exception as e:
                    logger.debug(f"页面 {page_num} 处理失败: {e}")
                    continue
            
            if injected_count == 0:
                logger.warning("未能成功注入任何隐写内容")
                if doc:
                    doc.close()
                return False
            
            # 保存文件
            try:
                doc.save(output_pdf, garbage=4, deflate=True, clean=True)
                doc.close()
                
                if validate_pdf(output_pdf):
                    logger.debug(f"隐写术攻击注入成功: {output_pdf} (注入 {injected_count} 处)")
                    return True
                else:
                    if os.path.exists(output_pdf):
                        os.remove(output_pdf)
                    return False
                    
            except Exception as e:
                logger.error(f"保存PDF失败: {e}")
                if doc:
                    doc.close()
                return False
            
        except Exception as e:
            logger.error(f"隐写术攻击注入失败 {input_pdf}: {e}")
            if doc:
                try:
                    doc.close()
                except:
                    pass
            return False
    
    def generate_single_attack(self, input_pdf: str, attack_type: str, 
                             language: str = None) -> Optional[str]:
        """生成单个攻击样本 - 增强版"""
        # 预验证
        if not validate_pdf(input_pdf):
            logger.warning(f"输入PDF无效: {input_pdf}")
            return None
        
        try:
            # 生成安全的输出文件名
            base_name = os.path.splitext(os.path.basename(input_pdf))[0]
            # 清理文件名中的特殊字符
            safe_base_name = "".join(c for c in base_name if c.isalnum() or c in "._-")[:80]
            safe_language = (language or 'mixed').replace('/', '_')[:20]
            safe_attack_type = attack_type.replace('/', '_')[:30]
            
            output_name = f"{safe_base_name}_{safe_attack_type}_{safe_language}.pdf"
            output_pdf = os.path.join(self.output_dir, output_name)
            
            # 避免文件名冲突
            counter = 1
            while os.path.exists(output_pdf):
                base_output = os.path.splitext(output_pdf)[0]
                output_pdf = f"{base_output}_{counter}.pdf"
                counter += 1
            
            # 选择提示词
            prompts = self.select_prompts(language, count=3)
            if not prompts:
                logger.warning("无法获取提示词，使用默认值")
                prompts = [f"Default {attack_type} prompt"]
            
            # 执行攻击注入
            success = False
            injection_methods = {
                "white_text": self.inject_white_text,
                "metadata": self.inject_metadata_attack,
                "invisible_chars": self.inject_invisible_chars,
                "mixed_language": self.inject_mixed_language_attack,
                "steganographic": self.inject_steganographic_attack
            }
            
            if attack_type in injection_methods:
                try:
                    success = injection_methods[attack_type](input_pdf, output_pdf, prompts)
                except Exception as e:
                    logger.error(f"{attack_type} 攻击注入过程失败: {e}")
                    success = False
            else:
                logger.warning(f"未知攻击类型: {attack_type}")
                return None
            
            # 验证结果
            if success and os.path.exists(output_pdf) and validate_pdf(output_pdf):
                # 记录攻击信息
                try:
                    file_size = os.path.getsize(output_pdf)
                    attack_info = {
                        'original_file': input_pdf,
                        'attack_file': output_pdf,
                        'attack_type': attack_type,
                        'language': language or 'mixed',
                        'prompts_used': prompts,
                        'file_size': file_size,
                        'timestamp': pd.Timestamp.now().isoformat(),
                        'file_size_mb': round(file_size / (1024*1024), 3)
                    }
                    self.attack_samples.append(attack_info)
                    
                    logger.debug(f"攻击样本生成成功: {output_pdf}")
                    return output_pdf
                    
                except Exception as e:
                    logger.error(f"记录攻击信息失败: {e}")
                    return output_pdf  # 即使记录失败，文件生成成功也返回路径
            else:
                logger.debug(f"攻击样本生成失败: {output_pdf}")
                # 清理失败的文件
                if os.path.exists(output_pdf):
                    try:
                        os.remove(output_pdf)
                    except:
                        pass
                return None
                
        except Exception as e:
            logger.error(f"生成攻击样本时发生错误 {input_pdf}: {e}")
            return None
    
    def generate_attack_samples(self, clean_pdfs: List[str]) -> List[str]:
        """批量生成攻击样本 - 增强版"""
        if not clean_pdfs:
            logger.warning("没有输入的PDF文件")
            return []
        
        # 过滤有效的PDF文件
        logger.info("验证输入PDF文件...")
        valid_pdfs = []
        invalid_count = 0
        
        with ProgressTracker(len(clean_pdfs), "验证PDF文件") as progress:
            for pdf_path in clean_pdfs:
                if validate_pdf(pdf_path):
                    valid_pdfs.append(pdf_path)
                else:
                    invalid_count += 1
                    logger.debug(f"跳过无效PDF: {pdf_path}")
                progress.update()
        
        if invalid_count > 0:
            logger.warning(f"跳过了 {invalid_count} 个无效PDF文件")
        
        if not valid_pdfs:
            logger.error("没有有效的PDF文件可用于生成攻击样本")
            return []
        
        logger.info(f"有效PDF文件数: {len(valid_pdfs)}")
        
        # 计算攻击样本数量
        attack_ratio = self.attack_config.get('attack_ratio', 0.3)
        total_attack_samples = max(1, int(len(valid_pdfs) * attack_ratio))
        
        logger.info(f"计划生成 {total_attack_samples} 个攻击样本 (比例: {attack_ratio:.1%})")
        
        # 获取攻击类型配置
        attack_types = self._get_attack_types_list()
        if not attack_types:
            logger.error("没有配置攻击类型")
            return []
        
        # 根据生成策略选择方法
        generation_strategy = self.attack_config.get('generation_strategy', {})
        mode = generation_strategy.get('mode', 'random')
        
        if mode == 'proportional' and isinstance(self.attack_types, dict):
            return self._generate_proportional_samples(valid_pdfs, total_attack_samples, self.attack_types)
        else:
            return self._generate_random_samples(valid_pdfs, total_attack_samples, attack_types)
    
    def _generate_random_samples(self, clean_pdfs: List[str], total_samples: int, 
                               attack_types: List[str]) -> List[str]:
        """随机生成攻击样本"""
        generated_samples = []
        failed_attempts = 0
        max_failures = total_samples // 2  # 允许的最大失败次数
        
        # 随机选择PDF文件（允许重复使用）
        available_pdfs = clean_pdfs.copy()
        
        with ProgressTracker(total_samples, "生成攻击样本") as progress:
            for i in range(total_samples):
                try:
                    # 随机选择PDF文件
                    if not available_pdfs:
                        available_pdfs = clean_pdfs.copy()  # 重新填充列表
                    
                    pdf_path = random.choice(available_pdfs)
                    available_pdfs.remove(pdf_path)  # 避免短期重复
                    
                    # 随机选择攻击类型和语言
                    attack_type = random.choice(attack_types)
                    language = random.choice(list(self.prompt_templates.keys()))
                    
                    output_path = self.generate_single_attack(pdf_path, attack_type, language)
                    
                    if output_path:
                        generated_samples.append(output_path)
                        failed_attempts = 0  # 重置失败计数
                    else:
                        failed_attempts += 1
                        logger.debug(f"生成失败 {pdf_path} -> {attack_type}")
                        
                        if failed_attempts >= max_failures:
                            logger.warning(f"连续失败次数过多 ({failed_attempts})，停止生成")
                            break
                    
                    progress.update()
                    
                except Exception as e:
                    logger.error(f"生成攻击样本失败 {pdf_path}: {e}")
                    failed_attempts += 1
                    progress.update()
                    
                    if failed_attempts >= max_failures:
                        break
        
        # 保存攻击样本信息
        self.save_attack_info()
        
        success_rate = len(generated_samples) / total_samples if total_samples > 0 else 0
        logger.info(f"攻击样本生成完成: {len(generated_samples)}/{total_samples} ({success_rate:.1%} 成功率)")
        
        return generated_samples
    
    def _generate_proportional_samples(self, clean_pdfs: List[str], total_samples: int, 
                                     attack_proportions: Dict[str, float]) -> List[str]:
        """按比例生成攻击样本 - 增强版"""
        generation_strategy = self.attack_config.get('generation_strategy', {})
        min_samples = generation_strategy.get('min_samples_per_type', 1)
        max_samples = generation_strategy.get('max_samples_per_type', total_samples)
        
        # 计算每种攻击类型的目标数量
        target_counts = {}
        remaining_samples = total_samples
        
        # 首先分配最小样本数
        for attack_type in attack_proportions:
            target_counts[attack_type] = min_samples
            remaining_samples -= min_samples
        
        # 按比例分配剩余样本
        if remaining_samples > 0:
            for attack_type, proportion in attack_proportions.items():
                additional = int(remaining_samples * proportion)
                target_counts[attack_type] += additional
                # 确保不超过最大值
                target_counts[attack_type] = min(target_counts[attack_type], max_samples)
        
        # 调整总数
        actual_total = sum(target_counts.values())
        if actual_total > total_samples:
            # 按比例减少
            scale_factor = total_samples / actual_total
            for attack_type in target_counts:
                target_counts[attack_type] = max(min_samples, 
                                               int(target_counts[attack_type] * scale_factor))
        
        logger.info("攻击样本生成计划:")
        for attack_type, count in target_counts.items():
            percentage = (count / sum(target_counts.values()) * 100) if sum(target_counts.values()) > 0 else 0
            logger.info(f"  {attack_type}: {count} ({percentage:.1f}%)")
        
        # 生成样本
        generated_samples = []
        attack_type_counts = {attack_type: 0 for attack_type in target_counts}
        available_pdfs = clean_pdfs.copy()
        
        total_planned = sum(target_counts.values())
        with ProgressTracker(total_planned, "按比例生成攻击样本") as progress:
            
            for attack_type, target_count in target_counts.items():
                type_generated = 0
                attempts = 0
                max_attempts = target_count * 3  # 允许一定的失败重试
                
                while type_generated < target_count and attempts < max_attempts:
                    try:
                        # 选择PDF文件
                        if not available_pdfs:
                            available_pdfs = clean_pdfs.copy()
                        
                        pdf_path = random.choice(available_pdfs)
                        available_pdfs.remove(pdf_path)
                        
                        # 选择语言
                        language = random.choice(list(self.prompt_templates.keys()))
                        
                        output_path = self.generate_single_attack(pdf_path, attack_type, language)
                        
                        if output_path:
                            generated_samples.append(output_path)
                            attack_type_counts[attack_type] += 1
                            type_generated += 1
                        
                        attempts += 1
                        progress.update()
                        
                    except Exception as e:
                        logger.error(f"生成 {attack_type} 攻击样本失败: {e}")
                        attempts += 1
                        progress.update()
                        continue
                
                if type_generated < target_count:
                    logger.warning(f"{attack_type} 类型只生成了 {type_generated}/{target_count} 个样本")
        
        # 保存攻击样本信息
        self.save_attack_info()
        
        # 打印实际生成统计
        logger.info("攻击样本实际生成统计:")
        total_generated = sum(attack_type_counts.values())
        for attack_type, count in attack_type_counts.items():
            percentage = (count / total_generated * 100) if total_generated > 0 else 0
            target = target_counts.get(attack_type, 0)
            success_rate = (count / target * 100) if target > 0 else 0
            logger.info(f"  {attack_type}: {count}/{target} ({percentage:.1f}%, {success_rate:.1f}% 成功率)")
        
        return generated_samples
    
    def save_attack_info(self):
        """保存攻击样本信息 - 增强版"""
        if not self.attack_samples:
            logger.debug("没有攻击样本信息需要保存")
            return
        
        try:
            timestamp = pd.Timestamp.now()
            
            # 保存为JSON格式（程序读取）
            json_file = os.path.join(self.output_dir, "attack_samples_info.json")
            attack_summary = {
                'metadata': {
                    'total_samples': len(self.attack_samples),
                    'generation_time': timestamp.isoformat(),
                    'generator_version': '2.0',
                    'config_hash': hash(str(sorted(self.attack_config.items())))
                },
                'generation_config': {
                    'attack_types': self.attack_types,
                    'attack_ratio': self.attack_config.get('attack_ratio', 0.3),
                    'languages': list(self.prompt_templates.keys()),
                    'output_dir': self.output_dir
                },
                'statistics': self.get_attack_statistics(),
                'attack_samples': self.attack_samples
            }
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(attack_summary, f, indent=2, ensure_ascii=False, default=str)
            
            # 保存为CSV格式（人工查看）
            csv_file = os.path.join(self.output_dir, "attack_samples_info.csv")
            df = pd.DataFrame(self.attack_samples)
            df.to_csv(csv_file, index=False, encoding='utf-8')
            
            # 保存简要统计
            stats_file = os.path.join(self.output_dir, "generation_summary.txt")
            with open(stats_file, 'w', encoding='utf-8') as f:
                f.write(f"攻击样本生成摘要\n")
                f.write(f"={'='*50}\n")
                f.write(f"生成时间: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总样本数: {len(self.attack_samples)}\n")
                f.write(f"输出目录: {self.output_dir}\n\n")
                
                stats = self.get_attack_statistics()
                f.write(f"攻击类型分布:\n")
                for attack_type, count in stats['attack_types'].items():
                    percentage = count / stats['total_attacks'] * 100
                    f.write(f"  {attack_type}: {count} ({percentage:.1f}%)\n")
                
                f.write(f"\n语言分布:\n")
                for language, count in stats['languages'].items():
                    percentage = count / stats['total_attacks'] * 100
                    f.write(f"  {language}: {count} ({percentage:.1f}%)\n")
                
                f.write(f"\n总文件大小: {stats['total_size_mb']:.2f} MB\n")
            
            logger.info(f"攻击样本信息已保存: {json_file}, {csv_file}, {stats_file}")
            
        except Exception as e:
            logger.error(f"保存攻击样本信息失败: {e}")
    
    def load_attack_info(self) -> pd.DataFrame:
        """加载攻击样本信息"""
        csv_file = os.path.join(self.output_dir, "attack_samples_info.csv")
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file, encoding='utf-8')
                logger.info(f"加载了 {len(df)} 条攻击样本记录")
                return df
            except Exception as e:
                logger.error(f"加载攻击样本信息失败: {e}")
                return pd.DataFrame()
        else:
            logger.debug("攻击样本信息文件不存在")
            return pd.DataFrame()
    
    def get_attack_statistics(self) -> Dict:
        """获取攻击样本统计信息 - 增强版"""
        if not self.attack_samples:
            df = self.load_attack_info()
        else:
            df = pd.DataFrame(self.attack_samples)
        
        if df.empty:
            return {
                'total_attacks': 0,
                'attack_types': {},
                'languages': {},
                'total_size_mb': 0,
                'avg_size_mb': 0,
                'success_rate': 0
            }
        
        # 基础统计
        stats = {
            'total_attacks': len(df),
            'attack_types': df['attack_type'].value_counts().to_dict(),
            'languages': df['language'].value_counts().to_dict(),
        }
        
        # 文件大小统计
        if 'file_size' in df.columns:
            total_size_bytes = df['file_size'].sum()
            stats['total_size_mb'] = round(total_size_bytes / (1024*1024), 2)
            stats['avg_size_mb'] = round(total_size_bytes / len(df) / (1024*1024), 3)
            stats['size_stats'] = {
                'min_size_mb': round(df['file_size'].min() / (1024*1024), 3),
                'max_size_mb': round(df['file_size'].max() / (1024*1024), 3),
                'median_size_mb': round(df['file_size'].median() / (1024*1024), 3)
            }
        else:
            stats['total_size_mb'] = 0
            stats['avg_size_mb'] = 0
            stats['size_stats'] = {}
        
        # 时间统计
        if 'timestamp' in df.columns:
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                stats['time_stats'] = {
                    'first_generated': df['timestamp'].min().isoformat(),
                    'last_generated': df['timestamp'].max().isoformat(),
                    'generation_span_minutes': (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 60
                }
            except:
                stats['time_stats'] = {}
        
        return stats


class AdvancedAttackGenerator(AttackSampleGenerator):
    """高级攻击生成器 - 增强版"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.advanced_techniques = config.get('advanced_techniques', {})
        logger.info("初始化高级攻击生成器")
        logger.info(f"高级技术: {list(self.advanced_techniques.keys())}")
    
    @safe_pdf_operation
    def inject_contextual_attack(self, input_pdf: str, output_pdf: str, prompts: List[str]) -> bool:
        """注入上下文相关攻击 - 增强版"""
        doc = None
        try:
            doc = self._safe_open_pdf(input_pdf)
            if doc is None:
                return False
            
            context_keywords = [
                "conclusion", "reference", "acknowledge", "future work",
                "abstract", "introduction", "methodology", "discussion",
                "bibliography", "appendix", "thanks", "funding"
            ]
            
            injected_count = 0
            max_pages = min(5, doc.page_count)  # 扩大搜索范围
            
            for page_num in range(max_pages):
                try:
                    page = doc[page_num]
                    text_dict = page.get_text("dict")
                    
                    if not text_dict or "blocks" not in text_dict:
                        continue
                    
                    # 寻找合适的上下文
                    for block in text_dict["blocks"]:
                        if "lines" not in block:
                            continue
                        
                        for line in block["lines"]:
                            if "spans" not in line:
                                continue
                            
                            for span in line["spans"]:
                                text = span.get("text", "").lower()
                                
                                # 检查是否包含目标关键词
                                matching_keyword = None
                                for keyword in context_keywords:
                                    if keyword in text:
                                        matching_keyword = keyword
                                        break
                                
                                if matching_keyword:
                                    try:
                                        bbox = span["bbox"]
                                        original_size = span.get("size", 10)
                                        
                                        # 计算插入位置
                                        x = bbox[2] + 5  # 文本后面一点
                                        y = bbox[1]
                                        
                                        # 选择合适的提示词和样式
                                        prompt = random.choice(prompts) if prompts else "contextual insertion"
                                        
                                        # 根据上下文调整样式
                                        if matching_keyword in ["reference", "bibliography"]:
                                            # 在引用区域，使用更小的字体
                                            fontsize = max(4, original_size * 0.6)
                                            color = (0.8, 0.8, 0.8)
                                        elif matching_keyword in ["abstract", "conclusion"]:
                                            # 在重要部分，使用接近原文的样式
                                            fontsize = original_size * 0.9
                                            color = (0.95, 0.95, 0.95)
                                        else:
                                            # 默认样式
                                            fontsize = original_size * 0.8
                                            color = (0.9, 0.9, 0.9)
                                        
                                        page.insert_text(
                                            (x, y),
                                            f" {prompt[:50]}",  # 限制长度
                                            fontsize=fontsize,
                                            color=color,
                                            fontname="helv"
                                        )
                                        
                                        injected_count += 1
                                        logger.debug(f"在 '{matching_keyword}' 上下文中注入: {prompt[:20]}")
                                        
                                        # 每页限制注入数量
                                        if injected_count >= 2:
                                            break
                                            
                                    except Exception as e:
                                        logger.debug(f"上下文注入失败: {e}")
                                        continue
                            
                            if injected_count >= 2:
                                break
                        
                        if injected_count >= 2:
                            break
                    
                    if injected_count >= 2:
                        break
                        
                except Exception as e:
                    logger.debug(f"页面 {page_num} 上下文分析失败: {e}")
                    continue
            
            if injected_count == 0:
                logger.warning("未找到合适的上下文进行注入")
                if doc:
                    doc.close()
                return False
            
            # 保存文件
            try:
                doc.save(output_pdf, garbage=4, deflate=True, clean=True)
                doc.close()
                
                if validate_pdf(output_pdf):
                    logger.debug(f"上下文攻击注入成功: {output_pdf} (注入 {injected_count} 处)")
                    return True
                else:
                    if os.path.exists(output_pdf):
                        os.remove(output_pdf)
                    return False
                    
            except Exception as e:
                logger.error(f"保存PDF失败: {e}")
                if doc:
                    doc.close()
                return False
            
        except Exception as e:
            logger.error(f"上下文攻击注入失败 {input_pdf}: {e}")
            if doc:
                try:
                    doc.close()
                except:
                    pass
            return False
    
    def inject_semantic_attack(self, input_pdf: str, output_pdf: str, prompts: List[str]) -> bool:
        """注入语义攻击 - 新增方法"""
        # 这里可以实现更高级的语义攻击
        # 例如：同义词替换、语义对抗等
        return self.inject_mixed_language_attack(input_pdf, output_pdf, prompts)
    
    def inject_adaptive_attack(self, input_pdf: str, output_pdf: str, prompts: List[str]) -> bool:
        """注入自适应攻击 - 新增方法"""
        # 根据PDF内容自动选择最佳攻击策略
        attack_methods = [
            self.inject_white_text,
            self.inject_invisible_chars,
            self.inject_contextual_attack
        ]
        
        # 随机选择攻击方法
        method = random.choice(attack_methods)
        return method(input_pdf, output_pdf, prompts)
    
    def generate_single_attack(self, input_pdf: str, attack_type: str, 
                             language: str = None) -> Optional[str]:
        """生成单个高级攻击样本 - 扩展版"""
        
        # 高级攻击类型映射
        advanced_methods = {
            "contextual_attack": self.inject_contextual_attack,
            "semantic_attack": self.inject_semantic_attack,
            "adaptive_attack": self.inject_adaptive_attack
        }
        
        if attack_type in advanced_methods:
            try:
                if not validate_pdf(input_pdf):
                    return None
                
                base_name = os.path.splitext(os.path.basename(input_pdf))[0]
                safe_base_name = "".join(c for c in base_name if c.isalnum() or c in "._-")[:80]
                safe_language = (language or 'mixed').replace('/', '_')[:20]
                safe_attack_type = attack_type.replace('/', '_')[:30]
                
                output_name = f"{safe_base_name}_{safe_attack_type}_{safe_language}.pdf"
                output_pdf = os.path.join(self.output_dir, output_name)
                
                # 避免文件名冲突
                counter = 1
                while os.path.exists(output_pdf):
                    base_output = os.path.splitext(output_pdf)[0]
                    output_pdf = f"{base_output}_{counter}.pdf"
                    counter += 1
                
                prompts = self.select_prompts(language, count=3)
                success = advanced_methods[attack_type](input_pdf, output_pdf, prompts)
                
                if success and validate_pdf(output_pdf):
                    attack_info = {
                        'original_file': input_pdf,
                        'attack_file': output_pdf,
                        'attack_type': attack_type,
                        'language': language or 'mixed',
                        'prompts_used': prompts,
                        'file_size': os.path.getsize(output_pdf),
                        'timestamp': pd.Timestamp.now().isoformat(),
                        'generator_type': 'advanced'
                    }
                    self.attack_samples.append(attack_info)
                    return output_pdf
                else:
                    if os.path.exists(output_pdf):
                        os.remove(output_pdf)
                    return None
                
            except Exception as e:
                logger.error(f"生成高级攻击样本失败 {input_pdf}: {e}")
                return None
        else:
            # 使用父类方法处理标准攻击类型
            return super().generate_single_attack(input_pdf, attack_type, language)
