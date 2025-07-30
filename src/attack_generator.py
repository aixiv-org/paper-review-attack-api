import os
import random
import base64
import json
import fitz  # PyMuPDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import white, black, Color
from typing import List, Dict, Tuple, Optional
import pandas as pd
from pathlib import Path
from .utils import setup_logging, ensure_dir, ProgressTracker, validate_pdf

logger = setup_logging()

class AttackSampleGenerator:
    """攻击样本生成器"""
    
    def __init__(self, config: Dict):
        self.config = config
        # 🔧 修复1: 添加缺失的 attack_config 属性
        self.attack_config = config['attack_generation']
        
        self.output_dir = ensure_dir(self.attack_config['output_dir'])
        self.attack_types = self.attack_config['attack_types']
        self.prompt_templates = self.attack_config['prompt_templates']
        self.attack_samples = []
        
        # 🔧 修复2: 添加 logger 属性
        self.logger = logger
        
        logger.info(f"初始化攻击样本生成器，输出目录: {self.output_dir}")
        logger.info(f"支持的攻击类型: {self.attack_types}")
        logger.info(f"支持的语言: {list(self.prompt_templates.keys())}")
    
    def select_prompts(self, language: str = None, count: int = 2) -> List[str]:
        """选择提示词"""
        if language and language in self.prompt_templates:
            prompts = self.prompt_templates[language]
        else:
            # 随机选择语言
            language = random.choice(list(self.prompt_templates.keys()))
            prompts = self.prompt_templates[language]
        
        if not prompts:
            logger.warning(f"语言 {language} 没有可用的提示词模板")
            return ["Default malicious prompt for testing"]
        
        selected = random.sample(prompts, min(count, len(prompts)))
        return selected
    
    def inject_white_text(self, input_pdf: str, output_pdf: str, prompts: List[str]) -> bool:
        """注入白色字体攻击"""
        try:
            doc = fitz.open(input_pdf)
            
            for page_num in range(min(3, len(doc))):  # 只在前3页注入
                page = doc[page_num]
                rect = page.rect
                
                for i, prompt in enumerate(prompts):
                    # 随机选择注入位置
                    x = random.uniform(50, rect.width - 200)
                    y = random.uniform(50, rect.height - 50)
                    
                    # 插入白色文本，使用极小字号
                    page.insert_text(
                        (x, y),
                        prompt,
                        fontsize=random.uniform(0.5, 1.5),  # 极小字号
                        color=(1, 1, 1),  # 白色
                        fontname="helv"
                    )
                    
                    # 在页面边缘也插入一些
                    if i == 0:
                        page.insert_text(
                            (rect.width - 100, rect.height - 20),
                            prompt[:20],  # 截断以适应边缘
                            fontsize=0.1,
                            color=(1, 1, 1)
                        )
            
            doc.save(output_pdf)
            doc.close()
            
            logger.debug(f"白色字体攻击注入成功: {output_pdf}")
            return True
            
        except Exception as e:
            logger.error(f"白色字体攻击注入失败 {output_pdf}: {e}")
            return False
    
    def inject_metadata_attack(self, input_pdf: str, output_pdf: str, prompts: List[str]) -> bool:
        """注入元数据攻击"""
        try:
            doc = fitz.open(input_pdf)
            
            # 获取现有元数据
            metadata = doc.metadata.copy() if doc.metadata else {}
            
            # 注入恶意提示词到各个元数据字段
            metadata['subject'] = random.choice(prompts) if prompts else "Research"
            metadata['keywords'] = "; ".join(prompts[:2]) if len(prompts) >= 2 else "AI; ML"
            
            # 创建看似正常但包含提示词的字段
            if len(prompts) > 2:
                metadata['creator'] = f"LaTeX with {prompts[2][:30]}"
            
            # 注入到注释字段（如果支持）
            original_title = metadata.get('title', 'Research Paper')
            metadata['title'] = original_title + f" {prompts[0][:50]}" if prompts else original_title
            
            doc.set_metadata(metadata)
            doc.save(output_pdf)
            doc.close()
            
            logger.debug(f"元数据攻击注入成功: {output_pdf}")
            return True
            
        except Exception as e:
            logger.error(f"元数据攻击注入失败 {output_pdf}: {e}")
            return False
    
    def inject_invisible_chars(self, input_pdf: str, output_pdf: str, prompts: List[str]) -> bool:
        """注入不可见字符攻击"""
        try:
            doc = fitz.open(input_pdf)
            
            # 各种不可见字符
            invisible_chars = [
                '\u200b',  # 零宽空格
                '\u200c',  # 零宽非连接符
                '\u200d',  # 零宽连接符
                '\ufeff',  # 字节顺序标记
                '\u2060',  # 单词连接符
                '\u180e',  # 蒙古文元音分隔符
            ]
            
            for page_num in range(min(2, len(doc))):
                page = doc[page_num]
                rect = page.rect
                
                for prompt in prompts[:2]:
                    # 用不可见字符编码提示词
                    encoded_prompt = ""
                    for char in prompt:
                        encoded_prompt += char
                        if random.random() < 0.3:  # 30%概率插入不可见字符
                            encoded_prompt += random.choice(invisible_chars)
                    
                    # 随机位置插入
                    x = random.uniform(100, rect.width - 100)
                    y = random.uniform(100, rect.height - 100)
                    
                    page.insert_text(
                        (x, y),
                        encoded_prompt[:100],  # 限制长度避免过长
                        fontsize=random.uniform(0.1, 0.5),
                        color=(0.99, 0.99, 0.99)  # 几乎白色
                    )
            
            doc.save(output_pdf)
            doc.close()
            
            logger.debug(f"不可见字符攻击注入成功: {output_pdf}")
            return True
            
        except Exception as e:
            logger.error(f"不可见字符攻击注入失败 {output_pdf}: {e}")
            return False
    
    def inject_mixed_language_attack(self, input_pdf: str, output_pdf: str, prompts: List[str]) -> bool:
        """注入混合语言攻击"""
        try:
            doc = fitz.open(input_pdf)
            
            # 混合不同语言的提示词
            mixed_prompts = []
            languages = ['english', 'chinese', 'japanese']
            
            for lang in languages:
                if lang in self.prompt_templates and self.prompt_templates[lang]:
                    lang_prompts = self.prompt_templates[lang]
                    mixed_prompts.extend(random.sample(lang_prompts, min(1, len(lang_prompts))))
            
            # 如果没有混合语言，使用原始提示词
            if not mixed_prompts:
                mixed_prompts = prompts
            
            # 创建混合语言提示
            if len(mixed_prompts) >= 2:
                mixed_text = f"{mixed_prompts[0]} {mixed_prompts[1]}"
            else:
                mixed_text = " ".join(mixed_prompts) if mixed_prompts else "Mixed language test"
            
            # 注入到PDF
            for page_num in range(min(2, len(doc))):
                page = doc[page_num]
                rect = page.rect
                
                # 使用小字体插入混合语言文本
                x = random.uniform(50, rect.width - 300)
                y = random.uniform(50, rect.height - 50)
                
                page.insert_text(
                    (x, y),
                    mixed_text[:100],  # 限制长度
                    fontsize=1.0,
                    color=(0.95, 0.95, 0.95)  # 浅灰色
                )
            
            doc.save(output_pdf)
            doc.close()
            
            logger.debug(f"混合语言攻击注入成功: {output_pdf}")
            return True
            
        except Exception as e:
            logger.error(f"混合语言攻击注入失败 {output_pdf}: {e}")
            return False
    
    def inject_steganographic_attack(self, input_pdf: str, output_pdf: str, prompts: List[str]) -> bool:
        """注入隐写术攻击（高级）"""
        try:
            doc = fitz.open(input_pdf)
            
            # Base64编码提示词
            encoded_prompts = []
            for prompt in prompts[:2]:
                try:
                    encoded = base64.b64encode(prompt.encode('utf-8')).decode('ascii')
                    encoded_prompts.append(encoded)
                except Exception as e:
                    logger.warning(f"Base64编码失败: {e}")
                    encoded_prompts.append(prompt)
            
            # 将编码后的提示词作为"引用"或"URL"插入
            for page_num in range(min(1, len(doc))):
                page = doc[page_num]
                rect = page.rect
                
                for encoded in encoded_prompts:
                    # 伪装成参考文献或URL
                    fake_ref = f"https://example.com/ref/{encoded[:20]}"
                    fake_doi = f"DOI: 10.1000/{encoded[20:40]}" if len(encoded) > 20 else f"DOI: 10.1000/{encoded}"
                    
                    x = random.uniform(50, rect.width - 200)
                    y = random.uniform(rect.height - 100, rect.height - 20)
                    
                    page.insert_text(
                        (x, y),
                        fake_ref,
                        fontsize=8.0,
                        color=(0.7, 0.7, 0.7)  # 灰色，像引用
                    )
                    
                    # 添加DOI
                    page.insert_text(
                        (x, y - 10),
                        fake_doi,
                        fontsize=6.0,
                        color=(0.7, 0.7, 0.7)
                    )
            
            doc.save(output_pdf)
            doc.close()
            
            logger.debug(f"隐写术攻击注入成功: {output_pdf}")
            return True
            
        except Exception as e:
            logger.error(f"隐写术攻击注入失败 {output_pdf}: {e}")
            return False
    
    def generate_single_attack(self, input_pdf: str, attack_type: str, 
                             language: str = None) -> Optional[str]:
        """生成单个攻击样本"""
        if not validate_pdf(input_pdf):
            logger.warning(f"输入PDF无效: {input_pdf}")
            return None
        
        try:
            # 生成输出文件名
            base_name = os.path.splitext(os.path.basename(input_pdf))[0]
            output_name = f"{base_name}_{attack_type}_{language or 'mixed'}.pdf"
            output_pdf = os.path.join(self.output_dir, output_name)
            
            # 选择提示词
            prompts = self.select_prompts(language, count=3)
            
            # 根据攻击类型执行注入
            success = False
            if attack_type == "white_text":
                success = self.inject_white_text(input_pdf, output_pdf, prompts)
            elif attack_type == "metadata":
                success = self.inject_metadata_attack(input_pdf, output_pdf, prompts)
            elif attack_type == "invisible_chars":
                success = self.inject_invisible_chars(input_pdf, output_pdf, prompts)
            elif attack_type == "mixed_language":
                success = self.inject_mixed_language_attack(input_pdf, output_pdf, prompts)
            elif attack_type == "steganographic":
                success = self.inject_steganographic_attack(input_pdf, output_pdf, prompts)
            else:
                logger.warning(f"未知攻击类型: {attack_type}")
                return None
            
            if success and validate_pdf(output_pdf):
                # 记录攻击信息
                attack_info = {
                    'original_file': input_pdf,
                    'attack_file': output_pdf,
                    'attack_type': attack_type,
                    'language': language or 'mixed',
                    'prompts_used': prompts,
                    'file_size': os.path.getsize(output_pdf)
                }
                self.attack_samples.append(attack_info)
                
                logger.debug(f"攻击样本生成成功: {output_pdf}")
                return output_pdf
            else:
                logger.error(f"攻击样本生成失败: {output_pdf}")
                if os.path.exists(output_pdf):
                    os.remove(output_pdf)
                return None
                
        except Exception as e:
            logger.error(f"生成攻击样本时发生错误 {input_pdf}: {e}")
            return None
    
    def _parse_attack_types(self) -> Dict[str, float]:
        """解析攻击类型配置"""
        attack_types_config = self.attack_config.get('attack_types', [])
        
        if isinstance(attack_types_config, list):
            # 旧格式：等概率分配
            num_types = len(attack_types_config)
            if num_types == 0:
                return {'white_text': 1.0}  # 默认攻击类型
            return {attack_type: 1.0/num_types for attack_type in attack_types_config}
        elif isinstance(attack_types_config, dict):
            # 新格式：使用配置的比例
            total = sum(attack_types_config.values())
            if abs(total - 1.0) > 0.01:  # 允许小的舍入误差
                logger.warning(f"攻击类型比例总和为 {total:.3f}，将自动标准化为 1.0")
                return {k: v/total for k, v in attack_types_config.items()}
            return attack_types_config
        else:
            raise ValueError("Invalid attack_types configuration")

    def generate_attack_samples(self, clean_pdfs: List[str]) -> List[str]:
        """🔧 修复3: 统一方法签名，返回文件路径列表"""
        if not clean_pdfs:
            logger.warning("没有输入的PDF文件")
            return []
        
        attack_ratio = self.attack_config.get('attack_ratio', 0.3)
        total_attack_samples = int(len(clean_pdfs) * attack_ratio)
        
        if total_attack_samples == 0:
            logger.warning("根据攻击比例，不需要生成攻击样本")
            return []
        
        logger.info(f"计划生成 {total_attack_samples} 个攻击样本")
        
        # 🔧 修复4: 简化攻击类型处理
        if isinstance(self.attack_types, list):
            attack_types = self.attack_types
        else:
            # 如果是字典，取键
            attack_types = list(self.attack_types.keys()) if isinstance(self.attack_types, dict) else ['white_text']
        
        # 检查生成策略
        generation_strategy = self.attack_config.get('generation_strategy', {})
        mode = generation_strategy.get('mode', 'random')
        
        if mode == 'proportional' and isinstance(self.attack_types, dict):
            return self._generate_proportional_samples(clean_pdfs, total_attack_samples, self.attack_types)
        else:
            return self._generate_random_samples(clean_pdfs, total_attack_samples, attack_types)

    def _generate_random_samples(self, clean_pdfs: List[str], total_samples: int, attack_types: List[str]) -> List[str]:
        """🔧 修复5: 添加缺失的随机生成方法"""
        generated_samples = []
        
        # 随机选择PDF文件
        selected_pdfs = random.sample(clean_pdfs, min(total_samples, len(clean_pdfs)))
        
        progress = ProgressTracker(len(selected_pdfs), "生成攻击样本")
        
        for pdf_path in selected_pdfs:
            try:
                # 随机选择攻击类型和语言
                attack_type = random.choice(attack_types)
                language = random.choice(list(self.prompt_templates.keys()))
                
                output_path = self.generate_single_attack(pdf_path, attack_type, language)
                if output_path:
                    generated_samples.append(output_path)
                
                progress.update()
                
            except Exception as e:
                logger.error(f"生成攻击样本失败 {pdf_path}: {e}")
                progress.update()
                continue
        
        progress.finish()
        
        # 保存攻击样本信息
        self.save_attack_info()
        
        logger.info(f"攻击样本生成完成，共生成 {len(generated_samples)} 个样本")
        return generated_samples

    def _generate_proportional_samples(self, clean_pdfs: List[str], total_samples: int, 
                                    attack_proportions: Dict[str, float]) -> List[str]:
        """按比例生成攻击样本"""
        generation_strategy = self.attack_config.get('generation_strategy', {})
        min_samples = generation_strategy.get('min_samples_per_type', 1)
        max_samples = generation_strategy.get('max_samples_per_type', 50)
        
        # 计算每种攻击类型的目标数量
        target_counts = {}
        for attack_type, proportion in attack_proportions.items():
            target_count = int(total_samples * proportion)
            target_count = max(min_samples, min(max_samples, target_count))
            target_counts[attack_type] = target_count
        
        # 调整总数以匹配目标
        actual_total = sum(target_counts.values())
        if actual_total != total_samples:
            # 按比例调整
            adjustment_factor = total_samples / actual_total if actual_total > 0 else 1
            for attack_type in target_counts:
                target_counts[attack_type] = max(min_samples, 
                                            int(target_counts[attack_type] * adjustment_factor))
        
        logger.info("攻击样本生成计划:")
        for attack_type, count in target_counts.items():
            percentage = (count / sum(target_counts.values()) * 100) if sum(target_counts.values()) > 0 else 0
            logger.info(f"  {attack_type}: {count} ({percentage:.1f}%)")
        
        # 生成样本
        generated_samples = []
        attack_type_counts = {attack_type: 0 for attack_type in target_counts}
        selected_pdfs = random.sample(clean_pdfs, min(sum(target_counts.values()), len(clean_pdfs)))
        
        pdf_index = 0
        progress = ProgressTracker(sum(target_counts.values()), "生成攻击样本")
        
        for attack_type, target_count in target_counts.items():
            for _ in range(target_count):
                if pdf_index >= len(selected_pdfs):
                    logger.warning("PDF文件不足，无法生成所有攻击样本")
                    break
                
                try:
                    pdf_path = selected_pdfs[pdf_index]
                    language = random.choice(list(self.prompt_templates.keys()))
                    
                    output_path = self.generate_single_attack(pdf_path, attack_type, language)
                    if output_path:
                        generated_samples.append(output_path)
                        attack_type_counts[attack_type] += 1
                    
                    pdf_index += 1
                    progress.update()
                    
                except Exception as e:
                    logger.error(f"生成攻击样本失败 {pdf_path}: {e}")
                    pdf_index += 1
                    progress.update()
                    continue
        
        progress.finish()
        
        # 保存攻击样本信息
        self.save_attack_info()
        
        # 打印实际生成统计
        logger.info("攻击样本实际生成统计:")
        total_generated = sum(attack_type_counts.values())
        for attack_type, count in attack_type_counts.items():
            percentage = (count / total_generated * 100) if total_generated > 0 else 0
            logger.info(f"  {attack_type}: {count} ({percentage:.1f}%)")
        
        return generated_samples
    
    def save_attack_info(self):
        """🔧 修复6: 改进保存方法，同时支持CSV和JSON"""
        if not self.attack_samples:
            return
        
        try:
            # 保存为JSON格式（用于程序读取）
            json_file = os.path.join(self.output_dir, "attack_samples_info.json")
            attack_summary = {
                'total_samples': len(self.attack_samples),
                'attack_samples': self.attack_samples,
                'generation_config': {
                    'attack_types': self.attack_types,
                    'attack_ratio': self.attack_config.get('attack_ratio', 0.3),
                    'languages': list(self.prompt_templates.keys())
                }
            }
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(attack_summary, f, indent=2, ensure_ascii=False)
            
            # 保存为CSV格式（用于人工查看）
            csv_file = os.path.join(self.output_dir, "attack_samples_info.csv")
            df = pd.DataFrame(self.attack_samples)
            df.to_csv(csv_file, index=False, encoding='utf-8')
            
            logger.info(f"攻击样本信息已保存: {json_file}, {csv_file}")
            
        except Exception as e:
            logger.error(f"保存攻击样本信息失败: {e}")
    
    def load_attack_info(self) -> pd.DataFrame:
        """加载攻击样本信息"""
        info_file = os.path.join(self.output_dir, "attack_samples_info.csv")
        if os.path.exists(info_file):
            return pd.read_csv(info_file, encoding='utf-8')
        return pd.DataFrame()
    
    def get_attack_statistics(self) -> Dict:
        """获取攻击样本统计信息"""
        if not self.attack_samples:
            df = self.load_attack_info()
        else:
            df = pd.DataFrame(self.attack_samples)
        
        if df.empty:
            return {
                'total_attacks': 0,
                'attack_types': {},
                'languages': {},
                'total_size_mb': 0
            }
        
        stats = {
            'total_attacks': len(df),
            'attack_types': df['attack_type'].value_counts().to_dict(),
            'languages': df['language'].value_counts().to_dict(),
            'total_size_mb': df['file_size'].sum() / (1024*1024) if 'file_size' in df.columns else 0,
        }
        
        return stats


class AdvancedAttackGenerator(AttackSampleGenerator):
    """高级攻击生成器"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        logger.info("初始化高级攻击生成器")
    
    def inject_contextual_attack(self, input_pdf: str, output_pdf: str, prompts: List[str]) -> bool:
        """注入上下文相关攻击"""
        try:
            doc = fitz.open(input_pdf)
            
            # 分析PDF内容，找到合适的插入点
            for page_num in range(min(3, len(doc))):
                page = doc[page_num]
                text_blocks = page.get_text("dict")["blocks"]
                
                # 寻找参考文献或结论部分
                for block in text_blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span.get("text", "").lower()
                                if any(keyword in text for keyword in 
                                      ["conclusion", "reference", "acknowledge", "future work"]):
                                    
                                    # 在这些部分附近插入提示词
                                    bbox = span["bbox"]
                                    x = bbox[2] + 10  # 在文本后面
                                    y = bbox[1]
                                    
                                    prompt = random.choice(prompts) if prompts else "contextual test"
                                    page.insert_text(
                                        (x, y),
                                        f" {prompt}",
                                        fontsize=span.get("size", 10) * 0.8,
                                        color=(0.98, 0.98, 0.98)
                                    )
                                    break
            
            doc.save(output_pdf)
            doc.close()
            
            logger.debug(f"上下文攻击注入成功: {output_pdf}")
            return True
            
        except Exception as e:
            logger.error(f"上下文攻击注入失败 {output_pdf}: {e}")
            return False
    
    def generate_single_attack(self, input_pdf: str, attack_type: str, 
                             language: str = None) -> Optional[str]:
        """🔧 修复7: 扩展高级攻击生成"""
        if attack_type == "contextual_attack":
            try:
                if not validate_pdf(input_pdf):
                    return None
                
                base_name = os.path.splitext(os.path.basename(input_pdf))[0]
                output_name = f"{base_name}_{attack_type}_{language or 'mixed'}.pdf"
                output_pdf = os.path.join(self.output_dir, output_name)
                
                prompts = self.select_prompts(language, count=2)
                success = self.inject_contextual_attack(input_pdf, output_pdf, prompts)
                
                if success and validate_pdf(output_pdf):
                    attack_info = {
                        'original_file': input_pdf,
                        'attack_file': output_pdf,
                        'attack_type': attack_type,
                        'language': language or 'mixed',
                        'prompts_used': prompts,
                        'file_size': os.path.getsize(output_pdf)
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
