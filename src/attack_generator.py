import os
import random
import base64
import fitz  # PyMuPDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import white, black, Color
from typing import List, Dict, Tuple, Optional
import pandas as pd
from .utils import setup_logging, ensure_dir, ProgressTracker, validate_pdf

logger = setup_logging()

class AttackSampleGenerator:
    """攻击样本生成器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = ensure_dir(config['attack_generation']['output_dir'])
        self.attack_types = config['attack_generation']['attack_types']
        self.prompt_templates = config['attack_generation']['prompt_templates']
        self.attack_samples = []
        
        logger.info(f"初始化攻击样本生成器，输出目录: {self.output_dir}")
    
    def select_prompts(self, language: str = None, count: int = 2) -> List[str]:
        """选择提示词"""
        if language and language in self.prompt_templates:
            prompts = self.prompt_templates[language]
        else:
            # 随机选择语言
            language = random.choice(list(self.prompt_templates.keys()))
            prompts = self.prompt_templates[language]
        
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
            metadata = doc.metadata.copy()
            
            # 注入恶意提示词到各个元数据字段
            metadata['subject'] = random.choice(prompts)
            metadata['keywords'] = "; ".join(prompts[:2])
            
            # 创建看似正常但包含提示词的字段
            if len(prompts) > 2:
                metadata['creator'] = f"LaTeX with {prompts[2][:30]}"
            
            # 注入到注释字段（如果支持）
            metadata['title'] = metadata.get('title', '') + f" {prompts[0][:50]}"
            
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
                        encoded_prompt,
                        fontsize=random.uniform(0.1, 0.5),
                        color=(0.99, 0.99, 0.99)  # 几乎白色
                    )
                    
                    # 在正常文本中间插入不可见字符
                    text_blocks = page.get_text("dict")["blocks"]
                    if text_blocks:
                        block = random.choice([b for b in text_blocks if "lines" in b])
                        if "lines" in block:
                            line = random.choice(block["lines"])
                            # 这里可以进一步处理文本插入，但PDF修改较复杂
            
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
                if lang in self.prompt_templates:
                    lang_prompts = self.prompt_templates[lang]
                    mixed_prompts.extend(random.sample(lang_prompts, min(1, len(lang_prompts))))
            
            # 创建混合语言提示
            if len(mixed_prompts) >= 2:
                mixed_text = f"{mixed_prompts[0]} {mixed_prompts[1]}"
            else:
                mixed_text = " ".join(mixed_prompts)
            
            # 注入到PDF
            for page_num in range(min(2, len(doc))):
                page = doc[page_num]
                rect = page.rect
                
                # 使用小字体插入混合语言文本
                x = random.uniform(50, rect.width - 300)
                y = random.uniform(50, rect.height - 50)
                
                page.insert_text(
                    (x, y),
                    mixed_text,
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
                encoded = base64.b64encode(prompt.encode('utf-8')).decode('ascii')
                encoded_prompts.append(encoded)
            
            # 将编码后的提示词作为"引用"或"URL"插入
            for page_num in range(min(1, len(doc))):
                page = doc[page_num]
                rect = page.rect
                
                for encoded in encoded_prompts:
                    # 伪装成参考文献或URL
                    fake_ref = f"https://example.com/ref/{encoded[:20]}"
                    
                    x = random.uniform(50, rect.width - 200)
                    y = random.uniform(rect.height - 100, rect.height - 20)
                    
                    page.insert_text(
                        (x, y),
                        fake_ref,
                        fontsize=8.0,
                        color=(0.7, 0.7, 0.7)  # 灰色，像引用
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
            
            return output_pdf
        else:
            logger.error(f"攻击样本生成失败: {output_pdf}")
            if os.path.exists(output_pdf):
                os.remove(output_pdf)
            return None
    
    def generate_attack_samples(self, clean_pdfs: List[str]) -> List[Dict]:
        """批量生成攻击样本"""
        attack_ratio = self.config['attack_generation']['attack_ratio']
        num_attacks = int(len(clean_pdfs) * attack_ratio)
        
        if num_attacks == 0:
            logger.warning("没有足够的PDF文件生成攻击样本")
            return []
        
        # 随机选择要攻击的PDF
        selected_pdfs = random.sample(clean_pdfs, min(num_attacks, len(clean_pdfs)))
        
        total_tasks = len(selected_pdfs) * len(self.attack_types)
        progress = ProgressTracker(total_tasks, "生成攻击样本")
        
        generated_samples = []
        
        for pdf_path in selected_pdfs:
            for attack_type in self.attack_types:
                # 随机选择语言
                language = random.choice(list(self.prompt_templates.keys()))
                
                output_path = self.generate_single_attack(pdf_path, attack_type, language)
                
                if output_path:
                    generated_samples.append(output_path)
                    logger.debug(f"生成攻击样本: {output_path}")
                
                progress.update()
        
        progress.finish()
        
        # 保存攻击样本信息
        self.save_attack_info()
        
        logger.info(f"攻击样本生成完成，共生成 {len(generated_samples)} 个样本")
        return generated_samples
    
    def save_attack_info(self):
        """保存攻击样本信息"""
        if not self.attack_samples:
            return
        
        info_file = os.path.join(self.output_dir, "attack_samples_info.csv")
        df = pd.DataFrame(self.attack_samples)
        df.to_csv(info_file, index=False, encoding='utf-8')
        logger.info(f"攻击样本信息已保存到: {info_file}")
    
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
            return {}
        
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
                                text = span["text"].lower()
                                if any(keyword in text for keyword in 
                                      ["conclusion", "reference", "acknowledge", "future work"]):
                                    
                                    # 在这些部分附近插入提示词
                                    bbox = span["bbox"]
                                    x = bbox[2] + 10  # 在文本后面
                                    y = bbox[1]
                                    
                                    prompt = random.choice(prompts)
                                    page.insert_text(
                                        (x, y),
                                        f" {prompt}",
                                        fontsize=span["size"] * 0.8,
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
