import arxiv
import requests
import os
import time
from typing import List, Dict, Optional
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse
import fitz
from .utils import setup_logging, ensure_dir, calculate_file_hash, ProgressTracker, validate_pdf, get_file_size

logger = setup_logging()

class ArxivDatasetCollector:
    """arXiv数据集收集器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.download_dir = ensure_dir(config['data_collection']['download_dir'])
        self.max_papers = config['data_collection']['max_papers']
        self.delay = config['data_collection']['delay_between_downloads']
        self.papers_info = []
        
        logger.info(f"初始化数据收集器，下载目录: {self.download_dir}")
    
    def search_papers(self, query: str, max_results: int = 100) -> List[Dict]:
        """搜索arxiv论文"""
        logger.info(f"搜索论文: {query}, 最大结果数: {max_results}")
        
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        papers = []
        try:
            for result in client.results(search):
                paper_info = {
                    'title': result.title.strip(),
                    'authors': [author.name for author in result.authors],
                    'abstract': result.summary.strip(),
                    'pdf_url': result.pdf_url,
                    'arxiv_id': result.entry_id.split('/')[-1],
                    'categories': result.categories,
                    'published': result.published.isoformat(),
                    'updated': result.updated.isoformat() if result.updated else None,
                    'doi': result.doi,
                    'journal_ref': result.journal_ref,
                    'primary_category': result.primary_category
                }
                papers.append(paper_info)
                
            logger.info(f"搜索完成，找到 {len(papers)} 篇论文")
            
        except Exception as e:
            logger.error(f"搜索论文失败: {e}")
            
        return papers
    
    def download_paper(self, paper: Dict) -> Optional[str]:
        """下载单篇论文"""
        try:
            pdf_url = paper['pdf_url']
            arxiv_id = paper['arxiv_id']
            filename = f"{arxiv_id}.pdf"
            filepath = os.path.join(self.download_dir, filename)
            
            # 检查文件是否已存在且有效
            if os.path.exists(filepath) and validate_pdf(filepath):
                logger.debug(f"文件已存在且有效: {filename}")
                return filepath
            
            # 下载文件
            logger.debug(f"下载: {filename}")
            response = requests.get(pdf_url, stream=True, timeout=30)
            response.raise_for_status()
            
            # 保存文件
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            # 验证下载的文件
            if validate_pdf(filepath):
                file_size = get_file_size(filepath)
                paper['file_path'] = filepath
                paper['file_size'] = file_size
                paper['file_hash'] = calculate_file_hash(filepath)
                
                logger.debug(f"下载成功: {filename} ({file_size} bytes)")
                return filepath
            else:
                logger.warning(f"下载的PDF文件无效: {filename}")
                if os.path.exists(filepath):
                    os.remove(filepath)
                return None
                
        except Exception as e:
            logger.error(f"下载失败 {paper['arxiv_id']}: {e}")
            return None
    
    def download_papers(self, papers: List[Dict]) -> List[str]:
        """批量下载论文"""
        downloaded_files = []
        
        progress = ProgressTracker(len(papers), "下载论文")
        
        for i, paper in enumerate(papers):
            try:
                filepath = self.download_paper(paper)
                if filepath:
                    downloaded_files.append(filepath)
                    self.papers_info.append(paper)
                
                progress.update()
                
                # 延迟以避免请求过于频繁
                if i < len(papers) - 1:
                    time.sleep(self.delay)
                    
            except KeyboardInterrupt:
                logger.warning("用户中断下载")
                break
            except Exception as e:
                logger.error(f"处理论文失败: {e}")
                progress.update()
        
        progress.finish()
        logger.info(f"下载完成，成功下载 {len(downloaded_files)} 篇论文")
        
        return downloaded_files
    
    def collect_multi_category_papers(self) -> List[str]:
        """收集多个类别的论文"""
        queries = self.config['data_collection']['search_queries']
        papers_per_query = self.config['data_collection']['papers_per_query']
        
        all_papers = []
        
        for query in queries:
            logger.info(f"搜索类别: {query}")
            papers = self.search_papers(query, papers_per_query)
            all_papers.extend(papers)
            
            if len(all_papers) >= self.max_papers:
                break
        
        # 去重（基于arxiv_id）
        seen_ids = set()
        unique_papers = []
        for paper in all_papers:
            if paper['arxiv_id'] not in seen_ids:
                seen_ids.add(paper['arxiv_id'])
                unique_papers.append(paper)
        
        logger.info(f"去重后共 {len(unique_papers)} 篇论文")
        
        # 限制数量
        if len(unique_papers) > self.max_papers:
            unique_papers = unique_papers[:self.max_papers]
        
        # 下载论文
        downloaded_files = self.download_papers(unique_papers)
        
        # 保存论文信息
        self.save_papers_info()
        
        return downloaded_files
    
    def save_papers_info(self):
        """保存论文信息到CSV"""
        if not self.papers_info:
            return
        
        info_file = os.path.join(self.download_dir, "papers_info.csv")
        df = pd.DataFrame(self.papers_info)
        df.to_csv(info_file, index=False, encoding='utf-8')
        logger.info(f"论文信息已保存到: {info_file}")
    
    def load_papers_info(self) -> pd.DataFrame:
        """加载论文信息"""
        info_file = os.path.join(self.download_dir, "papers_info.csv")
        if os.path.exists(info_file):
            return pd.read_csv(info_file, encoding='utf-8')
        return pd.DataFrame()
    
    def get_paper_statistics(self) -> Dict:
        """获取论文统计信息"""
        if not self.papers_info:
            df = self.load_papers_info()
            if df.empty:
                return {}
        else:
            df = pd.DataFrame(self.papers_info)
        
        stats = {
            'total_papers': len(df),
            'categories': df['primary_category'].value_counts().to_dict() if 'primary_category' in df.columns else {},
            'total_size_mb': df['file_size'].sum() / (1024*1024) if 'file_size' in df.columns else 0,
            'avg_size_mb': df['file_size'].mean() / (1024*1024) if 'file_size' in df.columns else 0,
        }
        
        return stats

class PubMedDataCollector:
    """PubMed数据收集器（示例扩展）"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.download_dir = ensure_dir(config['data_collection']['download_dir'])
        logger.info("初始化PubMed数据收集器")
    
    def search_papers(self, query: str, max_results: int = 100) -> List[Dict]:
        """搜索PubMed论文（需要实现）"""
        # TODO: 实现PubMed API搜索
        logger.warning("PubMed搜索功能待实现")
        return []

class LocalPDFCollector:
    """本地PDF文件收集器"""
    
    def __init__(self, config: Dict):
        self.config = config
        logger.info("初始化本地PDF收集器")
    
    def collect_from_directory(self, directory: str) -> List[str]:
        """从目录收集PDF文件"""
        pdf_files = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.pdf'):
                    filepath = os.path.join(root, file)
                    if validate_pdf(filepath):
                        pdf_files.append(filepath)
                        logger.debug(f"发现有效PDF: {filepath}")
                    else:
                        logger.warning(f"无效PDF文件: {filepath}")
        
        logger.info(f"从 {directory} 收集到 {len(pdf_files)} 个PDF文件")
        return pdf_files
