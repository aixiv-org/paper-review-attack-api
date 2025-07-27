#!/usr/bin/env python3
"""
数据下载脚本
用于从arXiv等源下载学术论文数据
"""

import sys
import os
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_collector import ArxivDatasetCollector, LocalPDFCollector
from src.utils import setup_logging, load_config, ProgressTracker

def main():
    parser = argparse.ArgumentParser(description='下载论文数据集')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='配置文件路径')
    parser.add_argument('--source', type=str, default='arxiv',
                       choices=['arxiv', 'local'],
                       help='数据源类型')
    parser.add_argument('--local-dir', type=str,
                       help='本地PDF目录路径（当source=local时使用）')
    parser.add_argument('--max-papers', type=int,
                       help='最大下载论文数（覆盖配置文件）')
    parser.add_argument('--categories', nargs='+',
                       help='arXiv类别列表（覆盖配置文件）')
    parser.add_argument('--output-dir', type=str,
                       help='输出目录（覆盖配置文件）')
    parser.add_argument('--log-file', type=str,
                       help='日志文件路径')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='详细输出')
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level, args.log_file)
    
    logger.info("=" * 60)
    logger.info("论文数据下载器启动")
    logger.info("=" * 60)
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 命令行参数覆盖配置
        if args.max_papers:
            config['data_collection']['max_papers'] = args.max_papers
        
        if args.categories:
            config['data_collection']['search_queries'] = [f"cat:{cat}" for cat in args.categories]
        
        if args.output_dir:
            config['data_collection']['download_dir'] = args.output_dir
        
        # 创建数据收集器
        if args.source == 'arxiv':
            collector = ArxivDatasetCollector(config)
            logger.info("使用arXiv数据收集器")
            
            # 收集多类别论文
            downloaded_files = collector.collect_multi_category_papers()
            
            # 打印统计信息
            stats = collector.get_paper_statistics()
            logger.info(f"下载统计: {stats}")
            
        elif args.source == 'local':
            if not args.local_dir:
                logger.error("使用本地源时必须指定--local-dir参数")
                return 1
            
            collector = LocalPDFCollector(config)
            logger.info(f"使用本地PDF收集器，目录: {args.local_dir}")
            
            downloaded_files = collector.collect_from_directory(args.local_dir)
        
        else:
            logger.error(f"不支持的数据源: {args.source}")
            return 1
        
        logger.info(f"数据下载完成！共获得 {len(downloaded_files)} 个文件")
        
        # 验证下载的文件
        logger.info("验证下载文件...")
        valid_files = []
        invalid_files = []
        
        for file_path in downloaded_files:
            if os.path.exists(file_path) and os.path.getsize(file_path) > 1024:  # 至少1KB
                valid_files.append(file_path)
            else:
                invalid_files.append(file_path)
        
        logger.info(f"验证结果: {len(valid_files)} 个有效文件, {len(invalid_files)} 个无效文件")
        
        if invalid_files:
            logger.warning("无效文件列表:")
            for file_path in invalid_files[:10]:  # 只显示前10个
                logger.warning(f"  - {file_path}")
            if len(invalid_files) > 10:
                logger.warning(f"  ... 还有 {len(invalid_files) - 10} 个")
        
        # 保存文件列表
        output_dir = Path(config['data_collection']['download_dir'])
        file_list_path = output_dir / "downloaded_files.txt"
        
        with open(file_list_path, 'w', encoding='utf-8') as f:
            for file_path in valid_files:
                f.write(f"{file_path}\n")
        
        logger.info(f"文件列表已保存到: {file_list_path}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("用户中断下载")
        return 1
    except Exception as e:
        logger.error(f"下载过程中发生错误: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())
