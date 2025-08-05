#!/usr/bin/env python3
"""
API Client Example
示例客户端代码，展示如何在Python中使用API
"""

import requests
import json
from pathlib import Path

class PromptInjectionAPIClient:
    """API 客户端类"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self):
        """健康检查"""
        response = self.session.get(f"{self.base_url}/health")
        return response.json()
    
    def get_metrics(self):
        """获取系统指标"""
        response = self.session.get(f"{self.base_url}/metrics")
        return response.json()
    
    def detect_single_file(self, file_path, detector_type="ensemble", return_details=False, threshold=None):
        """检测单个文件"""
        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f, 'application/pdf')}
            params = {
                'detector_type': detector_type,
                'return_details': return_details
            }
            if threshold is not None:
                params['threshold'] = threshold
            
            response = self.session.post(
                f"{self.base_url}/detect/single",
                files=files,
                params=params
            )
        
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()
    
    def detect_batch_files(self, file_paths, detector_type="ensemble", return_details=False, max_files=10):
        """批量检测文件"""
        files = []
        file_handles = []
        
        try:
            for file_path in file_paths:
                f = open(file_path, 'rb')
                file_handles.append(f)
                files.append(('files', (Path(file_path).name, f, 'application/pdf')))
            
            params = {
                'detector_type': detector_type,
                'return_details': return_details,
                'max_files': max_files
            }
            
            response = self.session.post(
                f"{self.base_url}/detect/batch",
                files=files,
                params=params
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                response.raise_for_status()
        
        finally:
            # 关闭文件句柄
            for f in file_handles:
                f.close()

def main():
    """示例使用"""
    print("🔍 API 客户端示例")
    print("=" * 50)
    
    # 创建客户端
    client = PromptInjectionAPIClient()
    
    try:
        # 1. 健康检查
        print("1. 健康检查...")
        health = client.health_check()
        print(f"   状态: {health['status']}")
        print(f"   检测器加载: {health['detectors_loaded']}")
        print(f"   运行时间: {health['uptime']:.2f}s")
        print()
        
        # 2. 获取指标
        print("2. 系统指标...")
        metrics = client.get_metrics()
        print(f"   总请求数: {metrics['total_requests']}")
        print(f"   成功检测: {metrics['successful_detections']}")
        print(f"   平均处理时间: {metrics['average_processing_time']:.3f}s")
        print()
        
        # 3. 单文件检测
        print("3. 单文件检测...")
        test_files = list(Path("../data/attack_samples").glob("*.pdf"))
        if test_files:
            result = client.detect_single_file(
                test_files[0], 
                detector_type="ensemble",
                return_details=False
            )
            print(f"   文件: {result['file_name']}")
            print(f"   恶意: {result['is_malicious']}")
            print(f"   风险分数: {result['risk_score']:.3f}")
            print(f"   处理时间: {result['processing_time']:.3f}s")
        else:
            print("   未找到测试文件")
        print()
        
        # 4. 批量检测
        print("4. 批量检测...")
        if len(test_files) >= 2:
            result = client.detect_batch_files(
                test_files[:3],
                detector_type="ensemble"
            )
            print(f"   总文件数: {result['total_files']}")
            print(f"   恶意文件: {result['malicious_files']}")
            print(f"   清洁文件: {result['clean_files']}")
            print(f"   处理时间: {result['processing_time']:.3f}s")
            print(f"   恶意率: {result['summary']['malicious_rate']:.2%}")
        else:
            print("   测试文件不足")
        
        print()
        print("✅ 所有测试完成！")
        
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到API服务。请确保服务运行在 http://localhost:8000")
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    main()