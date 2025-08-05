#!/usr/bin/env python3
"""
Performance Testing Script
API 性能测试脚本
"""

import requests
import time
import threading
import statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

class PerformanceTest:
    def __init__(self, base_url="http://localhost:8000", test_file=None):
        self.base_url = base_url
        self.test_file = test_file or self._find_test_file()
        self.results = []
        
    def _find_test_file(self):
        """找到测试文件"""
        test_files = list(Path("../data/attack_samples").glob("*.pdf"))
        if not test_files:
            test_files = list(Path("../data/clean_papers").glob("*.pdf"))
        return test_files[0] if test_files else None
    
    def single_request_test(self, request_id):
        """单个请求测试"""
        if not self.test_file:
            return {"error": "No test file found"}
        
        start_time = time.time()
        
        try:
            with open(self.test_file, 'rb') as f:
                files = {'file': (f'test_{request_id}.pdf', f, 'application/pdf')}
                params = {'detector_type': 'ensemble', 'return_details': False}
                
                response = requests.post(
                    f"{self.base_url}/detect/single",
                    files=files,
                    params=params,
                    timeout=30
                )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            return {
                "request_id": request_id,
                "status_code": response.status_code,
                "response_time": response_time,
                "success": response.status_code == 200,
                "timestamp": start_time
            }
            
        except Exception as e:
            end_time = time.time()
            return {
                "request_id": request_id,
                "status_code": -1,
                "response_time": end_time - start_time,
                "success": False,
                "error": str(e),
                "timestamp": start_time
            }
    
    def run_load_test(self, num_requests=10, concurrent_users=5):
        """负载测试"""
        print(f"🚀 开始负载测试:")
        print(f"   请求数量: {num_requests}")
        print(f"   并发用户: {concurrent_users}")
        print(f"   测试文件: {self.test_file.name if self.test_file else 'None'}")
        print()
        
        if not self.test_file:
            print("❌ 未找到测试文件!")
            return
        
        # 执行测试
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [
                executor.submit(self.single_request_test, i) 
                for i in range(num_requests)
            ]
            
            results = []
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                results.append(result)
                
                # 显示进度
                progress = (i + 1) / num_requests * 100
                print(f"   进度: {progress:5.1f}% ({i+1}/{num_requests})", end='\r')
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n   完成时间: {total_time:.2f}秒")
        print()
        
        # 分析结果
        self._analyze_results(results, total_time)
    
    def _analyze_results(self, results, total_time):
        """分析测试结果"""
        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]
        
        response_times = [r['response_time'] for r in successful_requests]
        
        print("📊 测试结果分析:")
        print("=" * 50)
        
        # 基本统计
        print(f"总请求数:       {len(results)}")
        print(f"成功请求:       {len(successful_requests)} ({len(successful_requests)/len(results)*100:.1f}%)")
        print(f"失败请求:       {len(failed_requests)} ({len(failed_requests)/len(results)*100:.1f}%)")
        print(f"总测试时间:     {total_time:.2f}秒")
        print(f"请求吞吐量:     {len(results)/total_time:.2f} req/s")
        
        # 响应时间统计
        if response_times:
            print()
            print("响应时间统计:")
            print(f"  平均响应时间:   {statistics.mean(response_times):.3f}秒")
            print(f"  最小响应时间:   {min(response_times):.3f}秒")
            print(f"  最大响应时间:   {max(response_times):.3f}秒")
            print(f"  响应时间中位数: {statistics.median(response_times):.3f}秒")
            
            if len(response_times) > 1:
                print(f"  响应时间标准差: {statistics.stdev(response_times):.3f}秒")
            
            # 百分位数
            sorted_times = sorted(response_times)
            p95_index = int(len(sorted_times) * 0.95)
            p99_index = int(len(sorted_times) * 0.99)
            
            if p95_index < len(sorted_times):
                print(f"  95%响应时间:   {sorted_times[p95_index]:.3f}秒")
            if p99_index < len(sorted_times):
                print(f"  99%响应时间:   {sorted_times[p99_index]:.3f}秒")
        
        # 错误分析
        if failed_requests:
            print()
            print("错误分析:")
            error_types = {}
            for req in failed_requests:
                error = req.get('error', f"HTTP {req['status_code']}")
                error_types[error] = error_types.get(error, 0) + 1
            
            for error, count in error_types.items():
                print(f"  {error}: {count}次")
        
        # 性能评估
        print()
        print("🎯 性能评估:")
        
        if response_times:
            avg_time = statistics.mean(response_times)
            success_rate = len(successful_requests) / len(results) * 100
            throughput = len(results) / total_time
            
            print(f"  平均响应时间: {'✅ 优秀' if avg_time < 1 else '⚠️ 需优化' if avg_time < 3 else '❌ 较慢'}")
            print(f"  成功率:       {'✅ 优秀' if success_rate >= 99 else '⚠️ 一般' if success_rate >= 95 else '❌ 较低'}")
            print(f"  吞吐量:       {'✅ 优秀' if throughput > 5 else '⚠️ 一般' if throughput > 2 else '❌ 较低'}")

def main():
    """主函数"""
    print("⚡ API 性能测试工具")
    print("=" * 50)
    
    # 检查API是否运行
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print("❌ API服务不健康，请检查服务状态")
            return
    except requests.exceptions.RequestException:
        print("❌ 无法连接到API服务，请确保服务运行在 http://localhost:8000")
        return
    
    # 创建测试实例
    test = PerformanceTest()
    
    # 运行测试
    print("1. 轻量级测试 (10请求, 2并发)")
    test.run_load_test(num_requests=10, concurrent_users=2)
    
    print("\n" + "="*50)
    print("2. 中等负载测试 (20请求, 5并发)")
    test.run_load_test(num_requests=20, concurrent_users=5)
    
    # 可选：重负载测试
    user_input = input("\n是否运行重负载测试？(50请求, 10并发) [y/N]: ")
    if user_input.lower() == 'y':
        print("\n" + "="*50)
        print("3. 重负载测试 (50请求, 10并发)")
        test.run_load_test(num_requests=50, concurrent_users=10)

if __name__ == "__main__":
    main()