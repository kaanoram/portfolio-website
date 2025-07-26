#!/usr/bin/env python3
"""
Load testing script for the demo API (Lambda).
Tests throughput and latency under various load conditions.
"""

import asyncio
import aiohttp
import time
import json
import statistics
from datetime import datetime
from typing import List, Dict, Any
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class APILoadTester:
    """Load tester for HTTP API endpoints"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip('/')
        self.results = {
            'successful_requests': 0,
            'failed_requests': 0,
            'latencies': [],
            'errors': [],
            'start_time': None,
            'end_time': None
        }
    
    async def make_request(self, session: aiohttp.ClientSession, endpoint: str, method: str = 'GET') -> Dict[str, Any]:
        """Make a single API request and measure latency"""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            async with session.request(method, url) as response:
                latency = (time.time() - start_time) * 1000  # Convert to ms
                data = await response.json()
                
                result = {
                    'success': response.status == 200,
                    'status_code': response.status,
                    'latency_ms': latency,
                    'data': data,
                    'timestamp': datetime.now().isoformat()
                }
                
                if response.status == 200:
                    self.results['successful_requests'] += 1
                    self.results['latencies'].append(latency)
                else:
                    self.results['failed_requests'] += 1
                    self.results['errors'].append(f"HTTP {response.status}")
                
                return result
                
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            self.results['failed_requests'] += 1
            self.results['errors'].append(str(e))
            
            return {
                'success': False,
                'error': str(e),
                'latency_ms': latency,
                'timestamp': datetime.now().isoformat()
            }
    
    async def load_test_endpoint(
        self, 
        endpoint: str, 
        total_requests: int, 
        concurrent_requests: int,
        method: str = 'GET'
    ) -> Dict[str, Any]:
        """Run load test on a specific endpoint"""
        logger.info(f"Testing {method} {endpoint}")
        logger.info(f"Total requests: {total_requests}, Concurrent: {concurrent_requests}")
        
        self.results['start_time'] = time.time()
        
        # Create session
        connector = aiohttp.TCPConnector(limit=concurrent_requests)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Create batches
            tasks = []
            
            for i in range(total_requests):
                task = self.make_request(session, endpoint, method)
                tasks.append(task)
                
                # Process in batches
                if len(tasks) >= concurrent_requests:
                    await asyncio.gather(*tasks)
                    tasks = []
                    
                    # Progress update
                    if (i + 1) % 100 == 0:
                        logger.info(f"Progress: {i + 1}/{total_requests} requests completed")
            
            # Process remaining tasks
            if tasks:
                await asyncio.gather(*tasks)
        
        self.results['end_time'] = time.time()
        
        return self._generate_report(endpoint, total_requests, concurrent_requests)
    
    def _generate_report(self, endpoint: str, total_requests: int, concurrent_requests: int) -> Dict[str, Any]:
        """Generate load test report"""
        duration = self.results['end_time'] - self.results['start_time']
        throughput = total_requests / duration if duration > 0 else 0
        
        latencies = self.results['latencies']
        
        report = {
            'test_info': {
                'endpoint': endpoint,
                'total_requests': total_requests,
                'concurrent_requests': concurrent_requests,
                'duration_seconds': round(duration, 2),
                'timestamp': datetime.now().isoformat()
            },
            
            'results': {
                'successful_requests': self.results['successful_requests'],
                'failed_requests': self.results['failed_requests'],
                'success_rate': f"{(self.results['successful_requests'] / total_requests * 100):.1f}%",
                'throughput_rps': round(throughput, 2)
            },
            
            'latency_metrics': {
                'min_ms': round(min(latencies), 2) if latencies else 0,
                'max_ms': round(max(latencies), 2) if latencies else 0,
                'avg_ms': round(statistics.mean(latencies), 2) if latencies else 0,
                'median_ms': round(statistics.median(latencies), 2) if latencies else 0,
                'p95_ms': round(self._percentile(latencies, 95), 2) if latencies else 0,
                'p99_ms': round(self._percentile(latencies, 99), 2) if latencies else 0
            },
            
            'errors': {
                'count': len(self.results['errors']),
                'types': list(set(self.results['errors']))[:10]  # First 10 unique errors
            }
        }
        
        return report
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


async def run_comprehensive_test(base_url: str) -> Dict[str, Any]:
    """Run comprehensive load tests on all endpoints"""
    
    test_scenarios = [
        {
            'name': 'Light Load',
            'endpoint': '/api/demo/transaction',
            'total_requests': 100,
            'concurrent_requests': 10,
            'method': 'POST'
        },
        {
            'name': 'Medium Load',
            'endpoint': '/api/demo/transaction',
            'total_requests': 500,
            'concurrent_requests': 50,
            'method': 'POST'
        },
        {
            'name': 'Heavy Load',
            'endpoint': '/api/demo/transaction',
            'total_requests': 1000,
            'concurrent_requests': 100,
            'method': 'POST'
        },
        {
            'name': 'Burst Test',
            'endpoint': '/api/demo/transaction',
            'total_requests': 200,
            'concurrent_requests': 200,  # All at once
            'method': 'POST'
        },
        {
            'name': 'Sustained Load',
            'endpoint': '/api/demo/metrics',
            'total_requests': 1000,
            'concurrent_requests': 20,
            'method': 'GET'
        }
    ]
    
    all_results = {
        'test_suite': 'E-commerce Analytics API Load Test',
        'timestamp': datetime.now().isoformat(),
        'base_url': base_url,
        'scenarios': []
    }
    
    for scenario in test_scenarios:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running scenario: {scenario['name']}")
        logger.info('='*60)
        
        tester = APILoadTester(base_url)
        
        result = await tester.load_test_endpoint(
            scenario['endpoint'],
            scenario['total_requests'],
            scenario['concurrent_requests'],
            scenario['method']
        )
        
        result['scenario_name'] = scenario['name']
        all_results['scenarios'].append(result)
        
        # Print summary
        logger.info(f"\nResults for {scenario['name']}:")
        logger.info(f"  Success Rate: {result['results']['success_rate']}")
        logger.info(f"  Throughput: {result['results']['throughput_rps']} req/s")
        logger.info(f"  Avg Latency: {result['latency_metrics']['avg_ms']}ms")
        logger.info(f"  P99 Latency: {result['latency_metrics']['p99_ms']}ms")
        
        # Brief pause between scenarios
        await asyncio.sleep(2)
    
    # Generate summary
    all_results['summary'] = generate_test_summary(all_results['scenarios'])
    
    return all_results


def generate_test_summary(scenarios: List[Dict]) -> Dict[str, Any]:
    """Generate overall test summary"""
    
    # Aggregate metrics
    total_requests = sum(s['test_info']['total_requests'] for s in scenarios)
    total_successful = sum(s['results']['successful_requests'] for s in scenarios)
    all_latencies = []
    
    for scenario in scenarios:
        if scenario['latency_metrics']['avg_ms'] > 0:
            # Approximate latencies for summary
            all_latencies.extend([scenario['latency_metrics']['avg_ms']] * scenario['results']['successful_requests'])
    
    # Performance assessment
    avg_latency = statistics.mean(all_latencies) if all_latencies else 0
    max_throughput = max(s['results']['throughput_rps'] for s in scenarios)
    
    return {
        'total_requests_tested': total_requests,
        'overall_success_rate': f"{(total_successful / total_requests * 100):.1f}%",
        'max_throughput_achieved': f"{max_throughput:.1f} req/s",
        'overall_avg_latency': f"{avg_latency:.1f}ms",
        
        'performance_targets': {
            'supports_1m_daily_transactions': max_throughput > 11.6,  # 1M/day = 11.6 req/s
            'sub_100ms_latency': avg_latency < 100,
            'high_availability': (total_successful / total_requests) > 0.99
        },
        
        'production_readiness': all([
            max_throughput > 11.6,
            avg_latency < 100,
            (total_successful / total_requests) > 0.99
        ]),
        
        'recommendations': [
            "System successfully handles required load" if max_throughput > 11.6 else "Scale up to meet throughput requirements",
            "Latency meets targets" if avg_latency < 100 else "Optimize for lower latency",
            "High availability achieved" if (total_successful / total_requests) > 0.99 else "Improve error handling"
        ]
    }


async def main():
    parser = argparse.ArgumentParser(description='API Load Testing Tool')
    parser.add_argument('--url', default='http://localhost:8000', help='Base API URL')
    parser.add_argument('--output', default='api_load_test_results.json', help='Output file')
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("API LOAD TEST SUITE")
    logger.info("="*60)
    logger.info(f"Target: {args.url}")
    logger.info(f"Starting comprehensive load tests...\n")
    
    try:
        results = await run_comprehensive_test(args.url)
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print final summary
        logger.info("\n" + "="*60)
        logger.info("FINAL TEST SUMMARY")
        logger.info("="*60)
        
        summary = results['summary']
        logger.info(f"\nPerformance Targets:")
        for target, met in summary['performance_targets'].items():
            status = "✓" if met else "✗"
            logger.info(f"  {target}: {status}")
        
        logger.info(f"\nProduction Ready: {'YES' if summary['production_readiness'] else 'NO'}")
        
        logger.info(f"\nDetailed results saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Load test failed: {e}")
        raise


if __name__ == '__main__':
    asyncio.run(main())