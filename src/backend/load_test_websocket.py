#!/usr/bin/env python3
"""
Load testing script for WebSocket server.
Tests the system's ability to handle 2000+ concurrent connections.
"""

import asyncio
import aiohttp
import time
import json
import statistics
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import logging
from datetime import datetime
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ConnectionMetrics:
    """Metrics for a single WebSocket connection"""
    connection_id: int
    connect_time: float = 0
    first_message_time: float = 0
    messages_received: int = 0
    errors: int = 0
    latencies: List[float] = None
    status: str = "pending"
    
    def __post_init__(self):
        if self.latencies is None:
            self.latencies = []
    
    @property
    def avg_latency(self) -> float:
        return statistics.mean(self.latencies) if self.latencies else 0
    
    @property
    def p95_latency(self) -> float:
        if not self.latencies:
            return 0
        sorted_latencies = sorted(self.latencies)
        index = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[index] if index < len(sorted_latencies) else sorted_latencies[-1]


class WebSocketLoadTester:
    """Load tester for WebSocket connections"""
    
    def __init__(self, url: str, target_connections: int = 2000):
        self.url = url
        self.target_connections = target_connections
        self.metrics: Dict[int, ConnectionMetrics] = {}
        self.start_time = None
        self.end_time = None
        self.successful_connections = 0
        self.failed_connections = 0
        
    async def create_connection(self, session: aiohttp.ClientSession, connection_id: int) -> None:
        """Create and maintain a single WebSocket connection"""
        metrics = ConnectionMetrics(connection_id=connection_id)
        self.metrics[connection_id] = metrics
        
        try:
            connect_start = time.time()
            
            async with session.ws_connect(self.url) as ws:
                metrics.connect_time = time.time() - connect_start
                metrics.status = "connected"
                self.successful_connections += 1
                
                logger.debug(f"Connection {connection_id} established in {metrics.connect_time:.3f}s")
                
                # Listen for messages
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        message_time = time.time()
                        
                        # First message latency
                        if metrics.messages_received == 0:
                            metrics.first_message_time = message_time - connect_start
                        
                        # Parse message and calculate latency
                        try:
                            data = json.loads(msg.data)
                            if 'timestamp' in data:
                                # Calculate latency from server timestamp
                                server_time = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
                                latency = (message_time - server_time.timestamp()) * 1000  # ms
                                metrics.latencies.append(latency)
                        except Exception as e:
                            logger.debug(f"Error parsing message: {e}")
                        
                        metrics.messages_received += 1
                        
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        metrics.errors += 1
                        logger.error(f"Connection {connection_id} error: {ws.exception()}")
                        
                    elif msg.type == aiohttp.WSMsgType.CLOSED:
                        break
                        
        except Exception as e:
            metrics.status = "failed"
            metrics.errors += 1
            self.failed_connections += 1
            logger.error(f"Connection {connection_id} failed: {e}")
    
    async def run_load_test(self, connections_per_second: int = 100) -> Dict[str, Any]:
        """Run the load test with gradual connection ramp-up"""
        logger.info(f"Starting load test: {self.target_connections} connections")
        logger.info(f"Ramp-up rate: {connections_per_second} connections/second")
        
        self.start_time = time.time()
        
        # Create session with connection pooling
        connector = aiohttp.TCPConnector(limit=self.target_connections)
        timeout = aiohttp.ClientTimeout(total=300)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = []
            
            # Gradual ramp-up
            for i in range(self.target_connections):
                task = asyncio.create_task(self.create_connection(session, i))
                tasks.append(task)
                
                # Rate limiting
                if (i + 1) % connections_per_second == 0:
                    await asyncio.sleep(1)
                    logger.info(f"Connections created: {i + 1}/{self.target_connections}")
            
            # Wait for all connections to complete
            logger.info("All connections initiated, maintaining connections...")
            
            # Keep connections alive for test duration
            await asyncio.sleep(30)  # 30 seconds of sustained load
            
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for cleanup
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self.end_time = time.time()
        
        return self.generate_report()
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        total_duration = self.end_time - self.start_time
        
        # Connection statistics
        connection_times = [m.connect_time for m in self.metrics.values() if m.connect_time > 0]
        successful_metrics = [m for m in self.metrics.values() if m.status == "connected"]
        
        # Message statistics
        all_latencies = []
        for m in successful_metrics:
            all_latencies.extend(m.latencies)
        
        # Calculate aggregates
        report = {
            "test_summary": {
                "target_connections": self.target_connections,
                "successful_connections": self.successful_connections,
                "failed_connections": self.failed_connections,
                "success_rate": f"{(self.successful_connections / self.target_connections * 100):.1f}%",
                "total_duration_seconds": round(total_duration, 2),
                "test_timestamp": datetime.now().isoformat()
            },
            
            "connection_metrics": {
                "avg_connection_time": round(statistics.mean(connection_times), 3) if connection_times else 0,
                "min_connection_time": round(min(connection_times), 3) if connection_times else 0,
                "max_connection_time": round(max(connection_times), 3) if connection_times else 0,
                "p95_connection_time": round(self._percentile(connection_times, 95), 3) if connection_times else 0,
                "p99_connection_time": round(self._percentile(connection_times, 99), 3) if connection_times else 0
            },
            
            "message_metrics": {
                "total_messages_received": sum(m.messages_received for m in successful_metrics),
                "avg_messages_per_connection": round(
                    statistics.mean([m.messages_received for m in successful_metrics]), 1
                ) if successful_metrics else 0,
                "avg_latency_ms": round(statistics.mean(all_latencies), 2) if all_latencies else 0,
                "p50_latency_ms": round(self._percentile(all_latencies, 50), 2) if all_latencies else 0,
                "p95_latency_ms": round(self._percentile(all_latencies, 95), 2) if all_latencies else 0,
                "p99_latency_ms": round(self._percentile(all_latencies, 99), 2) if all_latencies else 0
            },
            
            "performance_assessment": self._assess_performance(
                self.successful_connections, 
                all_latencies
            )
        }
        
        return report
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile from data"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _assess_performance(self, successful_connections: int, latencies: List[float]) -> Dict[str, Any]:
        """Assess if performance meets requirements"""
        avg_latency = statistics.mean(latencies) if latencies else float('inf')
        p99_latency = self._percentile(latencies, 99) if latencies else float('inf')
        
        return {
            "meets_connection_target": successful_connections >= 2000,
            "meets_latency_target": avg_latency < 100,  # <100ms average
            "meets_p99_target": p99_latency < 500,  # <500ms p99
            "production_ready": (
                successful_connections >= 2000 and 
                avg_latency < 100 and 
                p99_latency < 500
            ),
            "recommendations": self._get_recommendations(successful_connections, avg_latency)
        }
    
    def _get_recommendations(self, connections: int, avg_latency: float) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        if connections < 2000:
            recommendations.append(f"Scale up to handle {2000 - connections} more connections")
        
        if avg_latency > 100:
            recommendations.append("Optimize message processing to reduce latency")
            
        if not recommendations:
            recommendations.append("System meets all performance targets!")
            
        return recommendations


async def main():
    parser = argparse.ArgumentParser(description='WebSocket Load Testing Tool')
    parser.add_argument('--url', default='ws://localhost:8000/ws', help='WebSocket URL')
    parser.add_argument('--connections', type=int, default=2000, help='Target connections')
    parser.add_argument('--ramp-up', type=int, default=100, help='Connections per second')
    parser.add_argument('--output', default='load_test_results.json', help='Output file')
    
    args = parser.parse_args()
    
    # Create load tester
    tester = WebSocketLoadTester(args.url, args.connections)
    
    # Run test
    logger.info("=" * 60)
    logger.info("WebSocket Load Test Starting")
    logger.info("=" * 60)
    
    try:
        report = await tester.run_load_test(args.ramp_up)
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("LOAD TEST RESULTS")
        logger.info("=" * 60)
        
        logger.info(f"\nConnection Summary:")
        logger.info(f"  Target: {report['test_summary']['target_connections']}")
        logger.info(f"  Successful: {report['test_summary']['successful_connections']}")
        logger.info(f"  Failed: {report['test_summary']['failed_connections']}")
        logger.info(f"  Success Rate: {report['test_summary']['success_rate']}")
        
        logger.info(f"\nLatency Metrics:")
        logger.info(f"  Average: {report['message_metrics']['avg_latency_ms']}ms")
        logger.info(f"  P95: {report['message_metrics']['p95_latency_ms']}ms")
        logger.info(f"  P99: {report['message_metrics']['p99_latency_ms']}ms")
        
        logger.info(f"\nPerformance Assessment:")
        assessment = report['performance_assessment']
        logger.info(f"  Meets Connection Target (2000+): {'✓' if assessment['meets_connection_target'] else '✗'}")
        logger.info(f"  Meets Latency Target (<100ms): {'✓' if assessment['meets_latency_target'] else '✗'}")
        logger.info(f"  Production Ready: {'✓' if assessment['production_ready'] else '✗'}")
        
        # Save detailed report
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\nDetailed report saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"Load test failed: {e}")
        raise


if __name__ == '__main__':
    asyncio.run(main())