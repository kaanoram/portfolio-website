#!/bin/bash
# Script to run comprehensive load tests and generate results for portfolio

echo "======================================"
echo "E-commerce Analytics Load Test Suite"
echo "======================================"
echo ""

# Configuration
WEBSOCKET_URL="ws://localhost:8000/ws"
API_URL="http://localhost:8000"
RESULTS_DIR="load_test_results"

# Create results directory
mkdir -p $RESULTS_DIR

# Function to check if server is running
check_server() {
    echo "Checking if server is running..."
    curl -s -o /dev/null -w "%{http_code}" $API_URL/health || {
        echo "❌ Server is not running. Please start the server first:"
        echo "   cd src/backend && python server.py"
        exit 1
    }
    echo "✓ Server is running"
}

# Function to run WebSocket load test
run_websocket_test() {
    echo ""
    echo "1. Running WebSocket Load Test (2000+ connections)..."
    echo "=================================================="
    
    python load_test_websocket.py \
        --url $WEBSOCKET_URL \
        --connections 2000 \
        --ramp-up 100 \
        --output $RESULTS_DIR/websocket_results.json
    
    if [ $? -eq 0 ]; then
        echo "✓ WebSocket test completed"
    else
        echo "❌ WebSocket test failed"
    fi
}

# Function to run API load test
run_api_test() {
    echo ""
    echo "2. Running API Load Test (Multiple Scenarios)..."
    echo "=============================================="
    
    python load_test_api.py \
        --url $API_URL \
        --output $RESULTS_DIR/api_results.json
    
    if [ $? -eq 0 ]; then
        echo "✓ API test completed"
    else
        echo "❌ API test failed"
    fi
}

# Function to generate summary report
generate_summary() {
    echo ""
    echo "3. Generating Summary Report..."
    echo "=============================="
    
    python -c "
import json
import os

results_dir = '$RESULTS_DIR'

# Load results
ws_results = {}
api_results = {}

ws_path = os.path.join(results_dir, 'websocket_results.json')
api_path = os.path.join(results_dir, 'api_results.json')

if os.path.exists(ws_path):
    with open(ws_path) as f:
        ws_results = json.load(f)

if os.path.exists(api_path):
    with open(api_path) as f:
        api_results = json.load(f)

# Create summary
summary = {
    'test_suite': 'E-commerce Analytics Platform - Performance Validation',
    'websocket_summary': {
        'passed': ws_results.get('test_summary', {}).get('successful_connections', 0) >= 2000,
        'connections': ws_results.get('test_summary', {}).get('successful_connections', 0),
        'avg_latency_ms': ws_results.get('message_metrics', {}).get('avg_latency_ms', 0)
    },
    'api_summary': {
        'passed': api_results.get('summary', {}).get('production_readiness', False),
        'max_throughput_rps': api_results.get('summary', {}).get('max_throughput_achieved', '0'),
        'daily_capacity': float(api_results.get('summary', {}).get('max_throughput_achieved', '0').replace(' req/s', '')) * 86400
    },
    'overall_result': 'PASSED' if (
        ws_results.get('test_summary', {}).get('successful_connections', 0) >= 2000 and
        api_results.get('summary', {}).get('production_readiness', False)
    ) else 'FAILED'
}

# Save summary
with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

# Print results
print('\\n=== PERFORMANCE TEST SUMMARY ===')
print(f'\\nWebSocket Test: {\"PASSED\" if summary[\"websocket_summary\"][\"passed\"] else \"FAILED\"}')
print(f'  - Connections: {summary[\"websocket_summary\"][\"connections\"]}')
print(f'  - Avg Latency: {summary[\"websocket_summary\"][\"avg_latency_ms\"]}ms')

print(f'\\nAPI Test: {\"PASSED\" if summary[\"api_summary\"][\"passed\"] else \"FAILED\"}')
print(f'  - Max Throughput: {summary[\"api_summary\"][\"max_throughput_rps\"]}')
print(f'  - Daily Capacity: {summary[\"api_summary\"][\"daily_capacity\"]:,.0f} transactions')

print(f'\\n🎯 Overall Result: {summary[\"overall_result\"]}')
print(f'\\nFull results saved to: {results_dir}/')
"
}

# Function to create portfolio-ready results
create_portfolio_results() {
    echo ""
    echo "4. Creating Portfolio-Ready Results..."
    echo "===================================="
    
    # Copy results to frontend
    cp $RESULTS_DIR/summary.json ../../../public/performance_results.json
    
    echo "✓ Results copied to public/performance_results.json"
    echo ""
    echo "To use in your portfolio:"
    echo "1. The results are now available at /performance_results.json"
    echo "2. Update PerformanceMetrics.jsx with actual data"
    echo "3. Include load test screenshots in your portfolio"
}

# Main execution
main() {
    echo "Starting load test suite..."
    echo "This will take approximately 5-10 minutes"
    echo ""
    
    # Check prerequisites
    check_server
    
    # Run tests
    run_websocket_test
    run_api_test
    
    # Generate reports
    generate_summary
    create_portfolio_results
    
    echo ""
    echo "======================================"
    echo "Load Testing Complete!"
    echo "======================================"
    echo ""
    echo "Next steps:"
    echo "1. Review results in: $RESULTS_DIR/"
    echo "2. Update your portfolio with the test results"
    echo "3. Take screenshots of the running tests for documentation"
    echo "4. Consider running tests on deployed infrastructure for cloud metrics"
}

# Run main function
main