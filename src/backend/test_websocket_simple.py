#!/usr/bin/env python3
"""
Simple WebSocket test to verify server functionality
"""

import asyncio
import websockets
import json
import time

async def test_websocket():
    uri = "ws://localhost:8000/ws"
    
    print("Connecting to WebSocket server...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✓ Connected successfully!")
            
            # Listen for messages for 10 seconds
            print("\nListening for messages (10 seconds)...")
            
            start_time = time.time()
            message_count = 0
            
            while time.time() - start_time < 10:
                try:
                    # Set timeout to avoid blocking forever
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    message_count += 1
                    
                    print(f"\nMessage #{message_count}:")
                    print(f"  Type: {data.get('type', 'unknown')}")
                    
                    if 'transaction' in data:
                        trans = data['transaction']
                        print(f"  Customer: {trans.get('customer_id')}")
                        print(f"  Amount: ${trans.get('amount', 0):.2f}")
                        print(f"  Prediction: {trans.get('purchase_probability', 0):.3f}")
                    
                    if 'metrics' in data:
                        metrics = data['metrics']
                        print(f"  Total Revenue: ${metrics.get('revenue', 0):.2f}")
                        print(f"  Total Orders: {metrics.get('orders', 0)}")
                        
                except asyncio.TimeoutError:
                    # No message received in 1 second, continue
                    continue
                except Exception as e:
                    print(f"Error parsing message: {e}")
            
            print(f"\n✓ Test completed!")
            print(f"  Total messages received: {message_count}")
            print(f"  Average rate: {message_count/10:.1f} messages/second")
            
            if message_count == 0:
                print("\n⚠️  No messages received. Check if server is sending data.")
            
    except Exception as e:
        print(f"\n✗ Connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure server is running: python server.py")
        print("2. Check the WebSocket URL is correct")
        print("3. Check server logs for errors")

if __name__ == "__main__":
    asyncio.run(test_websocket())