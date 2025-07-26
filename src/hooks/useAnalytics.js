import { useState, useEffect, useCallback, useRef } from 'react';

export const useAnalytics = () => {
    // State management
    const [metrics, setMetrics] = useState({
        revenue: 0,
        customers: 0,
        orders: 0,
        conversionRate: 0,
        ordersPerSecond: 0
    });
    const [transactions, setTransactions] = useState([]);
    const [isConnected, setIsConnected] = useState(false);
    const [error, setError] = useState(null);
    
    // WebSocket reference
    const wsRef = useRef(null);
    const reconnectAttempts = useRef(0);
    const maxReconnectAttempts = 5;
    
    // Reconnection function
    const connect = useCallback(() => {
        try {
            const ws = new WebSocket('ws://localhost:8000/ws');
            
            ws.onopen = () => {
                console.log('Connected to analytics server');
                setIsConnected(true);
                setError(null);
                reconnectAttempts.current = 0;
            };

            ws.onclose = () => {
                console.log('Disconnected from analytics server');
                setIsConnected(false);
                
                // Attempt to reconnect if we haven't exceeded max attempts
                if (reconnectAttempts.current < maxReconnectAttempts) {
                    reconnectAttempts.current += 1;
                    const timeout = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 10000);
                    setTimeout(() => {
                        console.log(`Attempting to reconnect (${reconnectAttempts.current}/${maxReconnectAttempts})`);
                        connect();
                    }, timeout);
                } else {
                    setError('Unable to connect to the analytics server after multiple attempts');
                }
            };

            ws.onerror = (event) => {
                console.error("WebSocket error:", event);
                setError('Error connecting to analytics server');
            };

            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    
                    // Update metrics if provided
                    if (data.metrics) {
                        setMetrics(prevMetrics => ({
                            ...prevMetrics,
                            ...data.metrics
                        }));
                    }
                    
                    // Update transactions if provided
                    if (data.transaction) {
                        setTransactions(prev => {
                            const newTransactions = [data.transaction, ...prev];
                            return newTransactions.slice(0, 5); // Keep only latest 5
                        });
                    }
                } catch (e) {
                    console.error('Error parsing websocket message:', e);
                }
            };

            wsRef.current = ws;
            
            // Cleanup function
            return () => {
                if (wsRef.current) {
                    wsRef.current.close();
                }
            };
        } catch (err) {
            setError(`Failed to establish WebSocket connection: ${err.message}`);
        }
    }, []);

    // Initialize connection
    useEffect(() => {
        connect();
        
        return () => {
            if (wsRef.current) {
                wsRef.current.close();
            }
        };
    }, [connect]);

    // Utility function for formatting currency
    const formatCurrency = useCallback((value) => {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD'
        }).format(value);
    }, []);

    return {
        metrics,
        transactions,
        isConnected,
        error,
        formatCurrency
    };
};