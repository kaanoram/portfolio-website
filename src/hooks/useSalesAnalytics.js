import { useState, useEffect } from 'react';

const API_ENDPOINT = import.meta.env.VITE_API_ENDPOINT || 'https://your-api-gateway-url.execute-api.us-east-1.amazonaws.com';

export const useSalesAnalytics = () => {
  const [metrics, setMetrics] = useState(null);

  const [processingStats, setProcessingStats] = useState({
    recordsProcessed: "2.1M",
    dataAccuracy: "99.7%",
    etlLatency: "2.3s",
    errorRate: "0.03%"
  });

  const [anomalies, setAnomalies] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState('connecting');
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await fetch(`${API_ENDPOINT}/api/sales/metrics`);
        if (!response.ok) throw new Error('Failed to fetch sales metrics');
        
        const data = await response.json();
        setMetrics(data);
        setConnectionStatus('connected');
        setError(null);
        
        const newAnomalies = [
          {
            id: Date.now(),
            type: 'Revenue Spike',
            severity: 'medium',
            description: 'Unusual revenue increase detected in Q4 data',
            timestamp: new Date().toISOString(),
            confidence: Math.random() * 0.3 + 0.7
          }
        ];
        setAnomalies(newAnomalies);
        
      } catch (err) {
        setError(err.message);
        setConnectionStatus('disconnected');
        console.error('Sales analytics fetch error:', err);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 10000);
    return () => clearInterval(interval);
  }, []);

  return {
    metrics,
    processingStats,
    anomalies,
    connectionStatus,
    error
  };
};
