import { useState, useEffect } from 'react';

const API_ENDPOINT = import.meta.env.VITE_API_ENDPOINT || 'https://your-api-gateway-url.execute-api.us-east-1.amazonaws.com';

export const useFraudAnalytics = () => {
  const [metrics, setMetrics] = useState(null);

  const [transactions, setTransactions] = useState([]);
  const [riskScores, setRiskScores] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState('connecting');
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await fetch(`${API_ENDPOINT}/api/fraud/metrics`);
        if (!response.ok) throw new Error('Failed to fetch fraud metrics');
        
        const data = await response.json();
        setMetrics(data);
        setConnectionStatus('connected');
        setError(null);
        
        const newTransactions = Array.from({ length: 5 }, (_, i) => ({
          id: `txn_${Date.now()}_${i}`,
          amount: Math.floor(Math.random() * 5000) + 100,
          riskScore: Math.floor(Math.random() * 100),
          status: Math.random() > 0.1 ? 'approved' : 'flagged',
          timestamp: new Date(Date.now() - Math.random() * 3600000).toISOString(),
          location: ['New York', 'London', 'Tokyo', 'Sydney'][Math.floor(Math.random() * 4)]
        }));
        setTransactions(prev => [...newTransactions, ...prev.slice(0, 45)]);

        const newRiskScores = Array.from({ length: 20 }, () => ({
          timestamp: Date.now() - Math.random() * 3600000,
          score: Math.random() * 100
        }));
        setRiskScores(newRiskScores);
        
      } catch (err) {
        setError(err.message);
        setConnectionStatus('disconnected');
        console.error('Fraud analytics fetch error:', err);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 10000);
    return () => clearInterval(interval);
  }, []);

  return {
    metrics,
    transactions,
    riskScores,
    connectionStatus,
    error
  };
};
