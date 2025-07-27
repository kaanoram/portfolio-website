import { useState, useEffect } from 'react';

export const useFraudAnalytics = () => {
  const [metrics, setMetrics] = useState({
    transactionsProcessed: "847K",
    fraudDetected: "1,247",
    accuracy: "96.8%",
    falsePositiveRate: "2.1%",
    avgProcessingTime: "12ms",
    riskScore: "23.4"
  });

  const [transactions, setTransactions] = useState([]);
  const [riskScores, setRiskScores] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState('connecting');
  const [error, setError] = useState(null);

  useEffect(() => {
    const simulateConnection = () => {
      setTimeout(() => {
        setConnectionStatus('connected');
        
        const interval = setInterval(() => {
          setMetrics(prev => ({
            ...prev,
            transactionsProcessed: `${Math.floor(847 + Math.random() * 10)}K`,
            fraudDetected: `${Math.floor(1247 + Math.random() * 50)}`,
            accuracy: `${(96.8 + (Math.random() - 0.5) * 0.4).toFixed(1)}%`,
            falsePositiveRate: `${(2.1 + (Math.random() - 0.5) * 0.3).toFixed(1)}%`,
            avgProcessingTime: `${Math.floor(12 + (Math.random() - 0.5) * 4)}ms`,
            riskScore: `${(23.4 + (Math.random() - 0.5) * 5).toFixed(1)}`
          }));

          const newTransaction = {
            id: `TXN-${Math.floor(Math.random() * 900000) + 100000}`,
            amount: (Math.random() * 5000).toFixed(2),
            riskScore: Math.floor(Math.random() * 100),
            timestamp: 'Just now'
          };
          
          setTransactions(prev => [newTransaction, ...prev.slice(0, 9)]);
        }, 4000);

        return () => clearInterval(interval);
      }, 1000);
    };

    simulateConnection();
  }, []);

  return {
    metrics,
    transactions,
    riskScores,
    connectionStatus,
    error
  };
};
