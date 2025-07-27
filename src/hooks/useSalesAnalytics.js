import { useState, useEffect } from 'react';

export const useSalesAnalytics = () => {
  const [metrics, setMetrics] = useState({
    totalRevenue: "$2.4M",
    salesGrowth: "18.3%",
    avgDealSize: "$45.2K",
    conversionRate: "24.7%",
    activeLeads: "1,247",
    pipelineValue: "$8.9M"
  });

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
    const simulateConnection = () => {
      setTimeout(() => {
        setConnectionStatus('connected');
        
        const interval = setInterval(() => {
          setMetrics(prev => ({
            ...prev,
            totalRevenue: `$${(2.4 + Math.random() * 0.2).toFixed(1)}M`,
            salesGrowth: `${(18.3 + (Math.random() - 0.5) * 2).toFixed(1)}%`,
            avgDealSize: `$${(45.2 + (Math.random() - 0.5) * 5).toFixed(1)}K`,
            conversionRate: `${(24.7 + (Math.random() - 0.5) * 2).toFixed(1)}%`,
            activeLeads: `${Math.floor(1247 + (Math.random() - 0.5) * 100)}`,
            pipelineValue: `$${(8.9 + Math.random() * 0.5).toFixed(1)}M`
          }));

          setProcessingStats(prev => ({
            ...prev,
            recordsProcessed: `${(2.1 + Math.random() * 0.1).toFixed(1)}M`,
            dataAccuracy: `${(99.7 + (Math.random() - 0.5) * 0.2).toFixed(1)}%`,
            etlLatency: `${(2.3 + (Math.random() - 0.5) * 0.5).toFixed(1)}s`,
            errorRate: `${(0.03 + Math.random() * 0.02).toFixed(2)}%`
          }));
        }, 3000);

        return () => clearInterval(interval);
      }, 1000);
    };

    simulateConnection();
  }, []);

  return {
    metrics,
    processingStats,
    anomalies,
    connectionStatus,
    error
  };
};
