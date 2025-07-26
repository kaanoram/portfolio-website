// Demo Analytics Hook for Cost-Optimized Backend
import { useState, useEffect, useCallback } from 'react';

const API_ENDPOINT = import.meta.env.VITE_API_ENDPOINT || 'https://your-api-gateway-url.execute-api.us-east-1.amazonaws.com';
const DEMO_MODE = import.meta.env.VITE_DEMO_MODE !== 'false';

export const useDemoAnalytics = () => {
  const [transactions, setTransactions] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [capabilities, setCapabilities] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);

  // Fetch system capabilities on mount
  useEffect(() => {
    const fetchCapabilities = async () => {
      try {
        const response = await fetch(`${API_ENDPOINT}/api/capabilities`);
        const data = await response.json();
        setCapabilities(data);
      } catch (err) {
        console.error('Failed to fetch capabilities:', err);
      }
    };

    fetchCapabilities();
  }, []);

  // Generate demo transaction
  const generateTransaction = useCallback(async () => {
    if (isGenerating) return;
    
    setIsGenerating(true);
    try {
      const response = await fetch(`${API_ENDPOINT}/api/demo/transaction`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      
      if (!response.ok) throw new Error('Failed to generate transaction');
      
      const transaction = await response.json();
      
      // Add to local state
      setTransactions(prev => {
        const updated = [...prev, transaction];
        // Keep only last 50 transactions
        return updated.slice(-50);
      });
      
      setIsConnected(true);
      setError(null);
    } catch (err) {
      setError(err.message);
      console.error('Transaction generation error:', err);
    } finally {
      setIsGenerating(false);
    }
  }, [isGenerating]);

  // Fetch metrics periodically
  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await fetch(`${API_ENDPOINT}/api/demo/metrics`);
        const data = await response.json();
        setMetrics(data);
      } catch (err) {
        console.error('Failed to fetch metrics:', err);
      }
    };

    // Initial fetch
    fetchMetrics();

    // Refresh every 10 seconds
    const interval = setInterval(fetchMetrics, 10000);

    return () => clearInterval(interval);
  }, []);

  // Auto-generate transactions for demo
  useEffect(() => {
    if (!DEMO_MODE) return;

    // Generate a transaction every 2-5 seconds
    const generateRandomTransaction = () => {
      generateTransaction();
      
      // Schedule next transaction
      const delay = Math.random() * 3000 + 2000; // 2-5 seconds
      setTimeout(generateRandomTransaction, delay);
    };

    // Start after a short delay
    const timeout = setTimeout(generateRandomTransaction, 1000);

    return () => clearTimeout(timeout);
  }, [generateTransaction]);

  // Calculate real-time statistics
  const getRealtimeStats = useCallback(() => {
    const recentTransactions = transactions.slice(-20);
    
    if (recentTransactions.length === 0) {
      return {
        avgPurchaseProbability: 0,
        riskDistribution: { high_risk: 0, medium_risk: 0, low_risk: 0 },
        totalRevenue: 0,
        avgInferenceTime: 0
      };
    }

    return {
      avgPurchaseProbability: 
        recentTransactions.reduce((sum, t) => sum + t.purchase_probability, 0) / recentTransactions.length,
      riskDistribution: recentTransactions.reduce((acc, t) => {
        acc[t.risk_category] = (acc[t.risk_category] || 0) + 1;
        return acc;
      }, {}),
      totalRevenue: recentTransactions.reduce((sum, t) => sum + t.amount, 0),
      avgInferenceTime: 
        recentTransactions.reduce((sum, t) => sum + (t.inference_time_ms || 0), 0) / recentTransactions.length
    };
  }, [transactions]);

  return {
    // Data
    transactions,
    metrics,
    capabilities,
    realtimeStats: getRealtimeStats(),
    
    // State
    isConnected,
    error,
    isGenerating,
    
    // Actions
    generateTransaction,
    clearTransactions: () => setTransactions([]),
    
    // Demo info
    isDemoMode: DEMO_MODE,
    demoInfo: {
      message: "This is a live demo with real ML predictions running on AWS Lambda",
      costInfo: "Infrastructure cost: ~$10-20/month",
      fullScaleInfo: "Full production system supports 1M+ daily transactions"
    }
  };
};