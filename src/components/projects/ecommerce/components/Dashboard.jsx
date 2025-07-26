import React, { useState, useEffect } from 'react';
import { useAnalytics } from '../../../../hooks/useAnalytics';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, PieChart, Pie, Cell
} from 'recharts';

// Custom hook for managing time-series data
const useTimeSeriesData = (initialData = []) => {
  const [data, setData] = useState(initialData);
  
  const addDataPoint = (newPoint) => {
    setData(prevData => {
      const newData = [...prevData, newPoint];
      // Keep last 20 data points for smooth visualization
      return newData.slice(-20);
    });
  };
  
  return [data, addDataPoint];
};

const Dashboard = () => {
  const { metrics, isConnected, error, formatCurrency } = useAnalytics();
  const [timeSeriesData, addTimeSeriesPoint] = useTimeSeriesData();
  const [selectedMetric, setSelectedMetric] = useState('revenue');
  
  // Update time series when metrics change
  useEffect(() => {
    const timestamp = new Date().toLocaleTimeString();
    addTimeSeriesPoint({
      timestamp,
      revenue: metrics.revenue,
      orders: metrics.orders,
      customers: metrics.customers,
      conversionRate: metrics.conversionRate
    });
  }, [metrics]);

  // Colors for consistent styling
  const COLORS = {
    revenue: '#ff9f43',
    orders: '#4834d4',
    customers: '#2e86de',
    conversion: '#00d2d3'
  };

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-gray-800 p-3 border border-gray-700 rounded-lg shadow-lg">
          <p className="text-gray-300 font-medium">{label}</p>
          {payload.map((entry, index) => (
            <p 
              key={index} 
              style={{ color: entry.color }}
              className="font-semibold"
            >
              {entry.name}: {entry.value.toLocaleString()}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-6">
      {/* Connection Status */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white">Real-time Analytics</h3>
        <div className={`px-3 py-1 rounded-full text-sm ${
          isConnected 
            ? 'bg-green-500/20 text-green-400' 
            : 'bg-yellow-500/20 text-yellow-400'
        }`}>
          {isConnected ? 'Connected' : 'Connecting...'}
        </div>
      </div>

      {error && (
        <div className="p-4 bg-red-500/10 border border-red-500 rounded-lg">
          <p className="text-red-500">{error}</p>
        </div>
      )}

      {/* Metric Selector */}
      <div className="flex gap-2">
        {Object.entries({
          revenue: 'Revenue',
          orders: 'Orders',
          customers: 'Customers',
          conversionRate: 'Conversion Rate'
        }).map(([key, label]) => (
          <button
            key={key}
            onClick={() => setSelectedMetric(key)}
            className={`px-3 py-1 rounded-full text-sm transition-colors ${
              selectedMetric === key
                ? 'bg-orange-400 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Time Series Chart */}
      <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={timeSeriesData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis 
              dataKey="timestamp" 
              stroke="#9CA3AF"
              tick={{ fill: '#9CA3AF' }}
            />
            <YAxis 
              stroke="#9CA3AF"
              tick={{ fill: '#9CA3AF' }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            <Line
              type="monotone"
              dataKey={selectedMetric}
              stroke={COLORS[selectedMetric.toLowerCase()]}
              strokeWidth={2}
              dot={false}
              name={selectedMetric.charAt(0).toUpperCase() + selectedMetric.slice(1)}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default Dashboard;