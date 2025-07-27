import React from 'react';
import { Shield, AlertTriangle, Clock, Target, TrendingUp, Activity } from 'lucide-react';

const MetricCard = ({ icon: Icon, label, value, change, status }) => (
  <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
    <div className="flex items-center justify-between mb-4">
      <Icon className="w-8 h-8 text-orange-500" />
      <span className={`text-sm px-2 py-1 rounded ${
        status === 'good' ? 'bg-green-900 text-green-300' : 
        status === 'warning' ? 'bg-yellow-900 text-yellow-300' : 
        status === 'critical' ? 'bg-red-900 text-red-300' :
        'bg-gray-700 text-gray-300'
      }`}>
        {change}
      </span>
    </div>
    <h3 className="text-2xl font-bold text-white mb-1">{value}</h3>
    <p className="text-gray-400 text-sm">{label}</p>
  </div>
);

const FraudMetrics = ({ metrics, connectionStatus, error }) => {
  if (error) {
    return (
      <div className="bg-red-900/20 border border-red-500 rounded-lg p-4">
        <p className="text-red-400">Unable to connect to the fraud detection server after multiple attempts</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-xl font-semibold text-white">Fraud Detection Performance</h3>
        <div className="flex items-center space-x-2">
          <div className={`w-2 h-2 rounded-full ${
            connectionStatus === 'connected' ? 'bg-green-500' : 
            connectionStatus === 'connecting' ? 'bg-yellow-500' : 'bg-red-500'
          }`}></div>
          <span className="text-sm text-gray-400 capitalize">{connectionStatus}...</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <MetricCard
          icon={Activity}
          label="Transactions Processed"
          value={metrics?.transactionsProcessed || "847K"}
          change="Real-time"
          status="good"
        />
        <MetricCard
          icon={AlertTriangle}
          label="Fraud Detected"
          value={metrics?.fraudDetected || "1,247"}
          change="+23"
          status="warning"
        />
        <MetricCard
          icon={Target}
          label="Detection Accuracy"
          value={metrics?.accuracy || "96.8%"}
          change="+0.3%"
          status="good"
        />
        <MetricCard
          icon={Shield}
          label="False Positive Rate"
          value={metrics?.falsePositiveRate || "2.1%"}
          change="-0.2%"
          status="good"
        />
        <MetricCard
          icon={Clock}
          label="Avg Processing Time"
          value={metrics?.avgProcessingTime || "12ms"}
          change="< 50ms SLA"
          status="good"
        />
        <MetricCard
          icon={TrendingUp}
          label="Avg Risk Score"
          value={metrics?.riskScore || "23.4"}
          change="Low Risk"
          status="good"
        />
      </div>
    </div>
  );
};

export default FraudMetrics;
