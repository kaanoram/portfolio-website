import React from 'react';
import { AlertTriangle, TrendingDown, TrendingUp, Info } from 'lucide-react';

const AnomalyCard = ({ anomaly, timestamp }) => {
  const getSeverityIcon = (severity) => {
    switch (severity) {
      case 'high': return <AlertTriangle className="w-5 h-5 text-red-500" />;
      case 'medium': return <TrendingDown className="w-5 h-5 text-yellow-500" />;
      case 'low': return <Info className="w-5 h-5 text-blue-500" />;
      default: return <Info className="w-5 h-5 text-gray-500" />;
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'high': return 'border-red-500 bg-red-900/20';
      case 'medium': return 'border-yellow-500 bg-yellow-900/20';
      case 'low': return 'border-blue-500 bg-blue-900/20';
      default: return 'border-gray-500 bg-gray-900/20';
    }
  };

  return (
    <div className={`p-4 rounded-lg border ${getSeverityColor(anomaly.severity)}`}>
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center space-x-2">
          {getSeverityIcon(anomaly.severity)}
          <h4 className="text-white font-semibold">{anomaly.type}</h4>
        </div>
        <span className="text-xs text-gray-400">{timestamp}</span>
      </div>
      <p className="text-gray-300 text-sm">{anomaly.description}</p>
      <div className="mt-2">
        <span className={`text-xs px-2 py-1 rounded ${
          anomaly.severity === 'high' ? 'bg-red-800 text-red-200' :
          anomaly.severity === 'medium' ? 'bg-yellow-800 text-yellow-200' :
          'bg-blue-800 text-blue-200'
        }`}>
          {anomaly.severity.toUpperCase()} PRIORITY
        </span>
      </div>
    </div>
  );
};

const AnomalyDetection = ({ anomalies, connectionStatus }) => {
  const mockAnomalies = [
    {
      type: "Revenue Spike",
      severity: "high",
      description: "Q4 revenue increased by 45% above forecast - investigate new market segment performance"
    },
    {
      type: "Deal Size Anomaly",
      severity: "medium", 
      description: "Average deal size in Enterprise segment dropped 15% - review pricing strategy"
    },
    {
      type: "Conversion Drop",
      severity: "high",
      description: "Lead-to-customer conversion rate decreased by 8% in the last 7 days"
    },
    {
      type: "Lead Quality Issue",
      severity: "low",
      description: "Lead scoring model detected unusual pattern in marketing qualified leads"
    }
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-xl font-semibold text-white">Anomaly Detection</h3>
        <div className="flex items-center space-x-2">
          <TrendingUp className="w-5 h-5 text-orange-500" />
          <span className="text-sm text-gray-400">ML-Powered Analysis</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {mockAnomalies.map((anomaly, index) => (
          <AnomalyCard
            key={index}
            anomaly={anomaly}
            timestamp={`${Math.floor(Math.random() * 24)}h ago`}
          />
        ))}
      </div>

      <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
        <h4 className="text-white font-semibold mb-4">Detection Summary</h4>
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <p className="text-2xl font-bold text-red-400">2</p>
            <p className="text-gray-400 text-sm">High Priority</p>
          </div>
          <div>
            <p className="text-2xl font-bold text-yellow-400">1</p>
            <p className="text-gray-400 text-sm">Medium Priority</p>
          </div>
          <div>
            <p className="text-2xl font-bold text-blue-400">1</p>
            <p className="text-gray-400 text-sm">Low Priority</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnomalyDetection;
