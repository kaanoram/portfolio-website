import React from 'react';
import { Database, Clock, CheckCircle, AlertTriangle } from 'lucide-react';

const ProcessingMetric = ({ icon: Icon, label, value, status }) => (
  <div className="flex items-center space-x-3 p-4 bg-gray-800 rounded-lg border border-gray-700">
    <Icon className={`w-6 h-6 ${
      status === 'good' ? 'text-green-500' : 
      status === 'warning' ? 'text-yellow-500' : 'text-red-500'
    }`} />
    <div>
      <p className="text-white font-semibold">{value}</p>
      <p className="text-gray-400 text-sm">{label}</p>
    </div>
  </div>
);

const DataProcessingPanel = ({ processingStats, connectionStatus }) => {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-xl font-semibold text-white">ETL Pipeline Status</h3>
        <div className="flex items-center space-x-2">
          <Database className="w-5 h-5 text-orange-500" />
          <span className="text-sm text-gray-400">AWS Redshift</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <ProcessingMetric
          icon={Database}
          label="Records Processed"
          value={processingStats?.recordsProcessed || "2.1M"}
          status="good"
        />
        <ProcessingMetric
          icon={CheckCircle}
          label="Data Accuracy"
          value={processingStats?.dataAccuracy || "99.7%"}
          status="good"
        />
        <ProcessingMetric
          icon={Clock}
          label="ETL Latency"
          value={processingStats?.etlLatency || "2.3s"}
          status="good"
        />
        <ProcessingMetric
          icon={AlertTriangle}
          label="Error Rate"
          value={processingStats?.errorRate || "0.03%"}
          status="good"
        />
      </div>

      <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
        <h4 className="text-white font-semibold mb-4">Pipeline Architecture</h4>
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
            <span className="text-gray-300">Salesforce CRM</span>
          </div>
          <div className="text-gray-500">→</div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-green-500 rounded-full"></div>
            <span className="text-gray-300">AWS Lambda</span>
          </div>
          <div className="text-gray-500">→</div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-orange-500 rounded-full"></div>
            <span className="text-gray-300">Redshift</span>
          </div>
          <div className="text-gray-500">→</div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
            <span className="text-gray-300">Tableau</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DataProcessingPanel;
