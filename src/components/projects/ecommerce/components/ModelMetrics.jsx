import React from 'react';
import { Brain, Target, TrendingUp, Database, Layers } from 'lucide-react';
import { modelMetrics } from '../data/constants';

const MetricCard = ({ icon: Icon, label, value, color = 'text-orange-400' }) => {
  return (
    <div className="bg-gray-700/50 p-4 rounded-lg border border-gray-600">
      <div className="flex items-center gap-2 mb-2">
        <Icon className={`w-5 h-5 ${color}`} />
        <span className="text-sm text-gray-400">{label}</span>
      </div>
      <p className="text-2xl font-bold text-white">{value}</p>
    </div>
  );
};

const ModelMetrics = () => {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white">Model Performance</h3>
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-orange-400" />
          <span className="text-sm text-gray-400">
            Ensemble of {modelMetrics.ensembleSize} models
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          icon={Target}
          label="Accuracy"
          value={`${modelMetrics.accuracy}%`}
          color="text-green-400"
        />
        <MetricCard
          icon={Brain}
          label="F1 Score"
          value={`${modelMetrics.f1Score}%`}
          color="text-blue-400"
        />
        <MetricCard
          icon={TrendingUp}
          label="AUC"
          value={`${modelMetrics.auc}%`}
          color="text-purple-400"
        />
        <MetricCard
          icon={Database}
          label="Training Data"
          value={modelMetrics.dataSize}
          color="text-yellow-400"
        />
      </div>

      <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
        <h4 className="text-sm font-medium text-gray-300 mb-3">Model Details</h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-400">Architecture:</span>
            <span className="text-white">Wide & Deep Neural Network</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Features:</span>
            <span className="text-white">{modelMetrics.features} engineered features</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Precision:</span>
            <span className="text-white">{modelMetrics.precision}%</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Recall:</span>
            <span className="text-white">{modelMetrics.recall}%</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelMetrics;