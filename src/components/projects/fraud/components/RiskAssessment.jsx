import React from 'react';
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import { Brain, Zap, Eye, Shield } from 'lucide-react';

const ModelCard = ({ icon: Icon, title, accuracy, latency, status }) => (
  <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
    <div className="flex items-center space-x-3 mb-3">
      <Icon className="w-6 h-6 text-orange-500" />
      <h4 className="text-white font-semibold">{title}</h4>
    </div>
    <div className="space-y-2 text-sm">
      <div className="flex justify-between">
        <span className="text-gray-400">Accuracy:</span>
        <span className="text-white">{accuracy}</span>
      </div>
      <div className="flex justify-between">
        <span className="text-gray-400">Latency:</span>
        <span className="text-white">{latency}</span>
      </div>
      <div className="flex justify-between">
        <span className="text-gray-400">Status:</span>
        <span className={`${status === 'Active' ? 'text-green-400' : 'text-yellow-400'}`}>
          {status}
        </span>
      </div>
    </div>
  </div>
);

const RiskAssessment = ({ riskScores, connectionStatus }) => {
  const riskDistribution = [
    { name: 'Low Risk', value: 78, color: '#10B981' },
    { name: 'Medium Risk', value: 15, color: '#F59E0B' },
    { name: 'High Risk', value: 5, color: '#F97316' },
    { name: 'Critical Risk', value: 2, color: '#EF4444' }
  ];

  const hourlyData = [
    { hour: '00:00', transactions: 1200, fraud: 15 },
    { hour: '04:00', transactions: 800, fraud: 8 },
    { hour: '08:00', transactions: 2400, fraud: 28 },
    { hour: '12:00', transactions: 3200, fraud: 45 },
    { hour: '16:00', transactions: 2800, fraud: 38 },
    { hour: '20:00', transactions: 2100, fraud: 25 }
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-xl font-semibold text-white">Risk Assessment Models</h3>
        <div className="flex items-center space-x-2">
          <Brain className="w-5 h-5 text-orange-500" />
          <span className="text-sm text-gray-400">ML Ensemble</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <ModelCard
          icon={Zap}
          title="Velocity Model"
          accuracy="94.2%"
          latency="3ms"
          status="Active"
        />
        <ModelCard
          icon={Eye}
          title="Behavioral Model"
          accuracy="91.8%"
          latency="8ms"
          status="Active"
        />
        <ModelCard
          icon={Shield}
          title="Device Model"
          accuracy="89.5%"
          latency="5ms"
          status="Active"
        />
        <ModelCard
          icon={Brain}
          title="Ensemble Model"
          accuracy="96.8%"
          latency="12ms"
          status="Active"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
          <h4 className="text-white font-semibold mb-4">Risk Score Distribution</h4>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={riskDistribution}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  dataKey="value"
                  label={({ name, value }) => `${name}: ${value}%`}
                >
                  {riskDistribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
          <h4 className="text-white font-semibold mb-4">Fraud Detection Trends</h4>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={hourlyData}>
                <XAxis dataKey="hour" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Bar dataKey="fraud" fill="#F97316" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
        <h4 className="text-white font-semibold mb-4">Model Performance Summary</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
          <div>
            <p className="text-2xl font-bold text-green-400">96.8%</p>
            <p className="text-gray-400 text-sm">Overall Accuracy</p>
          </div>
          <div>
            <p className="text-2xl font-bold text-blue-400">2.1%</p>
            <p className="text-gray-400 text-sm">False Positive Rate</p>
          </div>
          <div>
            <p className="text-2xl font-bold text-orange-400">12ms</p>
            <p className="text-gray-400 text-sm">Avg Response Time</p>
          </div>
          <div>
            <p className="text-2xl font-bold text-purple-400">99.9%</p>
            <p className="text-gray-400 text-sm">System Uptime</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RiskAssessment;
