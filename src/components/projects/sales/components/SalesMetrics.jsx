import React from 'react';
import { DollarSign, TrendingUp, Users, Target, PieChart, BarChart3 } from 'lucide-react';

const MetricCard = ({ icon: Icon, label, value, change, status }) => (
  <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
    <div className="flex items-center justify-between mb-4">
      <Icon className="w-8 h-8 text-orange-500" />
      <span className={`text-sm px-2 py-1 rounded ${
        status === 'up' ? 'bg-green-900 text-green-300' : 
        status === 'down' ? 'bg-red-900 text-red-300' : 
        'bg-gray-700 text-gray-300'
      }`}>
        {change}
      </span>
    </div>
    <h3 className="text-2xl font-bold text-white mb-1">{value}</h3>
    <p className="text-gray-400 text-sm">{label}</p>
  </div>
);

const SalesMetrics = ({ metrics, connectionStatus, error }) => {
  if (error) {
    return (
      <div className="bg-red-900/20 border border-red-500 rounded-lg p-4">
        <p className="text-red-400">Unable to connect to the sales analytics server after multiple attempts</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-xl font-semibold text-white">Sales Performance</h3>
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
          icon={DollarSign}
          label="Total Revenue"
          value={metrics?.totalRevenue || "$2.4M"}
          change="+12.5%"
          status="up"
        />
        <MetricCard
          icon={TrendingUp}
          label="Sales Growth"
          value={metrics?.salesGrowth || "18.3%"}
          change="+3.2%"
          status="up"
        />
        <MetricCard
          icon={Target}
          label="Avg Deal Size"
          value={metrics?.avgDealSize || "$45.2K"}
          change="+8.1%"
          status="up"
        />
        <MetricCard
          icon={BarChart3}
          label="Conversion Rate"
          value={metrics?.conversionRate || "24.7%"}
          change="-1.2%"
          status="down"
        />
        <MetricCard
          icon={Users}
          label="Active Leads"
          value={metrics?.activeLeads || "1,247"}
          change="+156"
          status="up"
        />
        <MetricCard
          icon={PieChart}
          label="Pipeline Value"
          value={metrics?.pipelineValue || "$8.9M"}
          change="+22.4%"
          status="up"
        />
      </div>
    </div>
  );
};

export default SalesMetrics;
