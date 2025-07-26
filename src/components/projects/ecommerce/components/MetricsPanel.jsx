import React from 'react';
import { TrendingUp, Users, ShoppingCart, DollarSign } from 'lucide-react';
import { useAnalytics } from '../../../../hooks/useAnalytics';

const MetricCard = ({ icon: Icon, title, value, trend, prefix = '', suffix = '' }) => {
  return (
    <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-600">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Icon className="w-5 h-5 text-orange-400" />
          <h4 className="text-sm font-medium text-gray-300">{title}</h4>
        </div>
        {trend && (
          <span className={`text-xs ${trend > 0 ? 'text-green-400' : 'text-red-400'}`}>
            {trend > 0 ? '+' : ''}{trend}%
          </span>
        )}
      </div>
      <div className="text-white">
        <span className="text-2xl font-bold">
          {prefix}{typeof value === 'number' ? value.toLocaleString() : value}{suffix}
        </span>
      </div>
    </div>
  );
};

const MetricsPanel = () => {
  const { metrics, isConnected, error } = useAnalytics();

  if (error) {
    return (
      <div className="p-4 bg-red-500/10 border border-red-500 rounded-lg">
        <p className="text-red-500">{error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white">Real-time Metrics</h3>
        <span className={`px-2 py-1 rounded-full text-xs ${
          isConnected ? 'bg-green-500/20 text-green-400' : 'bg-yellow-500/20 text-yellow-400'
        }`}>
          {isConnected ? 'Connected' : 'Connecting...'}
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          icon={DollarSign}
          title="Revenue"
          value={metrics.revenue}
          trend={2.5}
          prefix="$"
        />
        <MetricCard
          icon={Users}
          title="Active Customers"
          value={metrics.customers}
          trend={1.8}
        />
        <MetricCard
          icon={ShoppingCart}
          title="Orders"
          value={metrics.orders}
          trend={3.2}
        />
        <MetricCard
          icon={TrendingUp}
          title="Conversion Rate"
          value={metrics.conversionRate.toFixed(1)}
          trend={-0.5}
          suffix="%"
        />
      </div>
    </div>
  );
};

export default MetricsPanel;