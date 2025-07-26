import React from 'react';
import { ShoppingBag, Clock, TrendingUp, Package, MapPin, Activity } from 'lucide-react';
import { useAnalytics } from '../../../../hooks/useAnalytics';
import { transactionCategories } from '../data/constants';

const Transaction = ({ data }) => {
  const { formatCurrency } = useAnalytics();
  
  // Calculate the time difference
  const getTimeAgo = (timestamp) => {
    const seconds = Math.floor((new Date() - new Date(timestamp * 1000)) / 1000);
    
    if (seconds < 60) return `${seconds}s ago`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    return `${Math.floor(seconds / 3600)}h ago`;
  };

  // Determine transaction category based on probability
  const getTransactionCategory = (probability) => {
    for (const [category, details] of Object.entries(transactionCategories)) {
      if (probability >= details.minProbability) {
        return { category, ...details };
      }
    }
    return { category: 'Low Intent', ...transactionCategories['Low Intent'] };
  };

  const categoryInfo = getTransactionCategory(data.purchase_probability);
  const confidencePercent = (data.confidence * 100).toFixed(0);

  return (
    <div className="bg-gray-700/30 p-4 rounded-lg border border-gray-600 hover:border-orange-400 transition-all duration-300 animate-fadeIn">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div 
            className="p-2 rounded-full"
            style={{ backgroundColor: `${categoryInfo.color}20` }}
          >
            <ShoppingBag className="w-4 h-4" style={{ color: categoryInfo.color }} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <p className="text-white font-medium">{data.customer_id}</p>
              <span 
                className="px-2 py-0.5 rounded-full text-xs font-medium"
                style={{ 
                  backgroundColor: `${categoryInfo.color}20`,
                  color: categoryInfo.color 
                }}
              >
                {categoryInfo.category}
              </span>
            </div>
            <div className="flex items-center gap-3 mt-1">
              <p className="text-sm text-gray-400 flex items-center gap-1">
                <Clock className="w-3 h-3" />
                {data.formatted_time || getTimeAgo(data.timestamp)}
              </p>
              {data.quantity && (
                <p className="text-sm text-gray-400 flex items-center gap-1">
                  <Package className="w-3 h-3" />
                  {data.quantity} items
                </p>
              )}
              {data.country && (
                <p className="text-sm text-gray-400 flex items-center gap-1">
                  <MapPin className="w-3 h-3" />
                  {data.country}
                </p>
              )}
            </div>
          </div>
        </div>
        <div className="text-right">
          <p className="text-white font-semibold text-lg">
            {formatCurrency(data.amount)}
          </p>
          <div className="flex items-center justify-end gap-2 mt-1">
            <div className="flex items-center gap-1">
              <Activity className="w-3 h-3 text-gray-400" />
              <p className="text-sm text-gray-400">
                {(data.purchase_probability * 100).toFixed(0)}%
              </p>
            </div>
            <div 
              className="h-1.5 w-16 bg-gray-700 rounded-full overflow-hidden"
              title={`${confidencePercent}% confidence`}
            >
              <div 
                className="h-full transition-all duration-500"
                style={{ 
                  width: `${data.purchase_probability * 100}%`,
                  backgroundColor: categoryInfo.color 
                }}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const TransactionStream = () => {
  const { transactions, isConnected } = useAnalytics();

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-white">Live Transactions</h3>
        <div className="flex items-center gap-2 text-sm">
          <Clock className="w-4 h-4 text-orange-400" />
          <span className={`text-sm ${
            isConnected ? 'text-green-400' : 'text-yellow-400'
          }`}>
            {isConnected ? 'Live updates' : 'Connecting...'}
          </span>
        </div>
      </div>

      <div className="space-y-3">
        {transactions.map(transaction => (
          <Transaction 
            key={transaction.timestamp} 
            data={transaction} 
          />
        ))}
      </div>

      {transactions.length === 0 && (
        <div className="text-center py-8 text-gray-400 border border-gray-700 rounded-lg">
          <ShoppingBag className="w-8 h-8 mx-auto mb-2 text-gray-600" />
          Waiting for transactions...
        </div>
      )}
    </div>
  );
};

export default TransactionStream;