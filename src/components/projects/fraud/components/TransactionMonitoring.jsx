import React from 'react';
import { CreditCard, MapPin, Clock, AlertCircle } from 'lucide-react';

const TransactionCard = ({ transaction }) => {
  const getRiskColor = (score) => {
    if (score >= 90) return 'text-red-400 bg-red-900/20 border-red-500';
    if (score >= 70) return 'text-orange-400 bg-orange-900/20 border-orange-500';
    if (score >= 30) return 'text-yellow-400 bg-yellow-900/20 border-yellow-500';
    return 'text-green-400 bg-green-900/20 border-green-500';
  };

  const getRiskLabel = (score) => {
    if (score >= 90) return 'CRITICAL';
    if (score >= 70) return 'HIGH';
    if (score >= 30) return 'MEDIUM';
    return 'LOW';
  };

  return (
    <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <CreditCard className="w-4 h-4 text-orange-500" />
          <span className="text-white font-mono text-sm">{transaction.id}</span>
        </div>
        <div className={`px-2 py-1 rounded text-xs border ${getRiskColor(transaction.riskScore)}`}>
          {getRiskLabel(transaction.riskScore)} ({transaction.riskScore})
        </div>
      </div>
      
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <p className="text-gray-400">Amount</p>
          <p className="text-white font-semibold">${transaction.amount}</p>
        </div>
        <div>
          <p className="text-gray-400">Merchant</p>
          <p className="text-white">{transaction.merchant}</p>
        </div>
        <div className="flex items-center space-x-1">
          <MapPin className="w-3 h-3 text-gray-400" />
          <span className="text-gray-300">{transaction.location}</span>
        </div>
        <div className="flex items-center space-x-1">
          <Clock className="w-3 h-3 text-gray-400" />
          <span className="text-gray-300">{transaction.timestamp}</span>
        </div>
      </div>

      {transaction.flags && transaction.flags.length > 0 && (
        <div className="mt-3 pt-3 border-t border-gray-700">
          <div className="flex items-center space-x-1 mb-2">
            <AlertCircle className="w-3 h-3 text-yellow-500" />
            <span className="text-yellow-400 text-xs">Risk Factors</span>
          </div>
          <div className="flex flex-wrap gap-1">
            {transaction.flags.map((flag, index) => (
              <span key={index} className="text-xs px-2 py-1 bg-yellow-900/30 text-yellow-300 rounded">
                {flag}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

const TransactionMonitoring = ({ transactions, connectionStatus }) => {
  const mockTransactions = [
    {
      id: "TXN-847291",
      amount: "2,450.00",
      merchant: "Electronics Store",
      location: "New York, NY",
      timestamp: "2s ago",
      riskScore: 85,
      flags: ["High Amount", "New Location"]
    },
    {
      id: "TXN-847290",
      amount: "45.99",
      merchant: "Coffee Shop",
      location: "San Francisco, CA",
      timestamp: "5s ago",
      riskScore: 15,
      flags: []
    },
    {
      id: "TXN-847289",
      amount: "1,200.00",
      merchant: "Online Retailer",
      location: "Miami, FL",
      timestamp: "8s ago",
      riskScore: 92,
      flags: ["Velocity", "Device Mismatch", "Unusual Time"]
    },
    {
      id: "TXN-847288",
      amount: "89.50",
      merchant: "Gas Station",
      location: "Austin, TX",
      timestamp: "12s ago",
      riskScore: 25,
      flags: []
    },
    {
      id: "TXN-847287",
      amount: "3,750.00",
      merchant: "Jewelry Store",
      location: "Las Vegas, NV",
      timestamp: "15s ago",
      riskScore: 78,
      flags: ["High Amount", "Merchant Category"]
    },
    {
      id: "TXN-847286",
      amount: "25.00",
      merchant: "Fast Food",
      location: "Chicago, IL",
      timestamp: "18s ago",
      riskScore: 8,
      flags: []
    }
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-xl font-semibold text-white">Live Transaction Monitoring</h3>
        <div className="flex items-center space-x-2">
          <div className={`w-2 h-2 rounded-full ${
            connectionStatus === 'connected' ? 'bg-green-500 animate-pulse' : 'bg-red-500'
          }`}></div>
          <span className="text-sm text-gray-400">Real-time Stream</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {mockTransactions.map((transaction, index) => (
          <TransactionCard key={index} transaction={transaction} />
        ))}
      </div>

      <div className="bg-gray-800 p-4 rounded-lg border border-gray-700">
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-400">Processing Rate:</span>
          <span className="text-white">~2,400 TPS</span>
        </div>
      </div>
    </div>
  );
};

export default TransactionMonitoring;
