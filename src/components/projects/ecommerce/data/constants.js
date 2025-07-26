export const projectDetails = {
  title: "Real-Time E-commerce Analytics",
  description: "Experience a live analytics dashboard processing e-commerce transactions in real-time with ML-powered predictions",
  githubLink: "https://github.com/yourusername/ecommerce-analytics",
  demoLink: "#"
};

export const modelMetrics = {
  accuracy: 79.4,
  precision: 84.8,
  recall: 83.4,
  f1Score: 84.1,
  auc: 87.6,
  dataSize: "16,763 transactions",
  features: 26,
  ensembleSize: 5
};

export const transactionCategories = {
  'High Value Purchase': {
    color: '#10b981',
    description: 'Premium customer likely to purchase',
    minProbability: 0.75
  },
  'Likely Purchase': {
    color: '#3b82f6',
    description: 'Regular customer with good conversion chance',
    minProbability: 0.5
  },
  'Browsing': {
    color: '#f59e0b',
    description: 'Customer browsing with moderate interest',
    minProbability: 0.25
  },
  'Low Intent': {
    color: '#ef4444',
    description: 'Customer unlikely to purchase',
    minProbability: 0
  }
};

export const dashboardConfig = {
  updateInterval: 1000, // milliseconds
  maxTransactions: 5,
  maxChartPoints: 20,
  animationDuration: 300
};

export const customerSegments = [
  {
    id: 0,
    name: 'Premium Shoppers',
    color: '#8b5cf6',
    description: 'High-value customers with frequent purchases'
  },
  {
    id: 1,
    name: 'Regular Buyers',
    color: '#3b82f6',
    description: 'Consistent purchasers with moderate spending'
  },
  {
    id: 2,
    name: 'Occasional Visitors',
    color: '#10b981',
    description: 'Infrequent buyers with potential for growth'
  },
  {
    id: 3,
    name: 'New Customers',
    color: '#f59e0b',
    description: 'Recent acquisitions requiring nurturing'
  }
];