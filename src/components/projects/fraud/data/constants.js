export const projectDetails = {
  title: "Real-Time Fraud Detection Engine",
  description: "Experience a high-performance fraud detection system processing transactions with ML-powered risk scoring and real-time alerts",
  githubLink: "https://github.com/kaanoram/portfolio-website",
  demoLink: "#"
};

export const fraudMetrics = {
  transactionsProcessed: { label: "Transactions Processed", format: "number" },
  fraudDetected: { label: "Fraud Detected", format: "number" },
  accuracy: { label: "Detection Accuracy", format: "percentage" },
  falsePositiveRate: { label: "False Positive Rate", format: "percentage" },
  avgProcessingTime: { label: "Avg Processing Time", format: "time" },
  riskScore: { label: "Avg Risk Score", format: "score" }
};

export const riskCategories = [
  { category: "Low Risk", range: "0-30", color: "green", count: 0 },
  { category: "Medium Risk", range: "31-70", color: "yellow", count: 0 },
  { category: "High Risk", range: "71-90", color: "orange", count: 0 },
  { category: "Critical Risk", range: "91-100", color: "red", count: 0 }
];

export const fraudPatterns = [
  { pattern: "Velocity Check", description: "Multiple transactions in short time" },
  { pattern: "Geolocation", description: "Unusual location patterns" },
  { pattern: "Amount Anomaly", description: "Transaction amount outside normal range" },
  { pattern: "Device Fingerprint", description: "Suspicious device characteristics" },
  { pattern: "Behavioral Analysis", description: "Unusual user behavior patterns" }
];
