export const projectDetails = {
  title: "Enterprise Sales Analytics System",
  description: "Experience a comprehensive ETL pipeline processing enterprise sales data with advanced anomaly detection and executive dashboards",
  githubLink: "https://github.com/kaanoram/portfolio-website",
  demoLink: "#"
};

export const salesMetrics = {
  totalRevenue: { label: "Total Revenue", format: "currency" },
  salesGrowth: { label: "Sales Growth", format: "percentage" },
  avgDealSize: { label: "Avg Deal Size", format: "currency" },
  conversionRate: { label: "Conversion Rate", format: "percentage" },
  activeLeads: { label: "Active Leads", format: "number" },
  pipelineValue: { label: "Pipeline Value", format: "currency" }
};

export const processingMetrics = {
  recordsProcessed: { label: "Records Processed", format: "number" },
  dataAccuracy: { label: "Data Accuracy", format: "percentage" },
  etlLatency: { label: "ETL Latency", format: "time" },
  errorRate: { label: "Error Rate", format: "percentage" }
};

export const anomalyTypes = [
  { type: "Revenue Spike", severity: "high", description: "Unusual revenue increase detected" },
  { type: "Deal Size Anomaly", severity: "medium", description: "Deal size outside normal range" },
  { type: "Conversion Drop", severity: "high", description: "Significant conversion rate decrease" },
  { type: "Lead Quality Issue", severity: "low", description: "Lead scoring anomaly detected" }
];
