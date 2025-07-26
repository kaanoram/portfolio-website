export const skillsData = [
  {
    title: "Machine Learning & AI",
    description: "Expertise in building and deploying ML models with focus on real-time processing and AI system evaluation",
    skills: [
      "TensorFlow",
      "PyTorch",
      "scikit-learn",
      "Neural Networks",
      "NLP",
      "ML Pipelines",
      "Model Optimization"
    ]
  },
  {
    title: "Data Engineering",
    description: "Experience with big data processing and real-time analytics pipelines",
    skills: [
      "Apache Spark",
      "Hadoop",
      "MongoDB",
      "PostgreSQL",
      "ETL Pipeline",
      "Data Warehousing",
      "Stream Processing"
    ]
  },
  {
    title: "Cloud & DevOps",
    description: "Proficient in cloud platforms and containerization for scalable deployments",
    skills: [
      "AWS Lambda",
      "AWS ECS",
      "AWS S3",
      "Docker",
      "Kubernetes",
      "CI/CD",
      "Microservices"
    ]
  }
];

export const projectsData = [
  {
    id: 1,
    title: "Real-Time E-commerce Analytics Platform",
    shortDescription: "ML pipeline processing millions of transactions with sub-second latency",
    fullDescription: `A highly scalable analytics platform that processes over 1 million daily transactions 
    in real-time, providing instant insights for e-commerce businesses. The system uses advanced machine 
    learning techniques for customer segmentation and purchase prediction, deployed on AWS with auto-scaling 
    capabilities.`,
    challenges: [
      "Processing high-volume transaction data with sub-second latency requirements",
      "Implementing efficient customer segmentation across multiple behavioral dimensions",
      "Deploying and scaling ML models in production environment",
      "Managing concurrent user access while maintaining performance"
    ],
    techDetails: {
      architecture: "Microservices architecture with event-driven processing",
      mlComponents: "Neural networks for prediction, K-means clustering for segmentation",
      dataFlow: "Real-time stream processing with Apache Kafka and AWS Kinesis",
      deployment: "Containerized deployment on AWS ECS with auto-scaling"
    },
    techStack: ["Python", "TensorFlow", "AWS ECS", "AWS RDS", "AWS S3", "React"],
    metrics: [
      "1M+ daily transactions processed",
      "Sub-second latency achieved",
      "88% prediction accuracy",
      "2000+ concurrent users supported",
      "40% improvement over baseline"
    ],
    githubLink: "https://github.com/yourusername/ecommerce-analytics",
    demoLink: "/projects/ecommerce-analytics",
    images: [
      {
        url: "/api/placeholder/800/400",
        alt: "Analytics Dashboard",
        caption: "Real-time analytics dashboard showing key metrics"
      },
      {
        url: "/api/placeholder/800/400",
        alt: "Architecture Diagram",
        caption: "System architecture overview"
      }
    ]
  },
  {
    id: 2,
    title: "Enterprise Sales Analytics System",
    shortDescription: "Cloud-based platform processing terabytes of sales data with real-time insights",
    fullDescription: `A comprehensive analytics solution that transforms raw sales data into actionable 
    insights for enterprise decision-makers. Features include automated anomaly detection, real-time ETL 
    pipelines, and interactive executive dashboards.`,
    challenges: [
      "Processing and analyzing 10TB+ of historical sales data",
      "Maintaining data accuracy in real-time ETL processes",
      "Reducing false positives in anomaly detection",
      "Creating intuitive visualizations for complex data"
    ],
    techDetails: {
      architecture: "Serverless architecture with AWS Lambda",
      dataProcessing: "Real-time ETL with AWS Glue and Redshift",
      monitoring: "Automated monitoring with CloudWatch",
      visualization: "Interactive dashboards using Tableau"
    },
    techStack: ["Python", "AWS Redshift", "AWS Lambda", "AWS CloudWatch", "Tableau"],
    metrics: [
      "10TB+ data processed",
      "99.99% data accuracy",
      "60% reduction in false alerts",
      "50+ KPIs tracked in real-time"
    ],
    githubLink: "#",
    demoLink: "#",
    images: [
      {
        url: "/api/placeholder/800/400",
        alt: "Sales Dashboard",
        caption: "Executive dashboard showing sales analytics"
      }
    ]
  },
  {
    id: 3,
    title: "Real-Time Fraud Detection Engine",
    shortDescription: "High-performance ML system for instant fraud detection",
    fullDescription: `A sophisticated fraud detection system that leverages machine learning to analyze 
    transactions in real-time. The system employs custom feature engineering and automated model retraining 
    to maintain high accuracy while processing thousands of transactions per second.`,
    challenges: [
      "Achieving high accuracy while maintaining low latency",
      "Handling high-velocity transaction streams",
      "Reducing model training time",
      "Maintaining model performance over time"
    ],
    techDetails: {
      architecture: "Stream processing with AWS Kinesis",
      mlPipeline: "Custom feature engineering pipeline",
      deployment: "Distributed processing with MongoDB sharding",
      monitoring: "Real-time performance monitoring and alerting"
    },
    techStack: ["Python", "TensorFlow", "AWS Lambda", "AWS Kinesis", "MongoDB"],
    metrics: [
      "95% detection accuracy",
      "0.1% false positive rate",
      "1000+ transactions/second",
      "50ms average latency",
      "40% faster training time"
    ],
    githubLink: "#",
    demoLink: "#",
    images: [
      {
        url: "/api/placeholder/800/400",
        alt: "Fraud Detection Dashboard",
        caption: "Real-time fraud detection monitoring"
      }
    ]
  }
];