import React from 'react';
import ProjectContainer from '../shared/ProjectContainer';
import { projectDetails } from './data/constants';
import FraudMetrics from './components/FraudMetrics';
import TransactionMonitoring from './components/TransactionMonitoring';
import RiskAssessment from './components/RiskAssessment';
import { useFraudAnalytics } from '../../../hooks/useFraudAnalytics';

const FraudDetection = () => {
  const { metrics, transactions, riskScores, connectionStatus, error } = useFraudAnalytics();

  return (
    <ProjectContainer 
      title={projectDetails.title}
      description={projectDetails.description}
      githubLink={projectDetails.githubLink}
      demoLink={projectDetails.demoLink}
    >
      <div className="space-y-8">
        <FraudMetrics 
          metrics={metrics}
          connectionStatus={connectionStatus}
          error={error}
        />
        
        <TransactionMonitoring 
          transactions={transactions}
          connectionStatus={connectionStatus}
        />
        
        <RiskAssessment 
          riskScores={riskScores}
          connectionStatus={connectionStatus}
        />
      </div>
    </ProjectContainer>
  );
};

export default FraudDetection;
