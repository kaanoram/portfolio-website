import React from 'react';
import ProjectContainer from '../shared/ProjectContainer';
import Dashboard from './components/Dashboard';
import MetricsPanel from './components/MetricsPanel';
import TransactionStream from './components/TransactionStream';
import ModelMetrics from './components/ModelMetrics';
import { projectDetails } from './data/constants';

const EcommerceAnalytics = () => {
  return (
    <ProjectContainer
      title={projectDetails.title}
      description={projectDetails.description}
      githubLink={projectDetails.githubLink}
      demoLink={projectDetails.demoLink}
    >
      <div className="space-y-6">
        {/* Real-time metrics panel */}
        <MetricsPanel />
        
        {/* Model performance metrics */}
        <ModelMetrics />
        
        {/* Main dashboard with charts */}
        <Dashboard />
        
        {/* Real-time transaction stream */}
        <TransactionStream />
      </div>
    </ProjectContainer>
  );
};

export default EcommerceAnalytics;