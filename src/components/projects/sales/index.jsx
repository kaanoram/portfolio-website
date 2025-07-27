import React from 'react';
import ProjectContainer from '../shared/ProjectContainer';
import { projectDetails } from './data/constants';
import SalesMetrics from './components/SalesMetrics';
import DataProcessingPanel from './components/DataProcessingPanel';
import AnomalyDetection from './components/AnomalyDetection';
import { useSalesAnalytics } from '../../../hooks/useSalesAnalytics';

const SalesAnalytics = () => {
  const { metrics, processingStats, anomalies, connectionStatus, error } = useSalesAnalytics();

  return (
    <ProjectContainer 
      title={projectDetails.title}
      description={projectDetails.description}
      githubLink={projectDetails.githubLink}
      demoLink={projectDetails.demoLink}
    >
      <div className="space-y-8">
        <SalesMetrics 
          metrics={metrics}
          connectionStatus={connectionStatus}
          error={error}
        />
        
        <DataProcessingPanel 
          processingStats={processingStats}
          connectionStatus={connectionStatus}
        />
        
        <AnomalyDetection 
          anomalies={anomalies}
          connectionStatus={connectionStatus}
        />
      </div>
    </ProjectContainer>
  );
};

export default SalesAnalytics;
