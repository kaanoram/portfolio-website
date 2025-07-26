import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Hero from './components/Hero';
import Skills from './components/Skills';
import Projects from './components/Projects';
import EcommerceAnalytics from './components/projects/ecommerce';
import SalesAnalytics from './components/projects/sales';
import FraudDetection from './components/projects/fraud';

const HomePage = () => {
  return (
    <>
      <Hero />
      <Skills />
      <Projects />
    </>
  );
};

const App = () => {
  return (
    <Router>
      <div className="min-h-screen bg-gray-900">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/projects/ecommerce-analytics" element={<EcommerceAnalytics />} />
          <Route path="/projects/sales-analytics" element={<SalesAnalytics />} />
          <Route path="/projects/fraud-detection" element={<FraudDetection />} />
        </Routes>
      </div>
    </Router>
  );
};

export default App;
