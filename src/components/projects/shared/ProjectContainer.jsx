import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft, Github, ExternalLink } from 'lucide-react';

const ProjectContainer = ({ 
  title,
  description,
  children,
  githubLink,
  demoLink 
}) => {
  return (
    <div className="min-h-screen bg-gray-900 py-8">
      <div className="w-full max-w-7xl mx-auto px-4 space-y-6">
        {/* Navigation */}
        <Link 
          to="/" 
          className="inline-flex items-center gap-2 text-gray-400 hover:text-orange-400 transition-colors mb-6"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Portfolio
        </Link>

        {/* Project Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-orange-400 mb-3">{title}</h1>
          <p className="text-lg text-gray-300">{description}</p>
        </div>

        {/* Project Content */}
        <div className="space-y-6">
          {children}
        </div>

        {/* Footer Actions */}
        <div className="flex gap-4 mt-12 pt-6 border-t border-gray-700">
          {githubLink && (
            <a 
              href={githubLink}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 text-orange-400 rounded-lg transition-colors"
            >
              <Github className="w-5 h-5" />
              View Source Code
            </a>
          )}
          <Link 
            to="/"
            className="px-4 py-2 bg-orange-400 hover:bg-orange-500 text-gray-900 font-medium rounded-lg transition-colors"
          >
            Back to Portfolio
          </Link>
        </div>
      </div>
    </div>
  );
};

export default ProjectContainer;