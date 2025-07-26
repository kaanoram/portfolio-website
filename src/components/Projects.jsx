import React, { useState } from 'react';
import { Github, ExternalLink, ChevronDown, ChevronUp, Code, Database, Server } from 'lucide-react';
import { projectsData } from '../constants/data';
import { Link } from 'react-router-dom';

// MetricBadge subcomponent for consistent styling of metrics
const MetricBadge = ({ children }) => (
  <div className="bg-gray-700 px-3 py-1 rounded-full text-sm text-orange-400 hover:scale-105 transition-transform duration-200">
    {children}
  </div>
);

// ProjectCard component to display individual project information
const ProjectCard = ({ project, isExpanded, onToggle }) => {
  return (
    <div className="bg-gray-800 rounded-lg p-6 transition-all duration-300 hover:scale-105 border border-gray-700 hover:border-orange-400">
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-2xl font-bold text-white mb-2">{project.title}</h3>
          <p className="text-gray-300">{project.shortDescription}</p>
        </div>
        <button 
          onClick={onToggle}
          className="text-orange-400 hover:text-orange-300 transition-colors p-2 rounded-full hover:bg-gray-700/50"
          aria-label={isExpanded ? "Show less" : "Show more"}
        >
          {isExpanded ? <ChevronUp size={24} /> : <ChevronDown size={24} />}
        </button>
      </div>

      <div className="flex flex-wrap gap-2 mb-4">
        {project.techStack.map((tech, index) => (
          <span 
            key={index}
            className="px-3 py-1 rounded-full text-sm bg-gray-700 text-orange-400"
          >
            {tech}
          </span>
        ))}
      </div>

      <div className={`transition-all duration-500 overflow-hidden ${
        isExpanded ? 'max-h-[2000px] opacity-100' : 'max-h-0 opacity-0'
      }`}>
        <div className="space-y-6 pt-4">
          {/* Project Description */}
          <div>
            <h4 className="text-lg font-semibold text-white mb-2">Overview</h4>
            <p className="text-gray-300">{project.fullDescription}</p>
          </div>

          {/* Technical Details */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-white mb-2">
                <Code size={20} className="text-orange-400" />
                <h4 className="text-lg font-semibold">Architecture</h4>
              </div>
              <p className="text-gray-300">{project.techDetails.architecture}</p>
              <p className="text-gray-300">{project.techDetails.mlComponents}</p>
            </div>
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-white mb-2">
                <Server size={20} className="text-orange-400" />
                <h4 className="text-lg font-semibold">Implementation</h4>
              </div>
              <p className="text-gray-300">{project.techDetails.dataFlow}</p>
              <p className="text-gray-300">{project.techDetails.deployment}</p>
            </div>
          </div>

          {/* Key Metrics */}
          <div>
            <h4 className="text-lg font-semibold text-white mb-3">Key Metrics</h4>
            <div className="flex flex-wrap gap-2">
              {project.metrics.map((metric, index) => (
                <MetricBadge key={index}>{metric}</MetricBadge>
              ))}
            </div>
          </div>

          {/* Technical Challenges */}
          <div>
            <h4 className="text-lg font-semibold text-white mb-3">Technical Challenges</h4>
            <ul className="space-y-2">
              {project.challenges.map((challenge, index) => (
                <li key={index} className="flex items-start text-gray-300">
                  <span className="text-orange-400 mr-2">•</span>
                  {challenge}
                </li>
              ))}
            </ul>
          </div>

          {/* Project Images */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {project.images.map((image, index) => (
              <div key={index} className="space-y-2">
                <img 
                  src={image.url}
                  alt={image.alt}
                  className="rounded-lg w-full h-48 object-cover"
                />
                <p className="text-sm text-gray-400 text-center">{image.caption}</p>
              </div>
            ))}
          </div>

          {/* Links */}
          <div className="flex gap-4 pt-4">
            {project.githubLink && (
              <a 
                href={project.githubLink}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 text-orange-400 hover:text-orange-300 transform hover:translate-y-[-2px] transition-all duration-300"
              >
                <Github size={20} />
                <span>View Code</span>
              </a>
            )}
            {project.demoLink && (
              project.demoLink.startsWith('/') ? (
                <Link 
                  to={project.demoLink}
                  className="flex items-center gap-2 text-orange-400 hover:text-orange-300 transform hover:translate-y-[-2px] transition-all duration-300"
                >
                  <ExternalLink size={20} />
                  <span>Live Demo</span>
                </Link>
              ) : (
                <a 
                  href={project.demoLink}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-2 text-orange-400 hover:text-orange-300 transform hover:translate-y-[-2px] transition-all duration-300"
                >
                  <ExternalLink size={20} />
                  <span>Live Demo</span>
                </a>
              )
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

// Main Projects component that manages the project cards and their state
const Projects = () => {
  const [expandedIds, setExpandedIds] = useState(new Set());

  const toggleExpansion = (projectId) => {
    setExpandedIds(prevIds => {
      const newIds = new Set(prevIds);
      if (newIds.has(projectId)) {
        newIds.delete(projectId);
      } else {
        newIds.add(projectId);
      }
      return newIds;
    });
  };

  return (
    <section className="max-w-5xl mx-auto px-4 py-16">
      <h2 className="text-3xl font-bold mb-8 text-orange-400 text-center">
        Featured Projects
      </h2>
      <div className="grid grid-cols-1 gap-8">
        {projectsData.map((project) => (
          <ProjectCard
            key={project.id}
            project={project}
            isExpanded={expandedIds.has(project.id)}
            onToggle={() => toggleExpansion(project.id)}
          />
        ))}
      </div>
    </section>
  );
};

export default Projects;