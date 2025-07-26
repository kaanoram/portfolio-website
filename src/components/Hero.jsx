import React from 'react';
import { Github, Mail, Linkedin } from 'lucide-react';

const Hero = () => {
  return (
    <header className="bg-gray-900 transition-all duration-300">
      <div className="max-w-5xl mx-auto px-4 py-16">
        <div className="flex flex-col md:flex-row items-center gap-8">
          <div className="w-48 h-48 rounded-full overflow-hidden bg-gray-700 flex-shrink-0 
                          transform hover:scale-105 transition-transform duration-300 shadow-lg hover:shadow-orange-400/20">
            <img 
              src="/profile_pic.jpg"
              alt="Kaan Oram"
              className="w-full h-full object-cover transform hover:scale-105 transition-all duration-500"
            />
          </div>
          <div className="animate-fadeIn">
            <h1 className="text-4xl font-bold text-white mb-4 hover:text-orange-400 transition-colors duration-300">
              Kaan Oram
            </h1>
            <p className="text-xl text-gray-300 mb-6">
              Data Scientist and Machine Learning Engineer specializing in AI systems evaluation 
              and real-time analytics solutions.
            </p>
            <div className="flex space-x-4">
              <a href="mailto:kaanoram@gmail.com" 
                 className="flex items-center text-orange-400 hover:text-orange-300 transform hover:translate-y-[-2px] transition-all duration-300">
                <Mail className="w-5 h-5 mr-2" /> Email
              </a>
              <a href="https://github.com/kaanoram" 
                 className="flex items-center text-orange-400 hover:text-orange-300 transform hover:translate-y-[-2px] transition-all duration-300">
                <Github className="w-5 h-5 mr-2" /> GitHub
              </a>
              <a href="https://linkedin.com/in/kaanoram" 
                 className="flex items-center text-orange-400 hover:text-orange-300 transform hover:translate-y-[-2px] transition-all duration-300">
                <Linkedin className="w-5 h-5 mr-2" /> LinkedIn
              </a>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Hero;