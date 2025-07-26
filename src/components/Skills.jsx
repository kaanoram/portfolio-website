import React from 'react';
import { Code, Database, Cloud } from 'lucide-react';
import { skillsData } from '../constants/data';

const SkillCard = ({ icon: Icon, title, skills }) => {
  return (
    <div className="bg-gray-800 p-6 rounded-lg transition-all duration-300 hover:scale-105 border border-gray-700 hover:border-orange-400">
      <div className="flex items-center mb-4">
        <Icon className="w-6 h-6 mr-2 text-orange-400" />
        <h3 className="text-xl font-semibold text-white">{title}</h3>
      </div>
      <div className="flex flex-wrap gap-2">
        {skills.map((skill, index) => (
          <span
            key={index}
            className="px-3 py-1 rounded-full text-sm bg-gray-700 text-orange-400 hover:scale-105 transition-transform duration-200"
          >
            {skill}
          </span>
        ))}
      </div>
    </div>
  );
};

const Skills = () => {
  const icons = [Code, Database, Cloud];

  return (
    <section className="max-w-5xl mx-auto px-4 py-16">
      <h2 className="text-3xl font-bold mb-8 text-orange-400 text-center">
        Technical Skills
      </h2>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        {skillsData.map((skillSet, index) => (
          <SkillCard 
            key={index} 
            icon={icons[index]} 
            {...skillSet} 
          />
        ))}
      </div>
    </section>
  );
};

export default Skills;