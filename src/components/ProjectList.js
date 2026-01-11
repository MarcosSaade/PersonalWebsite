import React from 'react';
import FeaturedProjectCard from './FeaturedProjectCard';
import orthopedicsImage from '../images/orthopedics.jpeg';
import solarPanelImage from '../images/solarpanel.png';
import financeImage from '../images/finance.jpeg';
import quantImage from '../images/quant.png';
import dementiaImage from '../images/dementia.jpeg';
import sirImage from '../images/sir.jpeg';
import chessImage from '../images/chess.webp';
import sugarzeroImage from '../images/sugarzero.png';
import neuroCaptureImage from '../images/neurocapture.png';
import './ProjectList.css';

function ProjectList() {
  const projects = [
    {
      title: 'Dementia Detection ML Pipeline',
      description: 'Speech-based ML pipeline using engineered acoustic features and an SVM to detect Alzheimer\'s dementia with a real-time React/Flask web app. Part of an academic paper where I was first author.',
      githubLink: 'https://github.com/MarcosSaade/DementiaDetection',
      readMoreLink: '/dementia',
      image: dementiaImage,
      tags: ['Machine Learning', 'Signal processing', 'Feature Engineering', 'WebApp', 'Scientific Research']
    },
    {
      title: 'S&P 500 Tactical Allocation with ML',
      description: 'Multi-stage ML pipeline forecasting S&P 500 returns + volatility with meta-labeling and regime-aware Kelly sizing for strong risk-adjusted performance. Made for a Kaggle competition.',
      githubLink: 'https://github.com/MarcosSaade/optimal-sp500',
      readMoreLink: '/market-prediction',
      image: quantImage,
      tags: ['Financial ML', 'Time Series', 'Quantitative Finance', 'Feature Engineering', 'Kaggle']
    },
    {
      title: 'SugarZero: Self-Play RL for a Custom Board Game',
      description: 'AlphaZero-style self-play RL agent for the Sugar board game using parallel MCTS and CNNs (PyTorch + Pygame).',
      githubLink: 'https://github.com/MarcosSaade/SugarZero',
      readMoreLink: '/sugarzero',
      image: sugarzeroImage,
      tags: ['PyTorch', 'Reinforcement Learning', 'Deep Learning', 'Game AI', 'Monte Carlo Tree Search', 'Pygame']
    },
    {
      title: 'Orthopedics Image Processing and Visualization App',
      description: 'Desktop app made for an Orthopedics clinic for clinical image acquisition. Features auto background removal, heatmap visualization, automatic PDF reporting, and hardware integration. Used by 4000+ patients yearly.',
      githubLink: 'https://github.com/MarcosSaade/OrthoApp',
      readMoreLink: '/orthopedics',
      image: orthopedicsImage,
      tags: ['Medical Computer Vision', 'OpenCV', 'PyQt5', 'Hardware Integration', 'Freelance', 'Desktop App']
    },
    {
      title: 'NeuroCapture: Multimodal Data Capture App',
      description: 'Fullstack app capturing video (OpenPose), audio, accelerometer, demographic and cognitive test data. Developed for a study conducted by the Center of Microsystems and Biodesign.',
      githubLink: 'https://github.com/MarcosSaade/NeuroCapture',
      readMoreLink: '/neurocapture',
      image: neuroCaptureImage,
      tags: ['Desktop App', 'Data Acquisition', 'SQL', 'Electron', 'Fullstack', 'Research']
    },
    {
      title: 'Financial Education App',
      description: 'Gen AI-powered mobile app with personalized finance lessons, interactive scenarios, and goal tracking. Made for Banorte Hackathon 2024.',
      githubLink: 'https://github.com/MarcosSaade/banorteach',
      readMoreLink: '/banorte',
      image: financeImage,
      tags: ['React Native', 'Gen AI', 'UI/UX Design', 'LLM integration', 'Mobile', 'Hackathon', 'Education']
    },
    {
      title: 'Differential Epidemic Model and Stochastic Simulation',
      description: 'Simulation and analysis of an SIR epidemic model with interactive visualization exploring urban center impact on disease dynamics.',
      githubLink: 'https://github.com/MarcosSaade/SIR-differential-visualizer',
      readMoreLink: '/sir',
      image: sirImage,
      tags: ['Simulation', 'Dynamic Systems', 'Differential Equations', 'Data Visualization', 'Numerical Methods', 'Epidemiology']
    },
    {
      title: 'Solar Panel Cleaning Simulation',
      description: 'MATLAB simulation of electric-field based solar panel cleaning under varying environmental conditions.',
      githubLink: 'https://github.com/MarcosSaade/solar-panel-cleaning',
      readMoreLink: '/solarpanel',
      image: solarPanelImage,
      tags: ['Computational Physics', 'MATLAB Modeling', 'Numerical Analysis', 'Renewable Energy']
    },
    {
      title: 'Chess-Playing AI',
      description: 'Minimax chess engine with alpha-beta pruning, opening book, position memory, and a Pygame interface.',
      githubLink: 'https://github.com/MarcosSaade/chess-engine',
      readMoreLink: '/chess',
      image: chessImage,
      tags: ['Algorithms and Data Structures', 'Pygame', 'AI', 'Alpha-Beta Pruning']
    }
  ];

  // Featured projects
  const featuredProjects = [
    projects.find(p => p.title === 'S&P 500 Tactical Allocation with ML'),
    projects.find(p => p.title === 'Dementia Detection ML Pipeline'),
    projects.find(p => p.title === 'SugarZero: Self-Play RL for a Custom Board Game')
  ].filter(Boolean);

  // Other projects (non-featured)
  const otherProjects = projects.filter(p => 
    !featuredProjects.some(fp => fp.title === p.title)
  );

  return (
    <section id="projects" className="project-list">
      <h2>Projects</h2>
      
      {/* Featured Projects Section */}
      <div className="featured-projects-section">
        <h3 className="featured-title">Featured Projects</h3>
        <div className="featured-grid">
          {featuredProjects.map((project, index) => (
            <FeaturedProjectCard
              key={project.title}
              title={project.title}
              description={project.description}
              githubLink={project.githubLink}
              readMoreLink={project.readMoreLink}
              image={project.image}
              tags={project.tags}
            />
          ))}
        </div>
      </div>

      {/* Other Projects as Links */}
      <div className="other-projects-section">
        <h3 className="other-projects-title">Other Projects</h3>
        <div className="project-links">
          {otherProjects.map((project) => (
            <a 
              key={project.title}
              href={project.readMoreLink} 
              className="project-link"
            >
              {project.title}
            </a>
          ))}
        </div>
      </div>
    </section>
  );
}

export default ProjectList;