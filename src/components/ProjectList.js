// src/components/ProjectList.js
import React from 'react';
import { useState } from 'react';
import ProjectCard from './ProjectCard';
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
      articleLink: null,
      readMoreLink: '/dementia',
      image: dementiaImage,
      tags: ['Machine Learning', 'Data Preprocessing', 'Feature Engineering', 'Full-Stack Development', 'Scientific Research'],
      category: 'Machine Learning & AI'
    },
    {
      title: 'S&P 500 Tactical Allocation with ML',
      description: 'Multi-stage ML pipeline forecasting S&P 500 returns + volatility with meta-labeling and regime-aware Kelly sizing for strong risk-adjusted performance. Made for a Kaggle competition.',
      githubLink: 'https://github.com/MarcosSaade/optimal-sp500',
      readMoreLink: '/market-prediction',
      image: quantImage,
      tags: ['Financial ML', 'Time Series', 'Quantitative Finance', 'Feature Engineering', 'Risk Management', 'Kaggle'],
      category: 'Machine Learning & AI'
    },
    {
      title: 'Orthopedics Image Processing and Visualization App',
      description: 'Desktop app made for an Orthopedics clinic for clinical image acquisition. Features auto background removal, heatmap visualization, automatic PDF reporting, and hardware integration. Used by 4000+ patients yearly.',
      githubLink: 'https://github.com/MarcosSaade/OrthoApp',
      readMoreLink: '/orthopedics',
      image: orthopedicsImage,
      tags: ['Medical Computer Vision', 'OpenCV', 'PyQt5', 'Hardware Integration', 'Freelance', 'Desktop App'],
      category: 'App Development'
    },
    {
      title: 'SugarZero: Self-Play RL for a Custom Board Game',
      description: 'AlphaZero-style self-play RL agent for the Sugar board game using parallel MCTS and CNNs (PyTorch + Pygame).',
      githubLink: 'https://github.com/MarcosSaade/SugarZero',
      readMoreLink: '/sugarzero',
      image: sugarzeroImage,
      tags: [
        'PyTorch',
        'Reinforcement Learning',
        'Deep Learning',
        'Game AI',
        'Monte Carlo Tree Search',
        'Pygame',
      ],
      category: 'Machine Learning & AI'
    },
    {
      title: 'NeuroCapture: Multimodal Data Capture App',
      description: 'Fullstack app capturing video (OpenPose), audio, accelerometer, demographic and cognitive test data. Developed for a study conducted by the Center of Microsystems and Biodesign.',
      githubLink: 'https://github.com/MarcosSaade/NeuroCapture',
      readMoreLink: '/neurocapture',
      image: neuroCaptureImage,
      tags: ['Desktop App', 'Data Acquisition', 'SQL', 'Electron', 'Fullstack', 'Research'],
      category: 'App Development'
    },
    {
      title: 'Financial Education App',
      description: 'Gen AI-powered mobile app with personalized finance lessons, interactive scenarios, and goal tracking. Made for Banorte Hackathon 2024.',
      githubLink: 'https://github.com/MarcosSaade/banorteach',
      readMoreLink: '/banorte',
      image: financeImage,
      tags: ['React Native', 'Gen AI', 'UI/UX Design',  'LLM integration', 'Mobile', 'Hackathon', 'Education'],
      category: 'App Development'
    },
    {
      title: 'Differential Epidemic Model and Stochastic Simulation',
      description: 'Simulation and analysis of an SIR epidemic model with interactive visualization exploring urban center impact on disease dynamics.',
      githubLink: 'https://github.com/MarcosSaade/SIR-differential-visualizer',
      readMoreLink: '/sir',
      image: sirImage,
      tags: ['Simulation', 'Dynamic Systems', 'Differential Equations', 'Data Visualization', 'Numerical Methods', 'Epidemiology'],
      category: 'Physics & Simulation'
    },
    {
      title: 'Solar Panel Cleaning Simulation',
      description: 'MATLAB simulation of electric-field based solar panel cleaning under varying environmental conditions.',
      githubLink: 'https://github.com/MarcosSaade/solar-panel-cleaning',
      readMoreLink: '/solarpanel',
      image: solarPanelImage,
      tags: [
        'Computational Physics',
        'MATLAB Modeling',
        'Numerical Analysis',
        'Renewable Energy',
      ],
      category: 'Physics & Simulation'
    },
    {
      title: 'Chess-Playing AI',
      description: 'Minimax chess engine with alpha-beta pruning, opening book, position memory, and a Pygame interface.',
      githubLink: 'https://github.com/MarcosSaade/chess-engine',
      readMoreLink: '/chess',
      image: chessImage,
      tags: ['Algorithms and Data Structures', 'Pygame', 'AI', 'Alpha-Beta Pruning',],
      category: 'Machine Learning & AI'
    },
  ];

  const categories = ['Machine Learning & AI', 'App Development', 'Physics & Simulation'];
  const [collapsed, setCollapsed] = useState({
    'Machine Learning & AI': false,
    'App Development': false,
    'Physics & Simulation': false,
  });

  const toggleCategory = (category) => {
    setCollapsed(prev => ({
      ...prev,
      [category]: !prev[category],
    }));
  };

  return (
    <section id="projects" className="project-list">
      <h2>Projects</h2>
      <div className="all-projects-section">
        <h3 className="all-projects-title">Projects by Category</h3>
        {categories.map(category => {
        const filtered = projects.filter(p => p.category === category);
        return (
          <div key={category}>
            <h3
              className="category-title collapsible"
              onClick={() => toggleCategory(category)}
            >
              {category} {collapsed[category] ? '▸' : '▾'}
            </h3>
            {!collapsed[category] && (
              <div className="project-grid">
                {filtered.map((project, index) => (
                  <div key={index} style={{ '--card-index': index }}>
                    <ProjectCard
                      title={project.title}
                      description={project.description}
                      githubLink={project.githubLink}
                      articleLink={project.articleLink || null}
                      readMoreLink={project.readMoreLink}
                      image={project.image}
                      tags={project.tags}
                    />
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
      </div>
    </section>
  );
}

export default ProjectList;