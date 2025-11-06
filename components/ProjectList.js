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
      description: 'A machine learning pipeline for detection of Alzheimer\'s Dementia through automatic voice analysis. This project encompasses data preprocessing, extraction of acoustic, temporal, and complexity features from speech recordings, and addressing class imbalance. I used Recursive Feature Elimination for feature selection and an SVM for classification. The project also features model interpretability and a fullstack (React.js, Flask) webapp for real time diagnosis. Developed during my third semester as part of an academic paper for which I am the lead author',
      githubLink: 'https://github.com/MarcosSaade/DementiaDetection',
      articleLink: null,
      readMoreLink: '/dementia',
      image: dementiaImage,
      tags: ['Machine Learning', 'Data Preprocessing', 'Feature Engineering', 'Full-Stack Development', 'Scientific Research'],
      category: 'Machine Learning & AI'
    },
    {
      title: 'S&P 500 Tactical Allocation with ML',
      description: 'A sophisticated machine learning system for predicting optimal S&P 500 allocations, developed for a Kaggle competition with obfuscated features. Built a multi-stage pipeline combining gradient boosting for return prediction, volatility forecasting, and meta-labeling for confidence estimation. Implemented purged cross-validation to prevent label leakage, engineered 2,000+ features through quantitative analysis, and applied regime-dependent Kelly criterion for position sizing. Achieved Sharpe ratios of 0.99-1.88 across validation folds, demonstrating consistent risk-adjusted performance.',
      githubLink: 'https://github.com/MarcosSaade/optimal-sp500',
      readMoreLink: '/market-prediction',
      image: quantImage,
      tags: ['Financial ML', 'Time Series', 'Quantitative Finance', 'Feature Engineering', 'Risk Management', 'Kaggle'],
      category: 'Machine Learning & AI'
    },
    {
      title: 'Orthopedics Image Processing and Visualization App',
      description: 'A desktop application designed for an orthopedics clinic, featuring hardware integration with an Epson scanner for clinical image acquisition. The app leverages computer vision techniques to automate background removal and applies heatmap-style visualizations for enhanced diagnostic clarity. It also generates professional PDF reports, streamlining workflows for orthotic design and documentation. Developed as a freelance solution tailored to the needs of a local orthopedic clinic and used by around 4000 patients yearly.',
      githubLink: 'https://github.com/MarcosSaade/OrthoApp',
      readMoreLink: '/orthopedics',
      image: orthopedicsImage,
      tags: ['Medical Computer Vision', 'OpenCV', 'PyQt5', 'Hardware Integration', 'Freelance', 'Desktop App'],
      category: 'App Development'
    },
    {
      title: 'SugarZero: Self-Play RL for a Custom Board Game',
      description: 'An AlphaZero-inspired AI that masters the abstract strategy game "Sugar" through neural networks and self-play. Built with PyTorch and Pygame, this project reveals fascinating insights about the relationship between game design and reinforcement learning. Features parallelized self-play, Monte Carlo Tree Search, and a two Convolutional Neural Networks. Developed as a side project to explore Reinforcement Learning and game AI.',
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
      description: 'An fullstack app for capturing multimodal inputs —video (OpenPose), audio, accelerometers, demographic data, and cognitive tests— into unified sessions with precise timestamps. Includes live monitoring, session metadata, and export to CSV/JSON for downstream analysis. Built for the Center of Microsystems and Biodesign as a tool to streamline data collection in a study on gait, speech, and cognitive decline.',
      githubLink: 'https://github.com/MarcosSaade/NeuroCapture',
      readMoreLink: '/neurocapture',
      image: neuroCaptureImage,
      tags: ['Desktop App', 'Data Acquisition', 'SQL', 'Electron', 'Fullstack', 'Research'],
      category: 'App Development'
    },
    {
      title: 'Financial Education App',
      description: 'Developed for Hackathon Banorte 2024, this AI-powered financial education app offers personalized learning through interactive modules. Each module presents concise lessons on key personal finance topics, followed by choose-your-own-adventure scenarios that simulate real-world decisions tailored to the user’s profile (e.g., income, career, family). These risk-free simulations help users test their knowledge and build financial confidence. The app also supports financial goal tracking and is designed for integration with Banorte’s reward system, encouraging long-term engagement through redeemable perks and habit-forming design.',
      githubLink: 'https://github.com/MarcosSaade/banorteach',
      readMoreLink: '/banorte',
      image: financeImage,
      tags: ['React Native', 'Cloud-based AI', 'UI/UX Design',  'LLM integration', 'Mobile', 'Hackathon', 'Education'],
      category: 'App Development'
    },
    {
      title: 'Differential Epidemic Model and Stochastic Simulation',
      description: 'A dynamic and interactive stochastic simulation of the SIR epidemic model, developed as part of a differential equations course project. This simulation delves into the impact of an urban center on disease transmission. It features clear visualizations and in-depth analyses to enhance understanding of epidemiological trends. Developed for a course on Differential Equations and Dynamic Systems.',
      githubLink: 'https://github.com/MarcosSaade/SIR-differential-visualizer',
      readMoreLink: '/sir',
      image: sirImage,
      tags: ['Simulation', 'Dynamic Systems', 'Differential Equations', 'Data Visualization', 'Nummerical Methods', 'Epidemiology'],
      category: 'Physics & Simulation'
    },
    {
      title: 'Solar Panel Cleaning Simulation',
      description: 'A MATLAB-based project simulating an innovative cleaning technique for solar panels using electric fields to effectively remove sand and dust. The simulation explores different environmental conditions and demonstrates how electric field applications can enhance solar panel efficiency. This work combines engineering insight with practical applications in renewable energy. Developed as part of a course on Computational Physics.',
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
      description: 'A chess-playing AI built with the minimax algorithm enhanced by alpha-beta pruning. This project features custom implementations for chess piece movements and data structures for the chess board, combined with an intuitive, user-friendly interface developed using Pygame. Additionally, it includes memory of previously played positions and an opening book. Developed as a side project the summer after high school.',
      githubLink: 'https://github.com/MarcosSaade/chess-engine',
      readMoreLink: '/chess',
      image: chessImage,
      tags: ['Algorithms and Data Structures', 'Pygame', 'AI', 'Alpha-Beta Pruning',],
      category: 'Machine Learning & AI'
    },
  ];

  // Featured projects - the ones to highlight at the top
  const featuredProjectTitles = [
    'Dementia Detection ML Pipeline',
    'S&P 500 Tactical Allocation with ML',
    'NeuroCapture: Multimodal Data Capture App',
    'Orthopedics Image Processing and Visualization App',
    'SugarZero: Self-Play RL for a Custom Board Game'
  ];

  const featuredProjects = projects.filter(project => 
    featuredProjectTitles.includes(project.title)
  );

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
      
      {/* Featured Projects Section */}
      <div className="featured-projects-section">
        <h3 className="featured-title">Featured Projects</h3>
        <div className="project-grid featured-grid">
          {featuredProjects.map((project, index) => (
            <div key={index} className="featured-project" style={{ '--card-index': index }}>
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
      </div>

      {/* All Projects by Category */}
      <div className="all-projects-section">
        <h3 className="all-projects-title">All Projects</h3>
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