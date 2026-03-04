import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import FeaturedProjectCard from './FeaturedProjectCard';
import orthopedicsImage from '../images/orthopedics.jpeg';
import financeImage from '../images/finance.jpeg';
import quantImage from '../images/quant.png';
import dementiaImage from '../images/dementia.png';
import sirImage from '../images/sir.jpeg';
import sugarzeroImage from '../images/sugarzero.png';
import neuroCaptureImage from '../images/neurocapture.png';
import fleetImage from '../images/fleet.png';
import salesImage from '../images/sales-analytics.png';
import visionSystemImage from '../images/vision-system.png';
import './ProjectList.css';

function ProjectList() {
  const [hoveredProject, setHoveredProject] = useState(null);

  const projects = [
    {
      title: 'Dementia Detection ML Pipeline',
      description: 'Speech-based Alzheimer\'s detection (86% accuracy) using engineered acoustic features and an SVM, with a real-time React/Flask web app. Part of an academic paper where I was first author.',
      githubLink: 'https://github.com/MarcosSaade/DementiaDetection',
      readMoreLink: '/dementia',
      image: dementiaImage,
      tags: ['Machine Learning', 'Signal processing', 'Feature Engineering', 'WebApp', 'Scientific Research']
    },
    {
      title: 'Optimal Daily S&P 500 Allocation with ML',
      description: 'ML pipeline optimizing daily S&P 500 allocation by predicting returns & volatility with multiple techniques (51.7% directional accuracy; 3x sharpe vs buy and hold). Made for a Kaggle competition.',
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
      shortDescription: 'Desktop app for orthopedic clinics, serving 4,000+ patients/year. OpenCV-based background removal, pressure heatmaps, automated PDF reports, and hardware scanner integration via WIA. Deployed in PyQt5.',
      description: 'Desktop app made for an Orthopedics clinic for clinical image acquisition. Features auto background removal, heatmap visualization, automatic PDF reporting, and hardware integration. Used by 4000+ patients yearly.',
      githubLink: 'https://github.com/MarcosSaade/OrthoApp',
      readMoreLink: '/orthopedics',
      image: orthopedicsImage,
      tags: ['Medical Computer Vision', 'OpenCV', 'PyQt5', 'Hardware Integration', 'Freelance', 'Desktop App']
    },
    {
      title: 'NeuroCapture: Multimodal Data Capture App',
      description: 'Research platform for neurodegenerative disease studies (Electron + React, FastAPI, PostgreSQL). Facilitates data collection and management across speech, gait (OpenPose), accelerometer, and cognitive assessments. Developed for a research lab.',
      githubLink: 'https://github.com/MarcosSaade/NeuroCapture',
      readMoreLink: '/neurocapture',
      image: neuroCaptureImage,
      tags: ['Desktop App', 'Data Acquisition', 'SQL', 'Electron', 'Fullstack', 'Research']
    },
    {
      title: 'Financial Education App',
      shortDescription: 'AI-powered financial literacy mobile app built for Banorte Hackathon 2024 (React Native + Node.js, Google Vertex AI). Features personalized lessons, choose-your-own-adventure financial scenarios, adaptive learning paths, and real-time AI feedback on financial decisions.',
      description: 'Gen AI-powered mobile app with personalized finance lessons, interactive scenarios, and goal tracking. Made for Banorte Hackathon 2024.',
      githubLink: 'https://github.com/MarcosSaade/banorteach',
      readMoreLink: '/banorte',
      image: financeImage,
      tags: ['React Native', 'Gen AI', 'UI/UX Design', 'LLM integration', 'Mobile', 'Hackathon', 'Education']
    },
    {
      title: 'Differential Epidemic Model and Stochastic Simulation',
      shortDescription: 'SIR epidemic dynamics modeled with coupled differential equations. Derives R₀ and peak infection timing analytically, implements Euler\'s method for numerical integration, and visualizes vaccination scenarios, variable contact rates, and spatial urban center effects.',
      description: 'Simulation and analysis of an SIR epidemic model with interactive visualization exploring urban center impact on disease dynamics.',
      githubLink: 'https://github.com/MarcosSaade/SIR-differential-visualizer',
      readMoreLink: '/sir',
      image: sirImage,
      tags: ['Simulation', 'Dynamic Systems', 'Differential Equations', 'Data Visualization', 'Numerical Methods', 'Epidemiology']
    },
    {
      title: 'Fleet Optimization with Monte Carlo Simulation',
      shortDescription: 'Python discrete-event simulation optimizing a university fleet across 48 configurations and 500 Monte Carlo runs each. Balances service quality (avg wait < 10 min) with operational costs using queueing theory and statistical analysis.',
      description: 'Discrete-event simulation framework optimizing corporate transportation fleet configuration using Monte Carlo analysis to balance service quality with operational costs.',
      githubLink: 'https://github.com/MarcosSaade/fleet-optimization',
      readMoreLink: '/fleet-optimization',
      image: fleetImage,
      tags: ['Operations Research', 'Monte Carlo Simulation', 'Optimization', 'Queueing Theory', 'Python']
    },
    {
      title: 'Retail Sales Forecasting with Machine Learning',
      shortDescription: 'ML pipeline for retail demand forecasting across Argentine regions and product categories. Applies time series decomposition, K-means clustering, and Random Forest models to deliver inventory and resource allocation insights.',
      description: 'Machine learning pipeline for forecasting retail demand across regions and product categories using time series analysis, clustering, and predictive modeling.',
      githubLink: 'https://github.com/MarcosSaade/sales-forecasting',
      readMoreLink: '/sales-prediction',
      image: salesImage,
      tags: ['Machine Learning', 'Time Series', 'Forecasting', 'Clustering', 'Data Science', 'Business Analytics']
    },
    {
      title: 'AI Vision System with Natural Language Queries',
      description: 'Edge AI powered object detection system with natural language queries using LLMs. Achieves 8x faster inference using NPU acceleration on Qualcomm Rubik Pi Board.',
      githubLink: 'https://github.com/MarcosSaade/smart-vision',
      readMoreLink: '/vision-system',
      image: visionSystemImage,
      tags: ['Computer Vision', 'Edge AI', 'LLM Integration', 'Hardware Acceleration', 'Fullstack', 'Real-Time Systems']
    }
  ];

  // Featured projects
  const featuredProjects = [
    projects.find(p => p.title === 'Optimal Daily S&P 500 Allocation with ML'),
    projects.find(p => p.title === 'Dementia Detection ML Pipeline'),
    projects.find(p => p.title === 'AI Vision System with Natural Language Queries')
  ].filter(Boolean);

  // Categorized non-featured projects
  const categorizedProjects = {
    'Data Science': [
      projects.find(p => p.title === 'Retail Sales Forecasting with Machine Learning')
    ],
    'Deep Learning': [
      projects.find(p => p.title === 'SugarZero: Self-Play RL for a Custom Board Game')
    ],
    'Apps': [
      projects.find(p => p.title === 'Orthopedics Image Processing and Visualization App'),
      projects.find(p => p.title === 'NeuroCapture: Multimodal Data Capture App'),
      projects.find(p => p.title === 'Financial Education App')
    ],
    'Optimization': [
      projects.find(p => p.title === 'Fleet Optimization with Monte Carlo Simulation')
    ],
    'Modeling': [
      projects.find(p => p.title === 'Differential Epidemic Model and Stochastic Simulation')
    ]
  };

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

      {/* Other Projects by Category */}
      <div className="other-projects-section">
        <h3 className="other-projects-title">Other Projects (click to read more)</h3>
        
        {/* Hover Description Display - Fixed on Right */}
        <div className={`project-description-display ${hoveredProject ? 'visible' : ''}`}>
          <p className="hover-description">{hoveredProject}</p>
        </div>

        {Object.entries(categorizedProjects).map(([category, categoryProjects]) => (
          <div key={category} className="project-category">
            <h4 className="category-title">{category}</h4>
            <div className="project-links">
              {categoryProjects.filter(Boolean).map((project) => (
                <Link 
                  key={project.title}
                  to={project.readMoreLink} 
                  className="project-link"
                  onMouseEnter={() => setHoveredProject(project.shortDescription || project.description)}
                  onMouseLeave={() => setHoveredProject(null)}
                >
                  {project.title}
                </Link>
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

export default ProjectList;