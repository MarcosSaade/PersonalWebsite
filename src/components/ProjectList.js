import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import FeaturedProjectCard from './FeaturedProjectCard';
import orthopedicsImage from '../images/orthopedics.jpeg';
import solarPanelImage from '../images/solarpanel.png';
import financeImage from '../images/finance.jpeg';
import quantImage from '../images/quant.png';
import dementiaImage from '../images/dementia.jpeg';
import sirImage from '../images/sir.jpeg';
import sugarzeroImage from '../images/sugarzero.png';
import neuroCaptureImage from '../images/neurocapture.png';
import fleetImage from '../images/fleet.png';
import salesImage from '../images/sales-analytics.png';
import visionSystemImage from '../images/vision-system.png'; // TODO: Add vision-system.png image
import './ProjectList.css';

function ProjectList() {
  const [hoveredProject, setHoveredProject] = useState(null);

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
      shortDescription: 'Production-grade medical imaging desktop application built for orthopedic clinics, serving 4,000+ patients annually across two locations. Implements custom computer vision pipeline using OpenCV for automatic background removal via HSV color space thresholding and morphological operations, generates diagnostic pressure heatmaps, integrates with hardware scanners via WIA interface, and produces automated clinic-branded PDF reports. Developed as freelance software engineering project with PyQt5 GUI and deployed in real clinical workflows.',
      description: 'Desktop app made for an Orthopedics clinic for clinical image acquisition. Features auto background removal, heatmap visualization, automatic PDF reporting, and hardware integration. Used by 4000+ patients yearly.',
      githubLink: 'https://github.com/MarcosSaade/OrthoApp',
      readMoreLink: '/orthopedics',
      image: orthopedicsImage,
      tags: ['Medical Computer Vision', 'OpenCV', 'PyQt5', 'Hardware Integration', 'Freelance', 'Desktop App']
    },
    {
      title: 'NeuroCapture: Multimodal Data Capture App',
      shortDescription: 'Full-stack research platform for neurodegenerative disease studies developed for the Center of Microsystems and Biodesign. Built with Electron + React frontend, FastAPI backend, and PostgreSQL database with Alembic migrations. Captures multimodal data (speech, gait/video via OpenPose, accelerometer, cognitive assessments) with referential integrity, transactional writes, and one-click CSV export. Implements automated audio preprocessing pipeline with noise reduction, VAD, and 150+ engineered acoustic features for downstream ML analysis.',
      description: 'Fullstack app capturing video (OpenPose), audio, accelerometer, demographic and cognitive test data. Developed for a study conducted by the Center of Microsystems and Biodesign.',
      githubLink: 'https://github.com/MarcosSaade/NeuroCapture',
      readMoreLink: '/neurocapture',
      image: neuroCaptureImage,
      tags: ['Desktop App', 'Data Acquisition', 'SQL', 'Electron', 'Fullstack', 'Research']
    },
    {
      title: 'Financial Education App',
      shortDescription: 'AI-powered financial literacy mobile app built for Banorte Hackathon 2024. Developed full stack (React Native + Node.js/Express) with Google Vertex AI integration for generating personalized, context-aware educational content and interactive choose-your-own-adventure financial scenarios. Features adaptive learning paths based on user demographics and goals, gamification with progress tracking, and real-time AI feedback on financial decisions to teach budgeting, investing, and credit management.',
      description: 'Gen AI-powered mobile app with personalized finance lessons, interactive scenarios, and goal tracking. Made for Banorte Hackathon 2024.',
      githubLink: 'https://github.com/MarcosSaade/banorteach',
      readMoreLink: '/banorte',
      image: financeImage,
      tags: ['React Native', 'Gen AI', 'UI/UX Design', 'LLM integration', 'Mobile', 'Hackathon', 'Education']
    },
    {
      title: 'Differential Epidemic Model and Stochastic Simulation',
      shortDescription: 'Mathematical modeling project implementing SIR (Susceptible-Infected-Recovered) epidemic dynamics using systems of coupled differential equations. Conducted analytical derivations for peak infection timing and basic reproduction number R₀, implemented numerical integration via Euler\'s method, and built interactive visualizations exploring vaccination scenarios, variable contact rates, and spatial urban center effects. Demonstrates scientific computing skills and understanding of dynamical systems theory.',
      description: 'Simulation and analysis of an SIR epidemic model with interactive visualization exploring urban center impact on disease dynamics.',
      githubLink: 'https://github.com/MarcosSaade/SIR-differential-visualizer',
      readMoreLink: '/sir',
      image: sirImage,
      tags: ['Simulation', 'Dynamic Systems', 'Differential Equations', 'Data Visualization', 'Numerical Methods', 'Epidemiology']
    },
    {
      title: 'Solar Panel Cleaning Simulation',
      shortDescription: 'Computational physics project simulating electric-field based solar panel cleaning for desert environments. Implemented 3D electrostatics model in MATLAB using Coulomb\'s law and superposition principle, applied numerical integration (Euler method) to compute particle trajectories under combined electric and gravitational forces, and optimized system parameters (charge values, bar positioning, rotation mechanisms) for water-free dust removal. Demonstrates applied physics, numerical methods, and engineering problem-solving.',
      description: 'MATLAB simulation of electric-field based solar panel cleaning under varying environmental conditions.',
      githubLink: 'https://github.com/MarcosSaade/solar-panel-cleaning',
      readMoreLink: '/solarpanel',
      image: solarPanelImage,
      tags: ['Computational Physics', 'MATLAB Modeling', 'Numerical Analysis', 'Renewable Energy']
    },
    {
      title: 'Fleet Optimization with Monte Carlo Simulation',
      shortDescription: 'Operations research project optimizing my university\'s transportation fleet configuration. Built discrete-event simulation framework in Python to model stochastic passenger arrivals, vehicle scheduling, and queueing dynamics. Evaluated 48 fleet configurations across 50 Monte Carlo runs each (2,400+ simulations), balancing service quality (average wait time < 10 min) with operational costs. Applied optimization techniques and statistical analysis to provide data-driven recommendations for resource allocation under uncertainty.',
      description: 'Discrete-event simulation framework optimizing corporate transportation fleet configuration using Monte Carlo analysis to balance service quality with operational costs.',
      githubLink: 'https://github.com/MarcosSaade/fleet-optimization',
      readMoreLink: '/fleet-optimization',
      image: fleetImage,
      tags: ['Operations Research', 'Monte Carlo Simulation', 'Optimization', 'Queueing Theory', 'Python']
    },
    {
      title: 'Retail Sales Forecasting with Machine Learning',
      shortDescription: 'End-to-end machine learning pipeline for retail demand forecasting across Argentine regions and product categories. Performed time series analysis with moving averages and seasonality decomposition, applied unsupervised clustering (K-means) to segment products, engineered temporal and categorical features, and trained Random Forest models with hyperparameter tuning. Delivered business insights on regional sales patterns, payment preferences, and demand cycles to optimize inventory management and resource allocation for a real retail dataset.',
      description: 'Machine learning pipeline for forecasting retail demand across regions and product categories using time series analysis, clustering, and predictive modeling.',
      githubLink: 'https://github.com/MarcosSaade/sales-forecasting',
      readMoreLink: '/sales-prediction',
      image: salesImage,
      tags: ['Machine Learning', 'Time Series', 'Forecasting', 'Clustering', 'Data Science', 'Business Analytics']
    },
    {
      title: 'AI Vision System with Natural Language Queries',
      shortDescription: 'Full-stack edge AI system combining real-time object detection (YOLO11) with natural language query interface powered by LLMs. Achieved 8x inference speedup (3 FPS → 27 FPS) using Qualcomm NPU hardware acceleration with INT8 quantization while running cooler. Built with React frontend, FastAPI backend, and dual LLM support (local Ollama and cloud Gemini). Features Server-Sent Events streaming, SQL query generation with validation, and optimized SQLite database for time-series detection data.',
      description: 'Real-time object detection with natural language query interface. 8x faster inference using NPU acceleration on Qualcomm Rubik Pi Board.',
      githubLink: 'https://github.com/MarcosSaade/smart-vision',
      readMoreLink: '/vision-system',
      image: visionSystemImage,
      tags: ['Computer Vision', 'Edge AI', 'LLM Integration', 'Hardware Acceleration', 'Fullstack', 'Real-Time Systems']
    }
  ];

  // Featured projects
  const featuredProjects = [
    projects.find(p => p.title === 'S&P 500 Tactical Allocation with ML'),
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
      projects.find(p => p.title === 'Differential Epidemic Model and Stochastic Simulation'),
      projects.find(p => p.title === 'Solar Panel Cleaning Simulation')
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