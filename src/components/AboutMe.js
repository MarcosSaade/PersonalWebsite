import React from 'react';
import './AboutMe.css';

function AboutMe() {
  return (
    <section id="about" className="about-me">
      <div className="about-container">
        <h2 className="about-title">About Me</h2>
        <p>
          Hi, I'm Marcos Saade — a Machine Learning Engineer and Applied Mathematician specializing in building 
          production-grade ML systems that solve real-world problems. My work spans computer vision, natural language 
          processing, audio signal processing, reinforcement learning, and optimization—with a consistent focus on 
          deploying models that make measurable impact.
        </p>
        <p>
          I've built systems that directly serve thousands of people: a clinical image processing application now used 
          by orthopedics clinics serving over 4,000 patients annually, an operations research solution that optimized 
          my university's transportation fleet (reducing costs while serving hundreds of students and staff daily), and 
          machine learning models for early Alzheimer's detection through voice analysis, a technology with potential to 
          help millions worldwide.
        </p>
        <p>
          My expertise includes model optimization and deployment at the edge and end-to-end ML pipelines, from data preprocessing and feature engineering to model training, 
          validation, and production deployment. I'm comfortable working across the full stack: building React frontends, 
          FastAPI backends, designing database schemas, and implementing real-time inference systems.
        </p>
        <p>
          This portfolio showcases projects across Computer Vision, NLP, audio ML, time-series forecasting, operations 
          research, and reinforcement learning.
        </p>

        <p><strong>Languages:</strong> Python, C++, JavaScript, MATLAB, R, SQL.</p>
        <p>
          <strong>ML/AI:</strong> NumPy, Pandas, PyTorch, Scikit-Learn, OpenCV, Keras, llama.cpp, ONNX, TensorRT, Librosa.
        </p>
        <p>
          <strong>Engineering:</strong> FastAPI, Flask, React, Node, Electron, PostgreSQL, SQLAlchemy, Docker, Git, REST APIs, CI/CD.
        </p>
        <p>
          <strong>GitHub:</strong> <a href="https://github.com/MarcosSaade" target="_blank" rel="noreferrer">github.com/MarcosSaade</a>
        </p>
      </div>
    </section>
  );
}

export default AboutMe;
