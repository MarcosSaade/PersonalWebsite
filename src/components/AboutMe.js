import React from 'react';
import './AboutMe.css';

function AboutMe() {
  return (
    <section id="about" className="about-me">
      <div className="about-container">
        <h2 className="about-title">About Me</h2>
        <p>
          Hi, I'm Marcos Saade — a Data Scientist, Software Engineer, and Mathematician with a passion for
          machine learning, AI, and building useful, human-centered technology. I enjoy working on projects that
          blend technical depth with real-world impact, from early diagnosis tools for neurological conditions to
          game-playing agents and finance models.
        </p>
        <p>
          I’ve also worked on freelance and research-based software projects, focusing on full-stack development,
          intelligent systems, and applications that combine data science with hardware integration. My experience
          includes handling diverse data types such as tabular data, audio, and images.
        </p>
        <p>
          Outside of class and research, I’m also deeply interested in physics, game development, and
          cybersecurity.
        </p>
        <p>
          On this website, you can explore some of my personal and academic projects. Each one includes a brief
          write-up explaining the problem, the approach I took, and what I learned along the way.
        </p>

        <p><strong>Languages:</strong> Python, C++, JavaScript, MATLAB, R, Java.</p>
        <p>
          <strong>Frameworks & Libraries:</strong> FastAPI, Flask, React.js/React Native, Express, Electron, SQLAlchemy,
          Alembic, PyTorch, Scikit-Learn, Pandas, NumPy, OpenCV, PyQt, Git.
        </p>
        <p>
          <strong>Technologies:</strong> PostgreSQL, REST APIs, full‑stack development, cross‑platform desktop apps.
        </p>
        <p>
          <strong>Skills:</strong> Machine Learning, Data Engineering, Backend Architecture, Statistical Modeling,
          Algorithms and Data Structures.
        </p>
        <p>
          <strong>GitHub:</strong> <a href="https://github.com/MarcosSaade" target="_blank" rel="noreferrer">github.com/MarcosSaade</a>
        </p>
      </div>
    </section>
  );
}

export default AboutMe;
