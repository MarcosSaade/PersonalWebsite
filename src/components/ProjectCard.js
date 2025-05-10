// src/components/ProjectCard.js
import React from 'react';
import './ProjectCard.css';
import { Link } from 'react-router-dom';

function ProjectCard({ title, description, githubLink, articleLink, image, tags, readMoreLink }) {
  return (
    <div className="project-card">
      <img src={image} alt={title} className="project-image" />
      <div className="project-content">
        <h3>{title}</h3>
        <p>{description}</p>

        {tags && tags.length > 0 && (
          <div className="project-tags">
            {tags.map((tag, idx) => (
              <span key={idx} className="project-tag">{tag}</span>
            ))}
          </div>
        )}

        <div className="project-links">
          {githubLink && (
            <a
              href={githubLink}
              target="_blank"
              rel="noopener noreferrer"
              className="project-link"
            >
              GitHub
            </a>
          )}

          {articleLink && (
            <a
              href={articleLink}
              target="_blank"
              rel="noopener noreferrer"
              className="project-link"
            >
              Read Preprint
            </a>
          )}

          {readMoreLink && (
            <Link to={readMoreLink} className="project-link">
              Read More
            </Link>
          )}
        </div>
      </div>
    </div>
  );
}

export default ProjectCard;
