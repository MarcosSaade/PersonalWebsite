import React, { useState, useRef, useEffect } from 'react';
import './FeaturedProjectCard.css';
import { Link } from 'react-router-dom';

function FeaturedProjectCard({ title, description, githubLink, readMoreLink, image, tags }) {
  const [isVisible, setIsVisible] = useState(false);
  const [imageLoaded, setImageLoaded] = useState(false);
  const cardRef = useRef(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.unobserve(entry.target);
        }
      },
      { threshold: 0.1 }
    );

    if (cardRef.current) {
      observer.observe(cardRef.current);
    }

    return () => observer.disconnect();
  }, []);

  return (
    <div 
      ref={cardRef}
      className={`featured-project-card ${isVisible ? 'animate-in' : ''}`}
    >
      <img 
        src={image} 
        alt={title} 
        className={`featured-project-image ${imageLoaded ? 'loaded' : 'loading'}`}
        onLoad={() => setImageLoaded(true)}
      />
      <div className="featured-project-content">
        <h3>{title}</h3>
        <p>{description}</p>

        {tags && tags.length > 0 && (
          <div className="featured-project-tags">
            {tags.map((tag, idx) => (
              <span key={idx} className="featured-project-tag">{tag}</span>
            ))}
          </div>
        )}

        <div className="featured-project-links">
          {githubLink && (
            <a
              href={githubLink}
              target="_blank"
              rel="noopener noreferrer"
              className="featured-text-link"
            >
              GitHub →
            </a>
          )}

          {readMoreLink && (
            <Link to={readMoreLink} className="featured-text-link">
              Read More →
            </Link>
          )}
        </div>
      </div>
    </div>
  );
}

export default FeaturedProjectCard;
