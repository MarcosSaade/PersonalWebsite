import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowLeft, ChevronUp, Twitter, Linkedin, Facebook } from 'lucide-react';
import './PageTemplate.css';

export default function PageTemplate({ title, image, children }) {
  const [tableOfContents, setTableOfContents] = useState([]);
  const [showScrollTop, setShowScrollTop] = useState(false);
  const [readingTime, setReadingTime] = useState(null);
  
  // Generate table of contents from h2 headings after component mounts and content is rendered
  useEffect(() => {
    // Wait for content to be rendered
    setTimeout(() => {
      const contentElement = document.querySelector('.page-content');
      if (!contentElement) return;
      
      // Find all h2 elements within the content
      const headings = Array.from(contentElement.querySelectorAll('h2'));
      
      // Generate TOC items
      const toc = headings.map(heading => {
        // Create an ID if it doesn't exist
        if (!heading.id) {
          const id = heading.textContent
            .toLowerCase()
            .replace(/[^\w\s]/g, '')
            .replace(/\s+/g, '-');
          heading.id = id;
        }
        
        return {
          id: heading.id,
          text: heading.textContent
        };
      });
      
      setTableOfContents(toc);
      
      // Calculate reading time
      if (contentElement) {
        const text = contentElement.textContent || '';
        const wordCount = text.split(/\s+/).filter(Boolean).length;
        const time = Math.ceil(wordCount / 200); // Average reading speed: 200 wpm
        setReadingTime(time);
      }
    }, 500); // Small delay to ensure content is rendered
  }, [children]);
  
  // Scroll to top button visibility
  useEffect(() => {
    const handleScroll = () => {
      setShowScrollTop(window.scrollY > 300);
    };
    
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);
  
  // Scroll to top function
  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <section className="page-template">
      <header className="page-header">
        <Link to="/#projects" className="back-link">
          <ArrowLeft size={20} />
          <span>Back to Projects</span>
        </Link>
        <div className="reading-time">
          {readingTime ? `${readingTime} min read` : ''}
        </div>
      </header>
      
      <h1 className="page-title">{title}</h1>
      
      {image && (
        <div className="featured-image-container">
          <img src={image} alt={title} className="featured-image" />
        </div>
      )}
      
      <div className="content-layout">
        {tableOfContents.length > 0 && (
          <aside className="table-of-contents">
            <div className="toc-sticky">
              <h2>Contents</h2>
              <nav>
                <ul>
                  {tableOfContents.map((item) => (
                    <li key={item.id} className="toc-item">
                      <a href={`#${item.id}`}>{item.text}</a>
                    </li>
                  ))}
                </ul>
              </nav>
            </div>
          </aside>
        )}
        
        <article className="page-content">
          {children}
        </article>
      </div>
      
      <footer className="page-footer">
        <div className="share-links">
          <h3>Share this article</h3>
          <div className="social-buttons">
            <button 
              className="social-button twitter" 
              onClick={() => window.open(`https://twitter.com/intent/tweet?url=${encodeURIComponent(window.location.href)}&text=${encodeURIComponent(title)}`, '_blank')}
              aria-label="Share on Twitter"
            >
              <Twitter size={18} />
              <span>Twitter</span>
            </button>
            <button 
              className="social-button linkedin"
              onClick={() => window.open(`https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(window.location.href)}`, '_blank')}
              aria-label="Share on LinkedIn"
            >
              <Linkedin size={18} />
              <span>LinkedIn</span>
            </button>
            <button 
              className="social-button facebook"
              onClick={() => window.open(`https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(window.location.href)}`, '_blank')}
              aria-label="Share on Facebook"
            >
              <Facebook size={18} />
              <span>Facebook</span>
            </button>
          </div>
        </div>
        
        <div className="page-navigation">
          <Link to="/#projects" className="back-navigation">
            <ArrowLeft size={20} />
            <span>Back to Projects</span>
          </Link>
        </div>
      </footer>
      
      {showScrollTop && (
        <button 
          className="scroll-to-top" 
          onClick={scrollToTop}
          aria-label="Scroll to top"
        >
          <ChevronUp size={24} />
        </button>
      )}
    </section>
  );
}