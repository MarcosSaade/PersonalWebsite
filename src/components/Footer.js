import React from 'react';
import './Footer.css';

function Footer() {
  return (
    <footer className="footer">
      <p>© {new Date().getFullYear()} Marcos Saade. All rights reserved.</p>
      <div className="contact-info">
        <a href="mailto:marcossr2626@gmail.com">marcossr2626@gmail.com</a>
        <a href="https://www.linkedin.com/in/marcos-saade" target="_blank" rel="noopener noreferrer">LinkedIn</a>
        <a href="https://github.com/MarcosSaade" target="_blank" rel="noopener noreferrer">GitHub</a>
      </div>
    </footer>
  );
}

export default Footer;
