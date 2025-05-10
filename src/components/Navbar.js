import React from 'react';
import './Navbar.css';
import { Link } from 'react-router-dom';
import { FaEnvelope, FaLinkedin, FaGithub } from 'react-icons/fa';

function Navbar() {
  return (
    <nav className="navbar" aria-label="Main Navigation">
      <div className="navbar-logo">
        <Link to="/" className="logo-text">Marcos Saade</Link>
      </div>
      <ul className="navbar-links">
        <li>
          <a href="mailto:marcossr2626@gmail.com" aria-label="Email">
            <FaEnvelope style={{ marginRight: '6px' }} />
          </a>
        </li>
        <li>
          <a href="https://www.linkedin.com/in/marcos-saade" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn">
            <FaLinkedin style={{ marginRight: '6px' }} />
          </a>
        </li>
        <li>
          <a href="https://github.com/MarcosSaade" target="_blank" rel="noopener noreferrer" aria-label="GitHub">
            <FaGithub style={{ marginRight: '6px' }} />
          </a>
        </li>
      </ul>
    </nav>
  );
}

export default Navbar;
