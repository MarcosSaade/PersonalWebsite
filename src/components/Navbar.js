import React from 'react';
import { Link } from 'react-router-dom';
import { FiMail, FiSun, FiMoon } from 'react-icons/fi';
import { FaLinkedinIn, FaGithub } from 'react-icons/fa6';
import { useTheme } from '../contexts/ThemeContext';
import './Navbar.css';

function Navbar() {
  const { isDarkMode, toggleTheme } = useTheme();

  return (
    <nav className="navbar">
      <div className="navbar-left">
        <Link to="/" className="navbar-logo">Marcos Saade</Link>
      </div>
      <div className="navbar-right">
        <button
          onClick={toggleTheme}
          aria-label={isDarkMode ? 'Switch to light mode' : 'Switch to dark mode'}
          className="theme-toggle"
        >
          {isDarkMode ? <FiSun /> : <FiMoon />}
        </button>
        <a
          href="mailto:marcossr2626@gmail.com"
          aria-label="Email"
          className="icon-link"
        >
          <FiMail />
        </a>
        <a
          href="https://www.linkedin.com/in/marcos-saade/"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="LinkedIn"
          className="icon-link"
        >
          <FaLinkedinIn />
        </a>
        <a
          href="https://github.com/MarcosSaade"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="GitHub"
          className="icon-link"
        >
          <FaGithub />
        </a>
      </div>
    </nav>
  );
}

export default Navbar;
