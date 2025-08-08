import React from 'react';
import './Footer.css';

function Footer() {
  return (
    <footer className="footer">
      <div className="footer-inner">&copy; {new Date().getFullYear()} Marcos Saade</div>
    </footer>
  );
}

export default Footer;
