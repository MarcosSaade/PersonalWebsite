// src/App.js
import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import AboutMe from './components/AboutMe';
import ProjectList from './components/ProjectList';

import DementiaPage from './pages/DementiaPage';
import ChessPage from './pages/ChessPage';
import BanortePage from './pages/BanortePage';
import SIRPage from './pages/SIRPage';
import OrthopedicsPage from './pages/OrthopedicsPage';
import SolarPanelPage from './pages/SolarPanelPage';
import SugarPage from './pages/SugarPage';

import './App.css';

function App() {
  return (
    <div className="App">
      <Navbar />
      <main>
        <Routes>
          <Route path="/" element={
            <>
              <AboutMe />
              <ProjectList />
            </>
          } />
          <Route path="/dementia" element={<DementiaPage />} />
          <Route path="/chess" element={<ChessPage />} />
          <Route path="/banorte" element={<BanortePage />} />
          <Route path="/sir" element={<SIRPage />} />
          <Route path="/orthopedics" element={<OrthopedicsPage />} />
          <Route path="/solarpanel" element={<SolarPanelPage />} />
          <Route path="/sugarzero" element={<SugarPage />} />
        </Routes>
      </main>
      <Footer />
    </div>
  );
}

export default App;
