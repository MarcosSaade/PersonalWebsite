import React from 'react';
import PageTemplate from '../components/PageTemplate';

import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

import saharaImage from '../images/solarpanel.png';
import forcesDiagram from '../images/solar/forces-diagram.png';
import electricFieldVisualization from '../images/solar/electric-field-visualization.png';
import panelCleaningDiagram from '../images/solar/panel-cleaning-diagram.png';
import systemOverview from '../images/solar/system-overview.png';
import systemOverview2 from '../images/solar/system-overview-3d.png';

export default function SolarPanelProject() {
  return (
    <PageTemplate title="Innovative Electric Field-Based Solar Panel Cleaning System for Desert Environments" image={saharaImage}>
      <p>
        In this computational physics course project, I implemented a computational simulation for a sustainable solar panel cleaning system 
        designed specifically for harsh desert environments like the Sahara. The solution leverages electric fields 
        to remove sand and dust particles from solar panels with minimal water usage, addressing one of the 
        most significant challenges in desert-based solar power generation: maintaining efficiency without 
        consuming precious water resources.
      </p>

      <h2>The Challenge: Desert Solar Panel Maintenance</h2>
      <p>
        Solar power generation in desert regions faces a persistent challenge: sand and dust accumulation on panel surfaces.
        In the Sahara Desert, where solar potential is enormous but water is scarce, traditional cleaning methods pose significant problems:
      </p>
      <ul>
        <li>Water-based cleaning methods consume valuable water resources in water-scarce regions</li>
        <li>Transporting water to remote solar installations adds significant logistical complexity and costs</li>
        <li>Mechanical brushing can damage panel surfaces over time, reducing their lifespan</li>
        <li>Manual cleaning is labor-intensive and inefficient for large-scale installations</li>
        <li>Uncleaned panels can lose 10-40% efficiency in desert environments within weeks</li>
        <li>The harsh climate accelerates dust accumulation compared to other regions</li>
      </ul>

      <h2>My Solution: Electric Field-Based Cleaning System</h2>
      <p>
        I developed a computational model that demonstrates how strategically positioned charged bars can generate electric fields 
        capable of removing sand and dust particles from solar panel surfaces. This approach takes advantage of the natural 
        electrostatic properties of sand particles in desert environments, where friction and low humidity contribute to static charge buildup.
      </p>

      <p>
        The system consistes of:
      </p>
      <ul>
        <li>Negatively charged bars positioned above the panel surface</li>
        <li>Positively charged bars creating an opposing electric field</li>
        <li>A controlled rotation mechanism that sweeps the electric field across the entire panel</li>
        <li>Precision power control to maintain optimal charge values for different environmental conditions</li>
      </ul>

      <div className="image-container">
        <img src={systemOverview} alt="Solar Panel Cleaning System Overview" className="page-image" />
        <img src={panelCleaningDiagram} alt="Electric Field Cleaning Mechanism" className="page-image" />
        <img src={systemOverview2} alt="3D Overview of the System" className="page-image" />
      </div>

      <h2>Technical Deep Dive: Computational Modeling</h2>
      <p>
        The core of my project involved developing a comprehensive mathematical model and simulation to test 
        and validate the electric field cleaning concept. This computational approach allowed me to optimize the system 
        parameters before physical prototyping, saving considerable resources while exploring multiple design variations.
      </p>

      <h3>Mathematical Framework</h3>
      <p>
        The model is built on fundamental electrostatics principles and numerical methods. I established a frame of reference 
        with precise dimensions matching standard solar panel configurations (1.7m × 1m), then implemented the following key elements:
      </p>

      <ul>
        <li><strong>Particle Modeling:</strong> Sand grains represented as charged particles with mass (<InlineMath>{"m = 6.3\\times10^{-5}"}</InlineMath>) and charge (<InlineMath>{"q = 0.5\\times10^{-12}"}</InlineMath>)</li>
        
        <li><strong>Electric Field Calculation:</strong> I built a custom implementation of 3D electric field computation using Coulomb's law, where the electric field <InlineMath>{"\\vec{E}"}</InlineMath> at position <InlineMath>{"\\vec{r}"}</InlineMath> due to a point charge <InlineMath>{"q"}</InlineMath> is given by:
          <BlockMath>
            {"\\vec{E}(\\vec{r}) = \\frac{1}{4\\pi\\varepsilon_0} \\frac{q}{|\\vec{r}|^3}\\vec{r}"}
          </BlockMath>
        </li>

        <li><strong>Force Analysis:</strong> Vector calculation of electric forces <InlineMath>{"\\vec{F}_e = q\\vec{E}"}</InlineMath> and gravitational forces <InlineMath>{"\\vec{F}_g = m\\vec{g}"}</InlineMath> on each particle, resulting in a net force:
          <BlockMath>
            {"\\vec{F}_{net} = q\\vec{E} + m\\vec{g}"}
          </BlockMath>
        </li>

        <li><strong>Numerical Integration:</strong> Euler's method for simulating particle trajectories over time, updating position <InlineMath>{"\\vec{r}"}</InlineMath> and velocity <InlineMath>{"\\vec{v}"}</InlineMath> with time step <InlineMath>{"\\Delta t"}</InlineMath>:
          <BlockMath>
            {"\\vec{v}_{i+1} = \\vec{v}_i + \\frac{\\vec{F}_{net}}{m}\\Delta t"}
          </BlockMath>
          <BlockMath>
            {"\\vec{r}_{i+1} = \\vec{r}_i + \\vec{v}_i\\Delta t"}
          </BlockMath>
        </li>

        <li><strong>Bar Optimization:</strong> Mathematical determination of optimal bar length for complete panel coverage, derived from the panel geometry using the Pythagorean relationship:
          <BlockMath>
            {"L = \\frac{1}{2}\\sqrt{b^2 + h^2}"}
          </BlockMath>
          where <InlineMath>{"b"}</InlineMath> is the panel width and <InlineMath>{"h"}</InlineMath> is the panel height.
        </li>
      </ul>

      <p>
        For the superposition of electric fields from multiple charge points, I applied the principle of superposition where the total electric field <InlineMath>{"\\vec{E}_{total}"}</InlineMath> at a point is the vector sum of the contributions from all individual charges:
      </p>

      <BlockMath>
        {"\\vec{E}_{total}(\\vec{r}) = \\sum_{i=1}^{N} \\frac{1}{4\\pi\\varepsilon_0} \\frac{q_i}{|\\vec{r} - \\vec{r}_i|^3}(\\vec{r} - \\vec{r}_i)"}
      </BlockMath>

      <p>
        The spinning charged bars were heuristically modeled as discrete charge points arranged in circular patterns. 
        This approach allowed for efficient computation while maintaining physical accuracy in the simulation:
      </p>

      <ul>
        <li>10 concentric rings with 100 charge points per ring (1,000 total points)</li>
        <li>Charge distribution of <InlineMath>{"7.44\\times10^{-10}"}</InlineMath> C across the system</li>
        <li>Optimal height positioning at 0.1m above the panel surface</li>
      </ul>

      <p>
        To determine the critical lift threshold, I calculated the minimum electric field strength <InlineMath>{"E_{min}"}</InlineMath> needed to overcome gravity:
      </p>

      <BlockMath>
        {"E_{min} = \\frac{mg}{q} = \\frac{(6.3\\times10^{-5})(9.8)}{0.5\\times10^{-12}} \\approx 1.23\\times10^{6} \\text{ N/C}"}
      </BlockMath>

      <p>
        The electric potential <InlineMath>{"\\phi"}</InlineMath> around the charged bars was also calculated to verify field uniformity:
      </p>

      <BlockMath>
        {"\\phi(\\vec{r}) = \\sum_{i=1}^{N} \\frac{1}{4\\pi\\varepsilon_0} \\frac{q_i}{|\\vec{r} - \\vec{r}_i|}"}
      </BlockMath>

      <h3>Force Analysis and Vector Fields</h3>

      <p>
        A critical aspect of the model was calculating the exact forces experienced by sand particles on the panel surface. 
        The simulation computed the combined effect of:
      </p>

      <ul>
        <li>Electric forces from the negatively charged bar (repulsive or attractive based on particle charge)</li>
        <li>Electric forces from the positively charged bar</li>
        <li>Gravitational force keeping particles on the panel</li>
        <li>The resultant vectors determining particle movement trajectories</li>
      </ul>

      <p>
        The force vectors were visualized in both 2D planes (XZ and YZ) and in 3D space, providing comprehensive 
        insight into particle behavior under the influence of the electric field. This visualization confirmed that 
        the system could generate sufficient lift and lateral movement to effectively clean the panel surface.
      </p>

      <img src={forcesDiagram} alt="Force Analysis on Sand Particles" className="page-image" />
      <img src={electricFieldVisualization} alt="Electric Field Visualization" className="page-image" />

      <h3>Particle Trajectory Simulation</h3>
      <p>
        To evaluate the physical feasibility of the electric-field-based cleaning mechanism, I developed a numerical simulation 
        that models the trajectories of sand particles under electrostatic and gravitational forces. The simulation employs a 
        time-stepping method grounded in classical mechanics and implemented using first-order Euler integration.
      </p>

      <ul>
        <li>Simulation time: <InlineMath>{"t \\in [0, 50]"}</InlineMath> seconds, divided into 1,000 uniform time steps</li>
        <li>Initial conditions: particle positions <InlineMath>{"\\vec{r}_0"}</InlineMath> uniformly sampled across the panel surface</li>
        <li>Velocity and position updates governed by Newton’s second law and computed using:
          <BlockMath>{"\\vec{v}_{i+1} = \\vec{v}_i + \\frac{\\vec{F}_{net}}{m} \\Delta t"}</BlockMath>
          <BlockMath>{"\\vec{r}_{i+1} = \\vec{r}_i + \\vec{v}_i \\Delta t"}</BlockMath>
        </li>
        <li>Termination criteria: particle removed from simulation once <InlineMath>{"z \\geq 0.1"}</InlineMath> m (height of the charged bar)</li>
      </ul>

      <p>
        This approach enabled high-resolution temporal tracking of particle dynamics under nonlinear force fields. 
        The results confirmed that appropriately tuned electric fields can generate sufficient lift to detach 
        sand grains from the panel surface while simultaneously displacing them laterally. Animated simulations 
        of particle paths offered visual confirmation of the cleaning effect and allowed iterative refinement 
        of field parameters and geometry.
      </p>

      <h2>Key Code Implementation</h2>
      <p>
        The computational model was implemented in MATLAB, with several core functions handling different aspects of the simulation.
        The following code snippets highlight some of the most important components:
      </p>

      <h3>System Parameter Definition</h3>
      <p>
        The model begins by establishing the physical parameters of the system, ensuring realistic simulation conditions:
      </p>

      <pre className="code-block">
{`b = 1.7; % Panel width in meters
h = 1; % Panel height in meters
L = 0.5*sqrt(b^2+h^2) % Optimal bar length for coverage
m = 6.3e-5; % Mass of sand grain in grams
qG = 0.5e-12; % Electrical charge of sand grain in Coulombs
g = 9.8; % Gravitational constant in m/s²
numGrains = 10; % Number of sand particles to simulate`}
      </pre>

      <h3>Electric Field Calculation</h3>
      <p>
        The electric field computation function calculates the field at any point in 3D space based on the distribution of charges:
      </p>

      <pre className="code-block">
{`function [Ex, Ey, Ez] = Efield3(xq, yq, zq, xp, yp, zp, Q)
    % Computes electric field components at point (xp,yp,zp)
    % from charges Q located at points (xq,yq,zq)
    k = 8.99e9; % Coulomb constant
    
    % Calculate distance components
    rx = xp - xq;
    ry = yp - yq;
    rz = zp - zq;
    
    % Calculate distance magnitude
    r = sqrt(rx.^2 + ry.^2 + rz.^2);
    
    % Calculate electric field components
    Ex = k * Q * rx ./ r.^3;
    Ey = k * Q * ry ./ r.^3;
    Ez = k * Q * rz ./ r.^3;
end`}
      </pre>

      <h3>Particle Motion Simulation</h3>
      <p>
        The following code implements the numerical integration of particle motion using Euler's method:
      </p>

      <pre className="code-block">
{`for i=1:numGrains
    k = 1;
    while rz(k, i) <= barZ - 0.001
        % Calculate electric field at particle position
        [Efx, Efy, Efz] = Efield3(ringsX, ringsY, ringsZ, rx(i), ry(i), rz(i), -Qbar);
        
        % Add gravitation force
        Fz = Efz - m*g;
        
        % Update velocity and position
        vz(k + 1, i) = vz(k, i) + h*(qG*sum(Fz)/m);
        rz(k + 1, i) = rz(k, i) + h*vz(k, i);
        
        k = k + 1;
    end
end`}
      </pre>

      <h2>Practical Applications and Benefits</h2>
      <p>
        This electric field-based cleaning solution offers numerous advantages over conventional methods for desert solar installations:
      </p>

      <ul>
        <li><strong>Water Conservation:</strong> Reduces water usage compared to traditional cleaning methods</li>
        <li><strong>Automatic Operation:</strong> Can be automated to activate based on efficiency metrics or scheduled maintenance</li>
        <li><strong>Non-contact Cleaning:</strong> Eliminates physical abrasion that could damage delicate panel surfaces</li>
        <li><strong>Energy Efficiency:</strong> Requires minimal power input compared to mechanical cleaning systems</li>
        <li><strong>Reduced Maintenance Costs:</strong> Lowers operational expenses and human labor requirements</li>
        <li><strong>Scalability:</strong> Easily adaptable to various panel configurations and installation sizes</li>
      </ul>

      <p>
        Implementation of this system could significantly enhance the viability of large-scale solar installations in desert regions 
        by addressing one of the most challenging operational issues: maintaining clean panel surfaces in sandy environments without 
        consuming precious water resources.
      </p>


      <h2>Reflection and Learnings</h2>
      <p>
        This project pushed me to integrate knowledge from multiple domains: electrostatics, computational physics, 
        renewable energy systems, and environmental engineering. Working at the intersection of these fields revealed 
        how innovative solutions often emerge when applying principles from one domain to challenges in another.
      </p>

      <p>
        The most significant insight was recognizing that desert environments, while challenging for solar power generation 
        in many ways, also offer unique opportunities. The naturally dry conditions that exacerbate dust accumulation also 
        enhance electrostatic effects, making electric field-based cleaning particularly effective in precisely the environments 
        where it's most needed.
      </p>

      <p>
        From a technical perspective, this computational physics course project deepened my expertise in computational modeling, particularly in creating 
        physically accurate simulations that can guide practical engineering solutions. The challenge of optimizing between 
        theoretical accuracy and computational efficiency taught me valuable lessons about model design that I've since applied 
        to other projects.
      </p>

      <p>
        As renewable energy installations continue to expand into challenging environments worldwide, innovations that address 
        practical operational issues like cleaning and maintenance will play a crucial role in making these systems truly sustainable. 
        I believe this electric field approach represents a promising direction that merges elegant physics with practical engineering 
        to solve a real-world challenge in renewable energy deployment.
      </p>
    </PageTemplate>
  );
}