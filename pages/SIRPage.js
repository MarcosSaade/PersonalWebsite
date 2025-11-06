import React from 'react';
import PageTemplate from '../components/PageTemplate';
import sirImage from '../images/sir.jpeg';
import simulationVid from '../images/sir/sir_simulation.mp4';
import peakImg from '../images/sir/peak-condition.png';
import betaComparisonImg from '../images/sir/beta-comparison.png';
import vaccinationImg from '../images/sir/vaccination-effect.png';
import pygameSimImg from '../images/sir/pygame-simulation.png';

import { BlockMath, InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';

export default function SIRPage() {
  return (
    <PageTemplate title="Modeling Epidemics: The SIR Dynamic System" image={sirImage}>
      <p>
        Infectious diseases spread in patterns that, while chaotic on the surface, can often be captured using surprisingly elegant mathematics. In this project, my team and I explored several variations of the classic <i>SIR model</i>, which categorizes a population into <i>Susceptible</i> (S), <i>Infected</i> (I), and <i>Recovered</i> (R) compartments. Our goal was to analyze how these quantities evolve over time under different assumptions—including population growth, vaccination scenarios, and spatial dynamics in urban environments.
      </p>

      <p>
        This project was part of a course on Systems Modeling with Differential Equations, providing an opportunity to bridge theoretical mathematical concepts with practical applications in epidemiology and public health.
      </p>

      <h2>The Classical SIR Model: Theory and Mathematical Analysis</h2>
      <p>
        The standard SIR model divides a population into three distinct compartments:
      </p>
      <ul>
        <li><strong>Susceptible (S)</strong>: Individuals who can contract the disease</li>
        <li><strong>Infected (I)</strong>: Individuals who have the disease and can transmit it</li>
        <li><strong>Recovered (R)</strong>: Individuals who have recovered and developed immunity</li>
      </ul>
      
      <p>
        The dynamics of disease transmission are captured by three coupled differential equations:
      </p>
      <BlockMath math={String.raw`\begin{aligned}
        \frac{dS}{dt} &= -\beta \frac{SI}{N} \\
        \frac{dI}{dt} &= \beta \frac{SI}{N} - \gamma I \\
        \frac{dR}{dt} &= \gamma I
      \end{aligned}`}/>
      
      <p>
        These equations represent a deterministic model where:
      </p>
      <ul>
        <li><InlineMath math="\beta" /> is the effective contact rate (the product of contact rate and transmission probability)</li>
        <li><InlineMath math="\gamma" /> is the recovery rate (the inverse of the infectious period)</li>
        <li><InlineMath math="N = S + I + R" /> is the total population (initially assumed constant)</li>
      </ul>
      
      <p>
        The term <InlineMath math="\beta \frac{SI}{N}" /> represents the force of infection—the rate at which susceptible individuals become infected. This formulation assumes homogeneous mixing, where any infected individual can potentially infect any susceptible individual with equal probability.
      </p>

      <h2>Analyzing Disease Dynamics: Peak Infection and Epidemic Threshold</h2>
      <p>
        A critical question in epidemiology is determining when an infection will reach its peak. The infection count <InlineMath math="I(t)" /> reaches its maximum when:
      </p>
      <BlockMath math={String.raw`\frac{dI}{dt} = 0 \Rightarrow \beta \frac{S}{N} = \gamma \Rightarrow S = \frac{\gamma N}{\beta}`}/>
      
      <p>
        This elegant result gives us the precise number of susceptible individuals at the infection peak. For example, with <InlineMath math="\beta = 1" />, <InlineMath math="\gamma = 0.1" />, and <InlineMath math="N = 10^6" />, infections peak when <InlineMath math="S \approx 100{,}000" /> individuals remain susceptible. This occurs because at this precise moment, new infections exactly balance recoveries.
      </p>
      <img src={peakImg} alt="Graph showing peak of infection curve at critical S = γN/β" className="page-image" />

      <h2>The Basic Reproduction Number: When Does an Epidemic Occur?</h2>
      <p>
        The most fundamental parameter in epidemic modeling is the basic reproduction number:
      </p>
      <BlockMath math={String.raw`R_0 = \frac{\beta}{\gamma}`}/>
      
      <p>
        <InlineMath math="R_0" /> represents the expected number of secondary infections produced by a single infected individual in a completely susceptible population. This value determines whether an epidemic will occur:
      </p>
      <ul>
        <li>If <InlineMath math="R_0 > 1" />, each infected person infects more than one new person on average, leading to exponential growth and an epidemic</li>
        <li>If <InlineMath math="R_0 \leq 1" />, the disease cannot sustain transmission and will gradually fade away</li>
      </ul>
      
      <p>
        We conducted numerical simulations with varying values of <InlineMath math="\beta" /> while keeping <InlineMath math="\gamma" /> constant to observe how <InlineMath math="R_0" /> affects the epidemic curve:
      </p>
      <img src={betaComparisonImg} alt="Comparison of SIR curves for different beta values" className="page-image" />
      
      <p>
        As shown above, higher values of <InlineMath math="\beta" /> (and consequently higher <InlineMath math="R_0" />) result in faster and more widespread outbreaks, with earlier peaks and larger portions of the population becoming infected. This mathematical relationship helps explain why highly contagious diseases like measles (<InlineMath math="R_0 \approx 12-18" />) spread more rapidly and widely than diseases like seasonal influenza (<InlineMath math="R_0 \approx 1.3" />).
      </p>

      <h2>The Early Growth Phase</h2>
      <p>
        During the early phase of an epidemic, when <InlineMath math="S \approx N" /> (most of the population is still susceptible), we can derive approximate analytical solutions. One interesting question is: how long does it take for the number of susceptible individuals to drop by half?
      </p>
      
      <p>
        Starting from the differential equation for S:
      </p>
      <BlockMath math={String.raw`\frac{dS}{dt} = -\beta \frac{SI}{N}`}/>
      
      <p>
        If we assume <InlineMath math="I \approx I_0" /> (approximately constant) during the early phase and integrate, we get:
      </p>
      <BlockMath math={String.raw`\int_{S_0}^{S(t)} \frac{dS}{S} = -\frac{\beta I_0}{N} \int_0^t dt`}/>
      
      <p>
        Which simplifies to:
      </p>
      <BlockMath math={String.raw`\ln\left(\frac{S(t)}{S_0}\right) = -\frac{\beta I_0 t}{N}`}/>
      
      <p>
        To find the time when <InlineMath math="S(t) = S_0/2" />, we solve:
      </p>
      <BlockMath math={String.raw`t = \frac{N}{\beta I_0} \ln\left(\frac{S_0}{S_0/2}\right) = \frac{N \ln 2}{\beta I_0}`}/>
      
      <p>
        This formula reveals that the time to halve the susceptible population is inversely proportional to <InlineMath math="\beta" /> and the initial number of infected individuals <InlineMath math="I_0" />. Our numerical simulations confirmed this relationship.
      </p>

      <h2>Vaccination and Herd Immunity</h2>
      <p>
        Vaccination is one of the most effective interventions against infectious diseases. We extended our analysis to understand how vaccination affects epidemic dynamics. When a fraction <InlineMath math="v" /> of the population is vaccinated before an outbreak begins, the effective initial susceptible population becomes:
      </p>
      <BlockMath math={String.raw`S_0' = S_0(1-v)`}/>
      
      <p>
        For an epidemic to be prevented entirely (i.e., for <InlineMath math="\frac{dI}{dt} < 0" /> from the very beginning), we need:
      </p>
      <BlockMath math={String.raw`\frac{S_0'}{N} < \frac{\gamma}{\beta} = \frac{1}{R_0}`}/>
      
      <p>
        This inequality leads to the critical vaccination threshold:
      </p>
      <BlockMath math={String.raw`v > 1 - \frac{1}{R_0}`}/>
      
      <p>
        For example, with <InlineMath math="R_0 = 4" />, at least 75% of the population must be vaccinated to prevent an outbreak through herd immunity. Our simulations confirmed this theoretical prediction:
      </p>
      <img src={vaccinationImg} alt="Vaccination simulation showing herd immunity effect" className="page-image" />
      
      <p>
        This mathematical relationship explains why diseases with higher <InlineMath math="R_0" /> values require higher vaccination coverage to achieve herd immunity.
      </p>

      <h2>Dynamic Population Model</h2>
      <p>
        Real populations aren't static—they experience births, deaths, and migration. We incorporated demographic dynamics into our model by modifying the equations:
      </p>
      <BlockMath math={String.raw`
        \begin{aligned}
        \frac{dS}{dt} &= -\beta \frac{SI}{N} + bN - \mu S \\
        \frac{dI}{dt} &= \beta \frac{SI}{N} - \gamma I - \mu I \\
        \frac{dR}{dt} &= \gamma I - \mu R
        \end{aligned}`}/>
      
      <p>
        Where:
      </p>
      <ul>
        <li><InlineMath math="b" /> is the birth rate</li>
        <li><InlineMath math="\mu" /> is the natural death rate (assumed equal across all groups)</li>
      </ul>
      
      <p>
        In a balanced demographic scenario where <InlineMath math="b = \mu" />, the total population remains constant, but the model exhibits different long-term behavior. Instead of the disease dying out completely, it can reach an endemic equilibrium where new infections balance recoveries and demographic changes.
      </p>
      
      <p>
        The addition of vital dynamics creates the possibility of endemic diseases—those that persist indefinitely in a population—which is impossible in the basic SIR model where epidemics always eventually burn out.
      </p>

      <h2>Agent-Based Simulation: Spatial Dynamics in Urban Environments</h2>
      <p>
        While differential equation models provide valuable insights into aggregate disease dynamics, they assume homogeneous mixing of the population. To explore how spatial dynamics and urban centers affect disease transmission, we developed an agent-based simulation using Python's <code>pygame</code> library.
      </p>
      
      <p>
        In this simulation, individuals are represented as moving particles in a two-dimensional space, with their state (susceptible, infected, or recovered) indicated by different colors (susceptible in blue, infected in red, and recovered in gray). Key aspects of our simulation include:
      </p>
      <ul>
        <li>Agents move with varying velocities and some persistence in their direction</li>
        <li>There's an attraction toward the center, simulating an urban center like a grocery store or school</li>
        <li>Infection occurs when infected agents come within a certain radius of susceptible ones</li>
        <li>Recovery happens probabilistically based on a recovery rate parameter</li>
      </ul>

      <video autoPlay loop muted className="page-video">
        <source src={simulationVid} type="video/mp4" />
        Your browser does not support the video tag.
      </video>

      <p>
        Unlike our analytical SIR model, this simulation doesn't directly implement the differential equations. Instead, it simulates behavior through sedicted by the mathematical SIR model:
      </p>

      <img src={pygameSimImg} alt="Results from agent-based pygame simulation showing SIR dynamics" className="page-image" />
      
      <p>
        This simulation revealed several insights about spatial effects on disease spread:
      </p>
      <ul>
        <li>Urban centers accelerate early disease spread due to higher contact rates</li>
        <li>Movement patterns significantly influence the time course of an epidemic</li>
      </ul>
      
      <p>
        The simulation code implements these dynamics through classes representing individuals and their environment:
      </p>


      <pre><code className='language-python'>{`
        class Person:
            def __init__(self, x, y):
                # State: 0 - Susceptible, 1 - Infected, 2 - Recovered
                self.pos = pygame.math.Vector2(x, y)
                self.last_velocity = pygame.math.Vector2(0, 0)
                self.state = 0
                
            # Methods for movement and interaction...
      `}
      </code></pre>
      
      <p>
        This approach provides a complementary perspective to our analytical models, highlighting how individual-level behaviors translate to population-level outcomes.
      </p>

      <h2>Implications and Applications</h2>
      <p>
        The mathematical and computational models we developed offer several important insights for public health policy and infectious disease management:
      </p>
      
      <ol>
        <li>
          <strong>Identifying Critical Thresholds:</strong> Our models precisely quantify when epidemics will occur (<InlineMath math="R_0 > 1" />), when they'll peak (<InlineMath math="S = \gamma N / \beta" />), and what vaccination coverage is needed for herd immunity (<InlineMath math="v > 1 - 1/R_0" />).
        </li>
        
        <li>
          <strong>Intervention Timing:</strong> The analysis shows that the effectiveness of interventions depends critically on timing. Early action before the epidemic peak has far greater impact than later interventions.
        </li>
        
        <li>
          <strong>Urban Planning Considerations:</strong> Our agent-based simulation suggests that urban density patterns influence disease spread, with implications for transportation planning and public gathering spaces during outbreaks.
        </li>
        
        <li>
          <strong>Demographics Matter:</strong> The dynamic population model demonstrates how birth rates and population structure can transform a one-time epidemic into an endemic disease, which is important for understanding long-term disease management strategies.
        </li>
      </ol>
      
      <p>
        These models also have limitations worth acknowledging. They assume random mixing (or simplistic spatial structure), homogeneous populations, and deterministic processes. Real-world epidemics involve heterogeneous contact networks, variable susceptibility, stochastic events, and behavioral adaptations—all factors that could be incorporated in future work.
      </p>

      <h2>Final Thoughts</h2>
      <p>
        This project demonstrated the power of mathematical modeling in understanding complex biological and social phenomena. The SIR framework provides a rigorous yet intuitive lens through which to view one of humanity's oldest adversaries: infectious disease.
      </p>
      
      <p>
        Perhaps most striking is how relatively simple mathematical formulations can generate rich, complex behaviors that match real-world observations. The harmony between our analytical solutions, numerical simulations, and agent-based models reinforces the fundamental principles governing epidemic spread across different scales and implementations.
      </p>
      
      <p>
        As we face new infectious disease challenges in our increasingly connected world, these quantitative approaches become ever more essential—both for theoretical understanding and for informing practical public health responses.
      </p>

      <p>
        As always, you can check out the code for this project on <a href="https://github.com/MarcosSaade/SIR-Differential-Visualizer" target="_blank" rel="noopener noreferrer">GitHub</a>.
      </p>
    </PageTemplate>
  );
}