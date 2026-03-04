import React from 'react';
import 'katex/dist/katex.min.css';
import { BlockMath, InlineMath } from 'react-katex';

import PageTemplate from '../components/PageTemplate';
import fleetImage from '../images/fleet.png';

// Import all fleet optimization plots
import costoVsEspera from '../images/fleet-optimization/01_costo_vs_espera.png';
import comparacionConfiguraciones from '../images/fleet-optimization/02_comparacion_configuraciones.png';
import timelineOperaciones from '../images/fleet-optimization/03_timeline_operaciones.png';
import ocupacionVehiculos from '../images/fleet-optimization/04_ocupacion_vehiculos.png';
import evolucionOcupacion from '../images/fleet-optimization/05_evolucion_ocupacion_horaria.png';
import topConfiguraciones from '../images/fleet-optimization/06_top_configuraciones.png';
import variabilidadMontecarlo from '../images/fleet-optimization/07_variabilidad_montecarlo.png';

export default function FleetOptimizationPage() {
  return (
    <PageTemplate title="Optimizing Corporate Transportation Fleet: A Monte Carlo Simulation Approach" image={fleetImage}>
      {/* Introduction */}
      <p>
        How do you design a transportation system that moves hundreds of students and staff daily while minimizing costs and keeping wait times under control? This was the challenge posed by my university's transport service, which operates busses and vans for transporting students and staff across campus. With multiple vehicle types, varying schedules, and stochastic passenger arrivals, finding the optimal fleet configuration required more than intuition—it demanded rigorous simulation and optimization.
      </p>
      <p>
        This project emerged from a real operational challenge: design a fleet of buses and vans to transport students and staff from 6:00 AM to 10:00 PM, ensuring average wait times stay below 10 minutes while minimizing operational costs. The twist? Vehicle availability was constrained by schedules, capacities, and quantities. Some buses only operated during specific hours, vans had different capacity limits, and each vehicle type came with its own cost structure.
      </p>
      <p>
        Rather than relying on back-of-the-envelope calculations or overly simplified deterministic models, I built a discrete-event simulation framework with Monte Carlo analysis to evaluate 48 valid fleet configurations under realistic conditions. Each configuration was tested across 50 simulations with stochastic passenger arrivals, generating over 2,400 simulation runs to identify robust, cost-effective solutions.
      </p>
      <p>
        The result was a comprehensive optimization study that balanced service quality with cost efficiency, providing actionable recommendations backed by statistical rigor. Along the way, I learned valuable lessons about queueing theory, simulation design, and the importance of modeling uncertainty in operations research.
      </p>

      <h2>The Problem: Balancing Service Quality and Cost</h2>
      <p>
        The transportation challenge had clear constraints and objectives:
      </p>

      <h3>The Fleet</h3>
      <p>
        Four vehicle types were available, each with different capacities, operating hours, and costs:
      </p>

      <table style={{margin: '20px 0', borderCollapse: 'collapse', width: '100%'}}>
        <thead>
          <tr style={{borderBottom: '2px solid #333'}}>
            <th style={{padding: '10px', textAlign: 'left'}}>Vehicle Type</th>
            <th style={{padding: '10px', textAlign: 'right'}}>Capacity</th>
            <th style={{padding: '10px', textAlign: 'center'}}>Operating Hours</th>
            <th style={{padding: '10px', textAlign: 'right'}}>Cost/Hour</th>
            <th style={{padding: '10px', textAlign: 'right'}}>Available</th>
            <th style={{padding: '10px', textAlign: 'center'}}>Priority</th>
          </tr>
        </thead>
        <tbody>
          <tr style={{borderBottom: '1px solid #ddd'}}>
            <td style={{padding: '10px'}}>Bus A</td>
            <td style={{padding: '10px', textAlign: 'right'}}>37</td>
            <td style={{padding: '10px', textAlign: 'center'}}>6:00 AM – 5:00 PM</td>
            <td style={{padding: '10px', textAlign: 'right'}}>$42.60</td>
            <td style={{padding: '10px', textAlign: 'right'}}>1</td>
            <td style={{padding: '10px', textAlign: 'center'}}>1 (Highest)</td>
          </tr>
          <tr style={{borderBottom: '1px solid #ddd'}}>
            <td style={{padding: '10px'}}>Bus B</td>
            <td style={{padding: '10px', textAlign: 'right'}}>31</td>
            <td style={{padding: '10px', textAlign: 'center'}}>8:00 AM – 2:00 PM</td>
            <td style={{padding: '10px', textAlign: 'right'}}>$42.60</td>
            <td style={{padding: '10px', textAlign: 'right'}}>2</td>
            <td style={{padding: '10px', textAlign: 'center'}}>3</td>
          </tr>
          <tr style={{borderBottom: '1px solid #ddd'}}>
            <td style={{padding: '10px'}}>Van A</td>
            <td style={{padding: '10px', textAlign: 'right'}}>19</td>
            <td style={{padding: '10px', textAlign: 'center'}}>6:00 AM – 8:00 PM</td>
            <td style={{padding: '10px', textAlign: 'right'}}>$24.80</td>
            <td style={{padding: '10px', textAlign: 'right'}}>1</td>
            <td style={{padding: '10px', textAlign: 'center'}}>4</td>
          </tr>
          <tr style={{borderBottom: '1px solid #ddd'}}>
            <td style={{padding: '10px'}}>Van B</td>
            <td style={{padding: '10px', textAlign: 'right'}}>13</td>
            <td style={{padding: '10px', textAlign: 'center'}}>7:00 AM – 10:00 PM</td>
            <td style={{padding: '10px', textAlign: 'right'}}>$24.80</td>
            <td style={{padding: '10px', textAlign: 'right'}}>3</td>
            <td style={{padding: '10px', textAlign: 'center'}}>2</td>
          </tr>
        </tbody>
      </table>

      <p>
        The priority ordering reflected operational preferences: Bus A was preferred for its large capacity and full-day coverage, followed by the flexible Van B units that could operate into the evening. Bus B had limited hours, and Van A was the fallback option.
      </p>

      <h3>The Constraints</h3>
      <p>
        The system had to satisfy several hard requirements:
      </p>
      <ul>
        <li><strong>Complete Coverage:</strong> At least one vehicle must be operating at all times from 6:00 AM to 10:00 PM (960 minutes of operation)</li>
        <li><strong>Service Quality:</strong> Average passenger wait time must remain below 10 minutes</li>
        <li><strong>Vehicle Limits:</strong> Cannot exceed the maximum available quantity for each vehicle type</li>
      </ul>

      <h3>The Objectives</h3>
      <p>
        Among configurations that satisfied the constraints, we needed to:
      </p>
      <ol>
        <li><strong>Minimize operational costs</strong> (primary objective)</li>
        <li><strong>Minimize wait times</strong> (secondary objective for premium service)</li>
        <li><strong>Identify balanced options</strong> that traded off cost and service quality optimally</li>
      </ol>

      <p>
        This is a classic multi-objective optimization problem under uncertainty. The stochastic nature of passenger arrivals meant that no single deterministic analysis would suffice—we needed to evaluate configurations across many scenarios to ensure robustness.
      </p>

      <h2>The Data: Modeling Realistic Demand</h2>
      <p>
        Accurate modeling of demand patterns was critical. The company provided historical data on vehicle arrival times—essentially, how often shuttles completed their routes and returned to pick up more passengers. This data was segmented into four time periods reflecting different demand patterns throughout the day:
      </p>

      <ul>
        <li><strong>6:00–9:00 AM (Morning Rush):</strong> High-frequency arrivals as students and staff arrive to campus</li>
        <li><strong>9:00 AM–2:00 PM (Mid-Day):</strong> Moderate activity with lunch and inter-facility transfers</li>
        <li><strong>2:00–6:00 PM (Afternoon):</strong> Increased activity as students arrive for afternoon classes</li>
        <li><strong>6:00–10:00 PM (Evening):</strong> Lower demand but still requiring coverage for late shifts</li>
      </ul>

      <p>
        The evening period (6:00–10:00 PM) lacked historical data. However, transportation managers indicated that evening demand was approximately 50% of the afternoon period. I scaled the 2:00–6:00 PM data accordingly to generate a reasonable estimate for evening operations.
      </p>

      <h3>Passenger Arrival Process</h3>
      <p>
        Passenger arrivals followed a non-homogeneous Poisson process—a standard model for arrivals in queueing theory where the rate varies over time. I modeled this with time-varying arrival rates (λ) based on observed patterns:
      </p>

      <BlockMath math="\lambda(t) = \begin{cases} 3.0 & \text{6:00-8:00 AM (high morning demand)} \\ 2.0 & \text{8:00-10:00 AM (tapering morning)} \\ 3.0 & \text{10:00 AM-1:00 PM (midday peak)} \\ 1.6 & \text{1:00-5:00 PM (afternoon, highest rate)} \\ 2.6 & \text{5:00-10:00 PM (evening)} \end{cases}" />

      <p>
        These rate parameters (λ) represent the average time between passenger arrivals in minutes. Lower values indicate more frequent arrivals. The afternoon period (1:00–5:00 PM) had the highest arrival rate, creating the day's critical bottleneck.
      </p>

      <p>
        For each simulation, passenger arrival times were generated by sampling from exponential distributions with these time-varying rates:
      </p>

      <BlockMath math="T_{\text{next}} = T_{\text{current}} + \text{Exp}(\lambda(T_{\text{current}}))" />

      <p>
        This stochastic modeling ensured that each simulation represented a different realization of demand, capturing the natural variability in passenger behavior.
      </p>

      <h2>The Simulation Framework: Discrete-Event Modeling</h2>
      <p>
        I built a discrete-event simulation engine that tracked the state of the system minute-by-minute over a 16-hour operating day. The simulation had three main components:
      </p>

      <h3>1. Fleet Configuration</h3>
      <p>
        Each configuration specified how many vehicles of each type to deploy. For example:
      </p>
      <ul>
        <li>Configuration A: 1 Bus A, 1 Van A, 1 Van B</li>
        <li>Configuration B: 1 Bus A, 2 Van B</li>
        <li>Configuration C: 1 Van A, 2 Van B</li>
      </ul>

      <p>
        The simulation generated all valid combinations (respecting vehicle availability and coverage requirements), resulting in <strong>48 feasible configurations</strong>.
      </p>

      <h3>2. Vehicle Trip Generation</h3>
      <p>
        For each vehicle in the fleet, trips were generated based on empirical interarrival times from the historical data. Trips represented when a vehicle arrived at the pickup location, ready to board passengers.
      </p>
      <p>
        The first trip for each vehicle occurred exactly at its start time. Subsequent trips were sampled from the historical distribution for the appropriate time period:
      </p>

      <BlockMath math="T_{\text{trip}}^{(n+1)} = T_{\text{trip}}^{(n)} + \Delta t" />

      <p>
        where <InlineMath math="\Delta t" /> is sampled from the empirical distribution of interarrival times for the current time period.
      </p>

      <p>
        All trips from all vehicles were pooled and sorted chronologically, creating a unified timeline of vehicle arrivals throughout the day.
      </p>

      <h3>3. Passenger Queueing and Boarding</h3>
      <p>
        Passengers arriving at the queue waited until the next available vehicle. When a vehicle arrived:
      </p>
      <ol>
        <li>Count passengers currently waiting in the queue</li>
        <li>Board passengers up to the vehicle's capacity</li>
        <li>Record wait times for boarded passengers (arrival time to boarding time)</li>
        <li>Update queue length statistics</li>
      </ol>

      <p>
        The simulation tracked detailed metrics for every trip:
      </p>
      <ul>
        <li>Queue length when the vehicle arrived</li>
        <li>Number of passengers boarded</li>
        <li>Vehicle occupancy percentage (boarded / capacity)</li>
        <li>Individual passenger wait times</li>
      </ul>

      <p>
        At the end of the day, any remaining passengers in the queue were assigned to the final vehicle, ensuring all passengers were eventually served.
      </p>

      <h3>Output Metrics</h3>
      <p>
        For each simulation run, the following performance metrics were calculated:
      </p>
      <ul>
        <li><strong>Average Wait Time:</strong> Mean time passengers spent waiting (minutes)</li>
        <li><strong>Median Wait Time:</strong> Median wait time (more robust to outliers)</li>
        <li><strong>Maximum Wait Time:</strong> Longest individual wait experienced</li>
        <li><strong>Average Queue Length:</strong> Mean number of passengers waiting when vehicles arrived</li>
        <li><strong>Maximum Queue Length:</strong> Peak queue size during the day</li>
        <li><strong>Total Trips:</strong> Number of vehicle trips completed</li>
        <li><strong>Passengers Served:</strong> Total number of passengers transported</li>
        <li><strong>Vehicle Occupancy:</strong> Percentage of vehicle capacity utilized per trip</li>
        <li><strong>Total Cost:</strong> Daily operational cost based on vehicle hours and rates</li>
      </ul>

      <h2>Monte Carlo Analysis: Quantifying Uncertainty</h2>
      <p>
        Running a single simulation for each configuration would give us point estimates, but operations research demands understanding variability. A configuration that performs well on average but has high variance in wait times is riskier than one with consistent performance.
      </p>
      <p>
        I implemented Monte Carlo analysis by running <strong>50 independent simulations</strong> for each of the 48 configurations, totaling <strong>2,400 simulation runs</strong>. Each simulation used different random seeds for:
      </p>
      <ul>
        <li>Passenger arrival times (stochastic Poisson process)</li>
        <li>Vehicle trip generation (sampling from empirical distributions)</li>
      </ul>

      <p>
        For each configuration, I calculated:
      </p>
      <ul>
        <li><strong>Mean metrics</strong> (average wait time, cost, etc.) across all 50 runs</li>
        <li><strong>Standard deviations</strong> (variability of performance)</li>
        <li><strong>Min/Max values</strong> (best and worst cases)</li>
      </ul>

      <p>
        This Monte Carlo approach provided several critical insights:
      </p>
      <ol>
        <li><strong>Robustness:</strong> Configurations with low standard deviation in wait times were more reliable</li>
        <li><strong>Risk Assessment:</strong> Maximum wait times revealed worst-case scenarios</li>
        <li><strong>Confidence Intervals:</strong> Standard errors quantified uncertainty in recommendations</li>
      </ol>

      <p>
        For example, one configuration (1 Van A + 1 Van B) had:
      </p>
      <ul>
        <li>Mean wait time: 10.42 minutes (± 0.74 minutes std. dev.)</li>
        <li>Range: [8.71, 12.40] minutes across 50 simulations</li>
      </ul>

      <p>
        This configuration narrowly exceeded the 10-minute requirement on average, it failed the constraint. More importantly, its high variability meant that in some scenarios, wait times exceeded 12 minutes—unacceptable for service quality.
      </p>

      <div style={{margin: '40px 0', textAlign: 'center'}}>
        <img src={variabilidadMontecarlo} alt="Monte Carlo Variability Analysis" style={{maxWidth: '100%', height: 'auto', border: '1px solid #ddd', borderRadius: '4px'}} />
        <p style={{fontSize: '0.9em', color: '#666', marginTop: '10px', fontStyle: 'italic'}}>
          Monte Carlo variability analysis showing error bars for top configurations. Tighter error bars indicate more consistent, predictable performance across different demand scenarios.
        </p>
      </div>

      <h2>Results: Three Recommended Configurations</h2>
      <p>
        After filtering out configurations that failed to meet the 10-minute wait time requirement, <strong>27 valid configurations</strong> remained. From these, I identified three optimal solutions:
      </p>

      <h3>1. Economic Option: Minimize Costs</h3>
      <p>
        <strong>Fleet:</strong> 1 Van A + 2 Van B
      </p>
      <ul>
        <li><strong>Daily Cost:</strong> $1,091.20</li>
        <li><strong>Monthly Cost (30 days):</strong> $32,736</li>
        <li><strong>Annual Cost (365 days):</strong> $398,288</li>
        <li><strong>Average Wait Time:</strong> 7.90 minutes (± 0.59)</li>
        <li><strong>Maximum Wait Time:</strong> 32.48 minutes</li>
        <li><strong>Average Queue Length:</strong> 4.33 passengers</li>
        <li><strong>Total Daily Trips:</strong> ~155 trips</li>
        <li><strong>Average Occupancy:</strong> ~73%</li>
      </ul>

      <p>
        This configuration used only smaller vehicles (no buses), leveraging their lower hourly costs. The three vehicles provided sufficient coverage across all hours, with Van A operating during the day and both Van B units extending into evening hours. Wait times remained comfortably below the 10-minute threshold with low variability.
      </p>

      <p>
        <strong>Trade-off:</strong> While cost-effective, this configuration had higher queue peaks during rush hours and longer maximum wait times compared to premium options.
      </p>

      <h3>2. Premium Option: Minimize Wait Times</h3>
      <p>
        <strong>Fleet:</strong> 1 Bus A + 1 Van A + 3 Van B
      </p>
      <ul>
        <li><strong>Daily Cost:</strong> $1,584.60</li>
        <li><strong>Monthly Cost (30 days):</strong> $47,538</li>
        <li><strong>Annual Cost (365 days):</strong> $578,379</li>
        <li><strong>Average Wait Time:</strong> 6.66 minutes (± 0.47)</li>
        <li><strong>Maximum Wait Time:</strong> 32.46 minutes</li>
        <li><strong>Average Queue Length:</strong> 3.38 passengers</li>
        <li><strong>Total Daily Trips:</strong> ~195 trips</li>
        <li><strong>Average Occupancy:</strong> ~64%</li>
      </ul>

      <p>
        This configuration deployed the maximum fleet: one bus for high-capacity morning coverage and four vans ensuring continuous service throughout the day. The increased frequency significantly reduced average wait times and queue lengths.
      </p>

      <p>
        <strong>Trade-off:</strong> Daily costs were 45% higher than the economic option ($493 more per day, or $180,091 annually). However, average wait times dropped by 1.23 minutes (16% reduction), and queues were 22% shorter.
      </p>

      <h3>3. Balanced Option: Optimize Cost-Benefit Trade-off</h3>
      <p>
        <strong>Fleet:</strong> 1 Bus A + 1 Van A + 1 Van B
      </p>
      <ul>
        <li><strong>Daily Cost:</strong> $1,187.80</li>
        <li><strong>Monthly Cost (30 days):</strong> $35,634</li>
        <li><strong>Annual Cost (365 days):</strong> $433,547</li>
        <li><strong>Average Wait Time:</strong> 8.23 minutes (± 0.71)</li>
        <li><strong>Maximum Wait Time:</strong> 30.73 minutes</li>
        <li><strong>Average Queue Length:</strong> 5.08 passengers</li>
        <li><strong>Total Daily Trips:</strong> ~165 trips</li>
        <li><strong>Average Occupancy:</strong> ~75%</li>
      </ul>

      <p>
        This middle-ground solution combined the high-capacity Bus A with two vans. It cost only $96.60 more per day than the economic option (9% increase, or $35,259 annually) while reducing wait times by 4% and maintaining service quality.
      </p>

      <p>
        <strong>Recommendation:</strong> The balanced option offers the best value proposition—marginally higher costs than the economic option but significantly better service quality than the premium option would justify.
      </p>

      <h3>Cost-Benefit Analysis</h3>
      <p>
        Comparing the three options reveals the marginal value of additional service quality:
      </p>

      <table style={{margin: '20px 0', borderCollapse: 'collapse', width: '100%'}}>
        <thead>
          <tr style={{borderBottom: '2px solid #333'}}>
            <th style={{padding: '10px', textAlign: 'left'}}>Comparison</th>
            <th style={{padding: '10px', textAlign: 'right'}}>Cost Difference</th>
            <th style={{padding: '10px', textAlign: 'right'}}>Wait Time Reduction</th>
            <th style={{padding: '10px', textAlign: 'right'}}>Cost per Minute Saved</th>
          </tr>
        </thead>
        <tbody>
          <tr style={{borderBottom: '1px solid #ddd'}}>
            <td style={{padding: '10px'}}>Balanced vs. Economic</td>
            <td style={{padding: '10px', textAlign: 'right'}}>+$35,259/year</td>
            <td style={{padding: '10px', textAlign: 'right'}}>-0.33 min (-4.2%)</td>
            <td style={{padding: '10px', textAlign: 'right'}}>$106,845/min</td>
          </tr>
          <tr style={{borderBottom: '1px solid #ddd'}}>
            <td style={{padding: '10px'}}>Premium vs. Balanced</td>
            <td style={{padding: '10px', textAlign: 'right'}}>+$144,832/year</td>
            <td style={{padding: '10px', textAlign: 'right'}}>-1.57 min (-19.1%)</td>
            <td style={{padding: '10px', textAlign: 'right'}}>$92,251/min</td>
          </tr>
          <tr style={{borderBottom: '1px solid #ddd', fontWeight: 'bold'}}>
            <td style={{padding: '10px'}}>Premium vs. Economic</td>
            <td style={{padding: '10px', textAlign: 'right'}}>+$180,091/year</td>
            <td style={{padding: '10px', textAlign: 'right'}}>-1.23 min (-15.6%)</td>
            <td style={{padding: '10px', textAlign: 'right'}}>$146,415/min</td>
          </tr>
        </tbody>
      </table>

      <p>
        The premium option costs $146,415 per year for each minute of wait time saved compared to the economic option—a steep price for marginal service improvements. The balanced option represents a more reasonable trade-off.
      </p>

      <div style={{margin: '40px 0', textAlign: 'center'}}>
        <img src={costoVsEspera} alt="Cost vs Wait Time Analysis" style={{maxWidth: '100%', height: 'auto', border: '1px solid #ddd', borderRadius: '4px'}} />
        <p style={{fontSize: '0.9em', color: '#666', marginTop: '10px', fontStyle: 'italic'}}>
          Scatter plot showing the trade-off between operational cost and average wait time. Valid configurations (below the 10-minute red line) form a Pareto frontier—cheaper options require accepting longer waits. The three recommended configurations are highlighted: Economic (blue star), Premium (green star), and the optimal region in between.
        </p>
      </div>

      <div style={{margin: '40px 0', textAlign: 'center'}}>
        <img src={comparacionConfiguraciones} alt="Configuration Comparison" style={{maxWidth: '100%', height: 'auto', border: '1px solid #ddd', borderRadius: '4px'}} />
        <p style={{fontSize: '0.9em', color: '#666', marginTop: '10px', fontStyle: 'italic'}}>
          Detailed comparison of the three recommended fleet configurations across multiple dimensions: total cost, wait times (average and maximum), queue lengths, fleet composition, and total daily trips. The balanced option emerges as the best value proposition.
        </p>
      </div>

      <div style={{margin: '40px 0', textAlign: 'center'}}>
        <img src={topConfiguraciones} alt="Top 10 Configurations Analysis" style={{maxWidth: '100%', height: 'auto', border: '1px solid #ddd', borderRadius: '4px'}} />
        <p style={{fontSize: '0.9em', color: '#666', marginTop: '10px', fontStyle: 'italic'}}>
          Analysis of the top 10 most economical configurations that meet service requirements. Charts show cost vs. wait time trade-offs, wait time rankings, fleet composition for top 5 options, and the relationship between cost and total daily trips.
        </p>
      </div>

      <h2>Operational Insights: Detailed Analysis</h2>
      <p>
        Beyond the top-level metrics, the simulation provided granular insights into system behavior throughout the day.
      </p>

      <h3>Time-of-Day Patterns</h3>
      <p>
        Analysis of the balanced configuration revealed distinct operational patterns across five time periods:
      </p>

      <table style={{margin: '20px 0', borderCollapse: 'collapse', width: '100%'}}>
        <thead>
          <tr style={{borderBottom: '2px solid #333'}}>
            <th style={{padding: '10px', textAlign: 'left'}}>Time Period</th>
            <th style={{padding: '10px', textAlign: 'right'}}>Trips</th>
            <th style={{padding: '10px', textAlign: 'right'}}>Passengers</th>
            <th style={{padding: '10px', textAlign: 'right'}}>Avg Queue</th>
            <th style={{padding: '10px', textAlign: 'right'}}>Max Queue</th>
            <th style={{padding: '10px', textAlign: 'right'}}>Occupancy</th>
          </tr>
        </thead>
        <tbody>
          <tr style={{borderBottom: '1px solid #ddd'}}>
            <td style={{padding: '10px'}}>6:00–8:00 AM (Morning)</td>
            <td style={{padding: '10px', textAlign: 'right'}}>38</td>
            <td style={{padding: '10px', textAlign: 'right'}}>112</td>
            <td style={{padding: '10px', textAlign: 'right'}}>3.8</td>
            <td style={{padding: '10px', textAlign: 'right'}}>12</td>
            <td style={{padding: '10px', textAlign: 'right'}}>68%</td>
          </tr>
          <tr style={{borderBottom: '1px solid #ddd'}}>
            <td style={{padding: '10px'}}>8:00–10:00 AM</td>
            <td style={{padding: '10px', textAlign: 'right'}}>22</td>
            <td style={{padding: '10px', textAlign: 'right'}}>58</td>
            <td style={{padding: '10px', textAlign: 'right'}}>3.2</td>
            <td style={{padding: '10px', textAlign: 'right'}}>10</td>
            <td style={{padding: '10px', textAlign: 'right'}}>71%</td>
          </tr>
          <tr style={{borderBottom: '1px solid #ddd'}}>
            <td style={{padding: '10px'}}>1:00–5:00 PM (Peak)</td>
            <td style={{padding: '10px', textAlign: 'right'}}>52</td>
            <td style={{padding: '10px', textAlign: 'right'}}>148</td>
            <td style={{padding: '10px', textAlign: 'right'}}>6.4</td>
            <td style={{padding: '10px', textAlign: 'right'}}>19</td>
            <td style={{padding: '10px', textAlign: 'right'}}>82%</td>
          </tr>
          <tr style={{borderBottom: '1px solid #ddd'}}>
            <td style={{padding: '10px'}}>5:00–10:00 PM (Evening)</td>
            <td style={{padding: '10px', textAlign: 'right'}}>53</td>
            <td style={{padding: '10px', textAlign: 'right'}}>110</td>
            <td style={{padding: '10px', textAlign: 'right'}}>2.8</td>
            <td style={{padding: '10px', textAlign: 'right'}}>9</td>
            <td style={{padding: '10px', textAlign: 'right'}}>63%</td>
          </tr>
        </tbody>
      </table>

      <p>
        The afternoon period (1:00–5:00 PM) emerged as the critical bottleneck, with:
      </p>
      <ul>
        <li>Highest average queue length (6.4 passengers)</li>
        <li>Peak queue reaching 19 passengers</li>
        <li>Highest vehicle occupancy (82%)</li>
        <li>Most trips required (52 trips in 4 hours)</li>
      </ul>

      <p>
        This period coincided with students with morning classes leaving and students with afternoon classes arriving —creating a demand surge that stressed the system. Interestingly, despite being classified as "rush hour" by the transportation manager, the morning period (6:00–8:00 AM) was less congested, likely because student arrivals were more staggered.
      </p>

      <p>
        Evening operations (5:00–10:00 PM) had the lowest queue lengths despite spanning 5 hours, confirming the appropriateness of the 50% demand scaling for that period.
      </p>

      <h3>Vehicle Utilization</h3>
      <p>
        Analyzing individual vehicle performance revealed interesting patterns:
      </p>
      <ul>
        <li><strong>Bus A:</strong> Operated 11 hours (6 AM–5 PM), completed ~48 trips, transported ~285 passengers (66% occupancy). High throughput during its operating window but offline during evening peak.</li>
        <li><strong>Van A:</strong> Operated 14 hours (6 AM–8 PM), completed ~65 trips, transported ~95 passengers (75% occupancy). Consistently utilized throughout the day, flexible coverage.</li>
        <li><strong>Van B:</strong> Operated 15 hours (7 AM–10 PM), completed ~52 trips, transported ~50 passengers (79% occupancy). Critical for evening coverage when Bus A was unavailable.</li>
      </ul>

      <p>
        Vehicle occupancy rates ranged from 63% to 82%, indicating reasonable capacity utilization without systematic overcrowding or underutilization. The 1:00–5:00 PM period pushed occupancy above 80%, suggesting that additional capacity during this window could further reduce wait times if desired.
      </p>

      <div style={{margin: '40px 0', textAlign: 'center'}}>
        <img src={timelineOperaciones} alt="Timeline of Daily Operations" style={{maxWidth: '100%', height: 'auto', border: '1px solid #ddd', borderRadius: '4px'}} />
        <p style={{fontSize: '0.9em', color: '#666', marginTop: '10px', fontStyle: 'italic'}}>
          Timeline visualization showing all vehicle trips throughout the 16-hour operating day. Top panel shows passengers boarded per trip (bubble size indicates occupancy %), with color coding by vehicle. Bottom panel tracks queue evolution, clearly showing the afternoon peak period (1:00-5:00 PM) where demand surges and queues grow.
        </p>
      </div>

      <div style={{margin: '40px 0', textAlign: 'center'}}>
        <img src={ocupacionVehiculos} alt="Vehicle Occupancy Analysis" style={{maxWidth: '100%', height: 'auto', border: '1px solid #ddd', borderRadius: '4px'}} />
        <p style={{fontSize: '0.9em', color: '#666', marginTop: '10px', fontStyle: 'italic'}}>
          Comprehensive vehicle utilization metrics: average occupancy percentage by vehicle (top left), distribution of trip occupancies across all trips (top right), total passengers transported per vehicle (bottom left), and trip count per vehicle (bottom right). Most trips operated at 60-80% capacity, indicating efficient resource utilization.
        </p>
      </div>

      <div style={{margin: '40px 0', textAlign: 'center'}}>
        <img src={evolucionOcupacion} alt="Hourly Occupancy Evolution" style={{maxWidth: '100%', height: 'auto', border: '1px solid #ddd', borderRadius: '4px'}} />
        <p style={{fontSize: '0.9em', color: '#666', marginTop: '10px', fontStyle: 'italic'}}>
          Hour-by-hour breakdown of system performance. Top panel shows occupancy percentage fluctuating throughout the day, dipping below the 80% target during morning and evening but peaking during afternoon hours. Bottom panel reveals the relationship between passenger demand (bars) and trip frequency (green line)—the afternoon period requires both more trips and higher passengers per trip.
        </p>
      </div>

      
      <h2>Challenges and Lessons Learned</h2>

      <h3>1. Modeling Realism vs. Complexity</h3>
      <p>
        One of the hardest decisions was determining the appropriate level of modeling detail. Real-world shuttle systems have many complexities:
      </p>
      <ul>
        <li>Travel time variability (traffic, route deviations)</li>
        <li>Vehicle breakdowns and maintenance</li>
        <li>Driver availability and shift constraints</li>
        <li>Passenger no-shows or early departures</li>
        <li>Weather impacts on demand patterns</li>
      </ul>

      <p>
        Including all these factors would have made the simulation intractable and difficult to validate. Instead, I focused on the primary sources of variability: stochastic passenger arrivals and vehicle trip timing based on empirical data.
      </p>
      <p>
        This taught me an important lesson: <strong>a useful model is not the most comprehensive one, but the simplest one that captures the essential dynamics</strong>. The goal wasn't to replicate every detail of reality, but to provide decision-makers with robust insights about fleet sizing and cost-service trade-offs.
      </p>

      <h3>2. Handling Missing Data</h3>
      <p>
        The absence of evening demand data (6:00–10:00 PM) was a significant challenge. Rather than ignoring this period or making arbitrary assumptions, I consulted with transportation managers to obtain their expert judgment: evening demand was roughly 50% of afternoon levels.
      </p>

      <h3>3. Validation and Calibration</h3>
      <p>
        How do you I if the simulation is correct? I used several validation approaches:
      </p>
      <ul>
        <li><strong>Sanity Checks:</strong> Total passengers served should approximately equal total passenger arrivals (validated by checking queue state at end of day)</li>
        <li><strong>Occupancy Bounds:</strong> Vehicle occupancy should never exceed 100% (enforced by capacity constraints)</li>
        <li><strong>Coverage Verification:</strong> At least one vehicle operating at all times (programmatically checked for all valid configurations)</li>
        <li><strong>Benchmark Comparison:</strong> Average wait times and queue lengths matched theoretical queueing models (M/M/c approximations) within reasonable ranges</li>
      </ul>

      <p>
        These checks increased confidence that the simulation was behaving correctly, though the ultimate validation would come from comparing predictions to actual operational data—a future step if my university implements one of the recommended configurations.
      </p>

      <h3>4. Communicating Uncertainty</h3>
      <p>
        Decision-makers often want definitive answers: "What's the best configuration?" But operations research rarely provides certainty—only probabilistic guidance. The Monte Carlo analysis was essential for communicating this uncertainty.
      </p>
      <p>
        Rather than saying "Configuration X has an average wait time of 7.9 minutes," I presented: "Configuration X has an average wait time of 7.9 minutes ± 0.6 minutes across 50 simulations, with a worst-case of 8.9 minutes." This framing helped stakeholders understand that <em>all</em> configurations have variability, and choosing a robust option means accepting some operational risk.
      </p>

      <h2>Extensions and Future Work</h2>
      <p>
        While the current analysis provided actionable recommendations, several extensions could further enhance the model:
      </p>

      <h3>Multi-Objective Optimization</h3>
      <p>
        Rather than manually selecting "balanced" configurations, Pareto optimization could systematically identify the entire efficient frontier of cost-service trade-offs. Algorithms like NSGA-II (Non-dominated Sorting Genetic Algorithm) could search the configuration space more exhaustively, potentially finding superior options.
      </p>

      <h3>Passenger Choice Modeling</h3>
      <p>
        The current model assumes passengers always wait for the next vehicle. In reality, passengers may have alternatives (mainly walking to campus) if wait times become excessive. Incorporating discrete choice models could capture this behavior and estimate passenger demand elasticity with respect to service quality.
      </p>

      <h3>Seasonal and Weekly Variation</h3>
      <p>
        Demand likely varies by day of week (Monday morning rush vs. Friday afternoon) and by season (summer vacations, holiday periods). Extending the model to accommodate multiple demand scenarios would make recommendations more robust to temporal variation.
      </p>


      <h2>Conclusion</h2>
      <p>
        Optimizing a corporate transportation fleet is a deceptively complex problem that sits at the intersection of queueing theory, stochastic simulation, and multi-objective optimization. What appears simple—"just add more vehicles until wait times drop"—becomes nuanced when balancing cost constraints, operational coverage requirements, and service quality under uncertainty.
      </p>
      <p>
        This project demonstrated that rigorous modeling and simulation can transform operational guesswork into data-driven decision-making. By evaluating 48 configurations across 2,400 Monte Carlo simulations, I identified three robust fleet designs that met all constraints while offering clear trade-offs between cost and service quality.
      </p>
      <p>
        The recommended balanced configuration—1 Bus A, 1 Van A, and 1 Van B—provided the best value proposition: only 9% more expensive than the cheapest option but with 4% lower wait times and more predictable performance. For an annual cost of $433,547, the company could operate a reliable shuttle service meeting its 10-minute service standard while keeping costs reasonable.
      </p>
      <p>
        More broadly, this project reinforced my appreciation for operations research as a discipline. It's not just about solving equations or running algorithms—it's about translating messy real-world problems into tractable models, quantifying uncertainty, and communicating results in ways that empower better decisions. From queueing theory to Monte Carlo methods to data visualization, this project integrated techniques from across the OR toolkit to deliver practical, actionable insights.
      </p>
      <p>
        And perhaps most importantly, it reminded me that good operations research doesn't end with a recommendation—it begins a conversation. The simulation framework I built can be reused as conditions change: new vehicles become available, demand patterns shift, or service standards tighten. The tools and methods remain valuable long after the initial analysis concludes.
      </p>
    </PageTemplate>
  );
}
