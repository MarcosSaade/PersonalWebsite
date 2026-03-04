import React from 'react';
import 'katex/dist/katex.min.css';
import { BlockMath, InlineMath } from 'react-katex';

import PageTemplate from '../components/PageTemplate';

// Import main header image (you'll need to add this)
import salesImage from '../images/sales-analytics.png';

// Import project visualizations
import outliersPlot from '../images/sales-project/outliers.png';
import correlationHeatmap from '../images/sales-project/correlacion_heatmap.png';
import ventasCategoria from '../images/sales-project/ventas_categoria.png';
import ventasRegion from '../images/sales-project/ventas_region.png';
import mediasMoviles from '../images/sales-project/medias_moviles.png';
import estacionalidad from '../images/sales-project/estacionalidad.png';
import metodosPago from '../images/sales-project/metodos_pago.png';
import elbowSilhouette from '../images/sales-project/elbow_silhouette.png';
import clusters3D from '../images/sales-project/clusters_3d.png';
import clustersPCA from '../images/sales-project/clusters_pca.png';
import maOptimization from '../images/sales-project/ma_optimization.png';
import featureImportance from '../images/sales-project/feature_importance.png';
import prediccionesRegion from '../images/sales-project/predicciones_region.png';
import predicciones2025Regiones from '../images/sales-project/predicciones_2025_regiones.png';
import predicciones2025Categoria from '../images/sales-project/predicciones_2025_categoria.png';

export default function SalesAnalysisPage() {
  return (
    <PageTemplate title="Forecasting Retail Demand: A Machine Learning Approach to Sales Pattern Analysis" image={salesImage}>
      {/* Introduction */}
      <p>
        This project analyzes sales patterns from an Argentine retail business to predict weekly demand across different regions and product categories. Built as part of my Multivariate Analysis for Data Science coursework, it demonstrates a practical application of machine learning to a real-world business problem: optimizing inventory management and resource allocation through data-driven demand forecasting.
      </p>
      <p>
        The dataset, sourced from <a href="https://www.kaggle.com/datasets/dataregina/datasets-para-proyecto-bi" target="_blank" rel="noopener noreferrer">Kaggle</a>, contains over 3,000 retail transactions spanning 11 months across six Argentine regions (Buenos Aires, Centro, Cuyo, NEA, NOA, and Patagonia) and six product categories (Bebidas, Carnicería, Frutas y Verduras, Galletitas y Snacks, Lácteos, and Limpieza).
      </p>
      <p>
        What makes this project interesting isn't just the technical implementation—it's the business context. Retail forecasting is fundamentally about balancing competing objectives: maintain enough inventory to meet demand without tying up capital in excess stock, allocate sales resources where they'll have the most impact, and anticipate seasonal patterns to prepare for demand fluctuations. This project addresses all three.
      </p>

      <h2>The Challenge: Predicting Demand in a Cyclical Market</h2>
      <p>
        Retail demand is notoriously difficult to forecast. Unlike financial markets where prices adjust to clear supply and demand, retail inventory decisions must be made in advance, often weeks or months before actual sales occur. Get it wrong in one direction, and you tie up capital in excess inventory. Get it wrong in the other, and you lose sales to stockouts.
      </p>
      <p>
        The challenge is compounded by several factors:
      </p>
      <ul>
        <li><strong>Cyclical patterns:</strong> Sales don't follow smooth trends—they exhibit peaks and valleys at various timescales</li>
        <li><strong>Regional heterogeneity:</strong> Different regions have different demand patterns, population sizes, and purchasing behaviors</li>
        <li><strong>Category differences:</strong> Fresh produce (Frutas y Verduras) behaves differently from packaged goods (Galletitas y Snacks)</li>
        <li><strong>Limited historical data:</strong> With only 11 months of data, distinguishing true seasonality from random variation is challenging</li>
      </ul>
      <p>
        My approach was to build a hybrid forecasting system that adapts to data availability: sophisticated gradient boosting models for high-volume regions with sufficient training data, and robust moving average methods for smaller regions where complex models would overfit.
      </p>

      <h2>Data Cleaning: The Foundation of Reliable Analysis</h2>
      <p>
        Before any analysis could begin, I needed to ensure data quality. Real-world datasets are messy, and this one was no exception. The cleaning process revealed several issues that required careful handling.
      </p>

      <h3>Duplicate Records and Data Integration</h3>
      <p>
        The initial dataset contained 3,029 sales records, but inspection revealed 29 duplicate transaction IDs—likely data entry errors. After removing duplicates, I was left with 3,000 unique transactions spanning February through December 2024.
      </p>
      <p>
        The data came fragmented across five separate tables: sales transactions, products, categories, customers, and payment methods. I integrated these using foreign key relationships to create a unified dataset containing all relevant information for each transaction.
      </p>

      <h3>Outlier Detection and Treatment</h3>
      <p>
        A critical decision involved how to handle outliers in sales amounts. Using the Interquartile Range (IQR) method, I identified transactions with amounts significantly above the norm:
      </p>
      <BlockMath math="\text{Outlier if } x < Q_1 - 1.5 \times \text{IQR} \text{ or } x > Q_3 + 1.5 \times \text{IQR}" />
      
      <div style={{margin: '30px 0', textAlign: 'center'}}>
        <img src={outliersPlot} alt="Outlier detection in sales data using IQR method" style={{width: '100%', maxWidth: '900px', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)'}} />
        <p style={{fontSize: '0.9em', color: '#666', marginTop: '10px'}}>
          Outlier detection revealed suspiciously uniform high-value transactions that didn't correspond to holidays or special events
        </p>
      </div>

      <p>
        Interestingly, the outliers weren't associated with Argentine holidays or special events—they appeared randomly distributed throughout the year and all had suspiciously identical amounts. This suggested data quality issues rather than genuine high-value purchases. After careful analysis, I removed them to prevent skewing the models.
      </p>

      <h3>Temporal Consistency</h3>
      <p>
        January data consisted of only a single day (January 31st), which would have severely biased any monthly analysis. I excluded this partial month to maintain temporal consistency. The final cleaned dataset spans February through December 2024, providing 11 months of reliable data for pattern detection and forecasting.
      </p>

      <h2>Exploratory Data Analysis: Finding the Patterns</h2>
      <p>
        With clean data in hand, I explored the patterns hiding in the sales records. The goal was to understand demand characteristics across regions, categories, and time—insights that would inform both the feature engineering and modeling strategies.
      </p>

      <h3>Regional and Category Insights</h3>
      
      <div style={{margin: '30px 0', textAlign: 'center'}}>
        <img src={correlationHeatmap} alt="Correlation between sales amount, quantity, and unit price" style={{width: '100%', maxWidth: '600px', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)'}} />
        <p style={{fontSize: '0.9em', color: '#666', marginTop: '10px'}}>
          Price affects sales more than quantity (0.71 vs 0.62), and higher prices don't reduce purchase quantity
        </p>
      </div>

      <p>
        The analysis revealed clear structural differences across regions and categories:
      </p>
      <p>
        <strong>Buenos Aires dominates sales volume</strong>, which isn't surprising given its population size. But when I calculated sales per customer, interesting patterns emerged: Cuyo region has the highest average spend per customer, while Patagonia has the lowest—about a 10% difference.
      </p>

      <div style={{margin: '30px 0', textAlign: 'center'}}>
        <img src={ventasRegion} alt="Sales by region and sales per customer" style={{width: '100%', maxWidth: '900px', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)'}} />
        <p style={{fontSize: '0.9em', color: '#666', marginTop: '10px'}}>
          Regional sales patterns show Buenos Aires leads in volume, but Cuyo has the highest spend per customer
        </p>
      </div>

      <p>
        <strong>Category analysis showed a price-volume tradeoff:</strong> Carnicería (butcher/meat products) leads in total sales revenue, but Frutas y Verduras (fruits and vegetables) leads in unit volume. This reflects the economic reality that meat products command higher prices than produce.
      </p>

      <div style={{margin: '30px 0', textAlign: 'center'}}>
        <img src={ventasCategoria} alt="Sales by category showing revenue vs volume" style={{width: '100%', maxWidth: '900px', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)'}} />
        <p style={{fontSize: '0.9em', color: '#666', marginTop: '10px'}}>
          Carnicería leads in revenue while Frutas y Verduras leads in volume
        </p>
      </div>

      <p>
        <strong>Payment methods strongly favor Mercado Pago</strong>, Argentina's dominant digital payment platform, across all regions and categories.
      </p>

      <div style={{margin: '30px 0', textAlign: 'center'}}>
        <img src={metodosPago} alt="Payment method analysis across regions and time" style={{width: '100%', maxWidth: '900px', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)'}} />
        <p style={{fontSize: '0.9em', color: '#666', marginTop: '10px'}}>
          Mercado Pago dominates across all dimensions, with summer increase in bank transfers
        </p>
      </div>

      <h3>Temporal Patterns and Cyclicality</h3>
      <p>
        The temporal analysis revealed the most interesting insights—and the most challenging forecasting aspects. Sales exhibit strong cyclical patterns at multiple timescales.
      </p>

      <div style={{margin: '30px 0', textAlign: 'center'}}>
        <img src={mediasMoviles} alt="Daily sales with 7, 14, and 30-day moving averages" style={{width: '100%', maxWidth: '1000px', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)'}} />
        <p style={{fontSize: '0.9em', color: '#666', marginTop: '10px'}}>
          Moving averages reveal cyclical patterns with peaks every 2-8 weeks. Stars indicate highest sales days.
        </p>
      </div>

      <p>
        Using peak detection algorithms on moving averages, I quantified the cycles:
      </p>
      <ul>
        <li><strong>7-day MA peaks:</strong> Average 19.67 days apart (2-4 week cycles)</li>
        <li><strong>14-day MA peaks:</strong> Average 46.86 days apart (6-8 week cycles)</li>
      </ul>

      <div style={{margin: '30px 0', textAlign: 'center'}}>
        <img src={estacionalidad} alt="Monthly seasonality showing percentage deviation from average" style={{width: '100%', maxWidth: '800px', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)'}} />
        <p style={{fontSize: '0.9em', color: '#666', marginTop: '10px'}}>
          Monthly seasonality patterns (though limited to 11 months of data)
        </p>
      </div>

      <h2>Feature Engineering: Extracting Temporal Intelligence</h2>
      <p>
        The cyclical patterns observed in EDA made it clear that successful forecasting would require rich temporal features. I implemented a comprehensive feature engineering pipeline that transforms raw transaction data into machine learning-ready features capturing multiple aspects of demand patterns.
      </p>

      <h3>Multi-Scale Temporal Features</h3>
      <p>
        I created a hierarchy of temporal features:
      </p>
      <ul>
        <li><strong>Basic temporal:</strong> Week of year, month, day of week</li>
        <li><strong>Lag features:</strong> Previous period quantities by category, region, and their combinations</li>
        <li><strong>Rolling statistics:</strong> Mean and standard deviation over 3, 7, 14, and 30-day windows</li>
        <li><strong>Multi-week aggregations:</strong> 2-6 week rolling features for weekly predictions</li>
        <li><strong>Interaction features:</strong> Ratios of current to historical demand, trend indicators, volatility combinations</li>
      </ul>
      <p>
        <strong>Critical implementation detail:</strong> To prevent data leakage, all rolling features use <code>.shift(1)</code> before computing statistics. This ensures we only use information available at the time of prediction, not future information.
      </p>

      <h2>Clustering: Segmenting Products by Demand Behavior</h2>
      <p>
        Before building predictive models, I applied K-Means clustering to segment products and regions by their demand patterns using quantity, rolling statistics, and volatility measures.
      </p>

      <div style={{margin: '30px 0', textAlign: 'center'}}>
        <img src={elbowSilhouette} alt="Elbow method and Silhouette Score for optimal cluster selection" style={{width: '100%', maxWidth: '900px', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)'}} />
        <p style={{fontSize: '0.9em', color: '#666', marginTop: '10px'}}>
          Both metrics pointed to k=6 as the optimal number of clusters
        </p>
      </div>

      <p>
        Using both the Elbow method (inertia) and Silhouette Score, I determined <strong>6 clusters</strong> as optimal. Visualization using PCA revealed clear separation between demand segments.
      </p>

      <div style={{margin: '30px 0', textAlign: 'center'}}>
        <img src={clusters3D} alt="3D visualization of 6 demand clusters using PCA" style={{width: '100%', maxWidth: '800px', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)'}} />
        <p style={{fontSize: '0.9em', color: '#666', marginTop: '10px'}}>
          Six distinct demand clusters visualized in 3D PCA space
        </p>
      </div>

      <h2>Predictive Modeling: A Hybrid Approach</h2>
      <p>
        Rather than applying a single model uniformly, I developed a hybrid approach that matches model complexity to data availability.
      </p>

      <h3>LightGBM for High-Volume Regions</h3>
      <p>
        For Buenos Aires, Centro, Cuyo, and Patagonia, I trained a LightGBM regressor using Optuna for Bayesian hyperparameter optimization:
      </p>
      <ul>
        <li>50 trials with TPE Sampler and Median Pruner</li>
        <li>Time Series Cross-Validation (5 folds) for unbiased evaluation</li>
        <li>Optimal parameters emphasized regularization (learning rate 0.021, feature fraction 0.74)</li>
      </ul>

      <div style={{margin: '30px 0', textAlign: 'center'}}>
        <img src={featureImportance} alt="Top 20 features by importance in LightGBM model" style={{width: '100%', maxWidth: '800px', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)'}} />
        <p style={{fontSize: '0.9em', color: '#666', marginTop: '10px'}}>
          Lag features and rolling means dominate model predictions
        </p>
      </div>

      <h3>Moving Average for Low-Volume Regions</h3>
      <p>
        For NEA and NOA, I used an optimized moving average approach. Testing windows from 1 to 16 weeks:
      </p>

      <div style={{margin: '30px 0', textAlign: 'center'}}>
        <img src={maOptimization} alt="RMSE vs window size for moving average optimization" style={{width: '100%', maxWidth: '800px', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)'}} />
        <p style={{fontSize: '0.9em', color: '#666', marginTop: '10px'}}>
          Optimal window of n=11 weeks minimized RMSE
        </p>
      </div>

      <p>
        The 11-week window captures medium-term trends while smoothing short-term volatility—appropriate for regions with limited training data.
      </p>

      <h2>Model Performance and Results</h2>
      <p>
        The hybrid approach successfully forecasts weekly demand across all regions:
      </p>

      <div style={{margin: '30px 0', textAlign: 'center'}}>
        <img src={prediccionesRegion} alt="Predictions vs actual for all 6 regions" style={{width: '100%', maxWidth: '1000px', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)'}} />
        <p style={{fontSize: '0.9em', color: '#666', marginTop: '10px'}}>
          Model predictions (dashed) track actual demand (solid) across all regions
        </p>
      </div>

      <p>
        The models effectively capture major peaks and troughs:
      </p>
      <ul>
        <li><strong>LightGBM regions:</strong> Capture complex patterns with slight underestimation of extremes</li>
        <li><strong>Moving average regions:</strong> Provide stable, robust estimates with characteristic lag</li>
      </ul>

      <p>
        These models resulted from empirical evaluation of various algorithms for each region. LightGBM outperformed alternatives (XGBoost, Lasso, Linear Regression, MA) in data-rich regions, while moving averages outperformed complex ML methods and ARIMA.
      </p>

      <h2>2025 Demand Forecasts</h2>
      <p>
        With trained models, I generated forward-looking forecasts for 2025 by projecting 2024 patterns forward, maintaining the 52-week cycle structure.
      </p>

      <div style={{margin: '30px 0', textAlign: 'center'}}>
        <img src={predicciones2025Regiones} alt="2025 demand forecasts by region" style={{width: '100%', maxWidth: '1000px', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)'}} />
        <p style={{fontSize: '0.9em', color: '#666', marginTop: '10px'}}>
          Weekly demand forecasts for 2025 across all six regions
        </p>
      </div>

      <div style={{margin: '30px 0', textAlign: 'center'}}>
        <img src={predicciones2025Categoria} alt="2025 demand forecasts by product category" style={{width: '100%', maxWidth: '1000px', borderRadius: '8px', boxShadow: '0 4px 6px rgba(0,0,0,0.1)'}} />
        <p style={{fontSize: '0.9em', color: '#666', marginTop: '10px'}}>
          Stacked view of 2025 weekly demand by product category
        </p>
      </div>

      <p>
        The forecasts preserve observed cyclical patterns and provide actionable inputs for:
      </p>
      <ul>
        <li>Weekly inventory planning by category and region</li>
        <li>Sales force allocation based on predicted demand peaks</li>
        <li>Capacity planning for the 2-8 week demand cycles</li>
      </ul>

      <p>
        I also created an interactive dashboard to help visualize predictions by category, region, and individual products.
      </p>

      <h2>Business Implications and Recommendations</h2>
      <p>
        The forecasts enable several operational improvements:
      </p>

      <h3>Inventory Management</h3>
      <ul>
        <li><strong>Category-specific strategies:</strong> Different replenishment cadences for high-volume (Frutas y Verduras) vs. low-volume (Limpieza) categories</li>
        <li><strong>Regional allocation:</strong> Adjust stock levels for the 10% spending difference between Cuyo and Patagonia</li>
        <li><strong>Cycle preparation:</strong> Pre-position inventory ahead of predictable 2-8 week demand peaks</li>
      </ul>
      <p>
        However, <strong>full inventory optimization requires cost data</strong> (storage, spoilage, holding costs) not present in this dataset.
      </p>

      <h3>Resource Allocation</h3>
      <ul>
        <li><strong>Dynamic staffing:</strong> Flex workforce with the 2-8 week demand cycles</li>
        <li><strong>Regional focus:</strong> While Buenos Aires has highest absolute sales, Cuyo's higher per-customer spend might warrant proportionally more attention</li>
        <li><strong>Category expertise:</strong> Carnicería's revenue dominance suggests specialized support for this high-value category</li>
      </ul>

      <h2>Limitations and Future Work</h2>
      <p>
        Several constraints limit the current analysis:
      </p>

      <h3>Data Constraints</h3>
      <ul>
        <li><strong>Single year of data:</strong> Limits confidence in seasonal patterns—multi-year data would enable stronger seasonality detection</li>
        <li><strong>Fixed prices:</strong> Prevents price elasticity analysis and revenue optimization</li>
        <li><strong>Low multi-product rate:</strong> Only 1.5% of transactions included multiple products—surprisingly low and worth validating</li>
        <li><strong>Missing cost structure:</strong> Prevents full inventory and workforce optimization</li>
      </ul>

      <h3>Model Enhancements</h3>
      <ul>
        <li>External factors (weather, economic indicators, competitor actions)</li>
        <li>Hierarchical forecasting with reconciliation</li>
        <li>Prediction intervals for uncertainty quantification</li>
        <li>Online learning for continuous model updates</li>
        <li>Deep learning approaches (LSTM/Transformer models)</li>
      </ul>

      <h2>Technical Implementation</h2>
      <p>
        The project is implemented in Python with modular architecture:
      </p>
      <ul>
        <li><strong>Data cleaning:</strong> Jupyter notebooks for exploratory analysis</li>
        <li><strong>Feature engineering:</strong> <code>FeatureEngineer</code> class with chained methods</li>
        <li><strong>Clustering:</strong> K-Means with PCA visualization</li>
        <li><strong>Modeling:</strong> Separate LightGBM and moving average workflows</li>
        <li><strong>Forecasting:</strong> 2025 prediction generation pipeline</li>
      </ul>
      <p>
        Key methodological choices ensure rigor:
      </p>
      <ul>
        <li>Time Series CV prevents data leakage</li>
        <li>Shifted rolling features avoid using future information</li>
        <li>Native missing value handling by LightGBM</li>
        <li>Fixed random seeds for reproducibility</li>
      </ul>

      <h2>Reflections</h2>
      <p>
        This project reinforced several lessons about applied machine learning in business contexts:
      </p>
      <p>
        <strong>Data quality matters more than model sophistication.</strong> Careful cleaning, outlier analysis, and temporal consistency checking were essential foundation work.
      </p>
      <p>
        <strong>Feature engineering bridges domain knowledge and ML.</strong> The cyclical patterns observed in EDA directly informed rolling feature windows and lag structures.
      </p>
      <p>
        <strong>Model complexity should match data availability.</strong> The hybrid approach—LightGBM for data-rich regions, moving averages for data-sparse regions—prevented both underfitting and overfitting.
      </p>
      <p>
        <strong>Business value requires more than predictions.</strong> The forecasts provide inputs to decision-making but need cost data and constraints to become actionable inventory or staffing plans.
      </p>
      <p>
        <strong>Validation reveals assumptions.</strong> The low multi-product transaction rate and random outliers surfaced potential data collection issues worth investigating.
      </p>
      <p>
        Looking forward, I'd extend this work with multi-year data to validate seasonality, external data sources to improve predictions, and cost structures to build full optimization models. Even with limitations, this project demonstrates how careful analysis and appropriate modeling extract actionable insights from retail transaction data.
      </p>

      <p style={{marginTop: '40px', fontSize: '0.9em', color: '#666', borderTop: '1px solid #ddd', paddingTop: '20px'}}>
        <strong>Technologies:</strong> Python, pandas, NumPy, LightGBM, scikit-learn, Optuna, matplotlib, seaborn
        <br />
        <strong>Techniques:</strong> Time Series Forecasting, Gradient Boosting, K-Means Clustering, Feature Engineering, Bayesian Optimization, Cross-Validation
      </p>
    </PageTemplate>
  );
}
