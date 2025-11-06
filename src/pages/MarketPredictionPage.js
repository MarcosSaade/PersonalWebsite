import React from 'react';
import 'katex/dist/katex.min.css';
import { BlockMath, InlineMath } from 'react-katex';

import PageTemplate from '../components/PageTemplate';
import financeImage from '../images/quant.png';

export default function MarketPredictionPage() {
  return (
    <PageTemplate title="Predicting S&P 500 Allocations with Machine Learning" image={financeImage}>
      {/* Introduction */}
      <p>
        This project emerged from the <a href="https://www.kaggle.com/competitions/hull-tactical-market-prediction" target="_blank" rel="noopener noreferrer">Hull Tactical Market Prediction</a> Kaggle competition—a challenging forecasting task where participants predicted optimal daily allocations to the S&P 500 index. The twist? All features were completely obfuscated. We had no idea what the actual data represented.
      </p>
      <p>
        Instead of traditional market data with clear labels like "VIX," "10-year yield," or "unemployment rate," we received cryptic names: E1 through E20 (economic features), I1 through I10 (index features), M1 through M14 (monetary features), and so on. This obfuscation was likely due to proprietary data from the competition organizers, but it created a fascinating constraint: I couldn't rely on domain expertise or financial intuition to engineer features.
      </p>
      <p>
        This forced me to approach the problem purely quantitatively. Without knowing which feature represented what, I had to extract every possible statistical pattern, interaction, and temporal relationship from the data. The result was a sophisticated pipeline generating over 2,000 engineered features from just 68 raw inputs—and then intelligently reducing them back down to what truly mattered.
      </p>
      <p>
        What started as a competition challenge became an exercise in modern financial machine learning. I drew heavily from concepts in Marcos López de Prado's "Advances in Financial Machine Learning," implementing purged cross-validation, meta-labeling, and regime-dependent position sizing. The project taught me as much about rigorous ML methodology as it did about quantitative finance.
      </p>

      <h2>The Challenge: Tactical Asset Allocation Under Uncertainty</h2>
      <p>
        Tactical asset allocation is about more than just predicting whether markets will go up or down. It requires three interconnected decisions:
      </p>
      <ul>
        <li><strong>Direction:</strong> Will the market rise or fall?</li>
        <li><strong>Magnitude:</strong> How much will it move?</li>
        <li><strong>Confidence:</strong> How certain are we about this prediction?</li>
      </ul>
      <p>
        These three factors determine your position size. A strong upward prediction with high confidence should warrant a larger allocation. A weak signal with low confidence might suggest staying close to neutral or even going to cash.
      </p>
      <p>
        The competition asked us to output a daily allocation recommendation ranging from 0% (fully out of the market) to 200% (leveraged 2x long). This meant we needed not just a return forecast, but a complete decision-making framework that could translate predictions into risk-managed positions.
      </p>
      <p>
        The evaluation metric was an adjusted Sharpe ratio that penalized strategies taking on excessive volatility or underperforming the market. The metric applied two penalties:
      </p>
      <ul>
        <li><strong>Volatility Penalty:</strong> If your strategy's annualized volatility exceeded 120% of the market's volatility, the Sharpe ratio was penalized proportionally to the excess</li>
        <li><strong>Return Penalty:</strong> If your strategy underperformed the market, the penalty scaled quadratically with the return gap</li>
      </ul>
      <p>
        This meant aggressive leverage or wild tactical bets would be punished unless they delivered proportionally higher returns.
      </p>
      <p>
        Traditional approaches often suffer from overfitting due to improper handling of time series data and fail to account for changing market regimes. My solution addressed these issues head-on through careful cross-validation, separate modeling of returns and volatility, and regime-adaptive position sizing.
      </p>

      <h2>The Data: 68 Obfuscated Features</h2>
      <p>
        The dataset contained 68 features across six categories, all with cryptic names:
      </p>
      <ul>
        <li><strong>Economic Indicators (E1–E20):</strong> Likely macroeconomic variables such as employment, inflation, or GDP metrics</li>
        <li><strong>Market Indices (I1–I10):</strong> Probably equity, fixed income, and commodity indices</li>
        <li><strong>Monetary Variables (M1–M14):</strong> Possibly interest rates, yield curves, and central bank metrics</li>
        <li><strong>Price Data (P1–P13):</strong> Presumably OHLC data and derived price metrics for various assets</li>
        <li><strong>Sentiment Indicators (S1–S12):</strong> Likely market sentiment and positioning data</li>
        <li><strong>Volatility Measures (V1–V13):</strong> Probably historical and implied volatility across timeframes</li>
      </ul>
      <p>
        Without knowing what each feature actually represented, I had to treat them as abstract signals and let the data speak for itself through statistical relationships.
      </p>

      <h3>Data Quality Challenges</h3>
      <p>
        Real-world financial data is messy. The dataset had significant quality issues that required careful handling:
      </p>
      <p>
        <strong>Missing values</strong> were pervasive. Some features like E7 had over 70% missing data and were excluded entirely. Others like V10, S3, and M1 had 40-50% missing values but showed up late in the time series, suggesting they represented newer data sources that became available partway through the historical period. I retained these with availability masks and forward-filled up to 5 consecutive missing values.
      </p>
      <p>
        <strong>Outliers</strong> were another challenge. Financial data naturally has fat tails—extreme events are more common than normal distributions would suggest. But some outliers were clearly data errors or measurement anomalies. I used Median Absolute Deviation (MAD) to detect outliers and applied feature-specific winsorization thresholds:
      </p>
      <ul>
        <li>Conservative (2.5x MAD) for features with frequent outliers</li>
        <li>Moderate (3.0x MAD) for medium outlier frequency</li>
        <li>Relaxed (4.0x MAD) for generally well-behaved features</li>
      </ul>
      <p>
        This preserved important tail information while controlling for data quality issues.
      </p>

      <h2>Preventing Look-Ahead Bias: Purged Cross-Validation</h2>
      <p>
        One of the most critical aspects of financial machine learning is avoiding look-ahead bias—the subtle ways information from the future can leak into your training process. Standard cross-validation, which randomly splits data into folds, is catastrophically inappropriate for time series.
      </p>
      <p>
        Consider this: if you're predicting tomorrow's return, your label is based on tomorrow's price. If your validation set includes today's data point but your training set includes tomorrow's, you've just trained on the future. This is called label leakage, and it's one of the most common reasons financial ML models fail in production after backtesting beautifully.
      </p>
      <p>
        I implemented purged k-fold cross-validation, which handles time series data properly:
      </p>
      <ul>
        <li><strong>Expanding Window:</strong> 5 folds where each training set only includes data from the past, never the future</li>
        <li><strong>Purging:</strong> Removes training samples whose labels overlap with the validation period. Since I was predicting 1-day forward returns, I purged at least 1 sample between training and validation</li>
        <li><strong>Embargo Period:</strong> An additional 1% buffer to prevent serial correlation from leaking between adjacent samples</li>
      </ul>
      <p>
        This methodology, inspired by López de Prado's work, ensures validation performance accurately reflects what you'd see in production.
      </p>

      <h2>Feature Engineering: Extracting Signal from Noise</h2>
      <p>
        With obfuscated features, I couldn't use financial intuition to create meaningful transformations. Instead, I took a comprehensive statistical approach, generating features that would capture temporal patterns, volatility dynamics, and regime characteristics regardless of what the underlying data represented.
      </p>

      <h3>Temporal Features</h3>
      <p>
        Financial patterns manifest across different time horizons. A trend visible in daily data might not appear in weekly aggregations, and vice versa. I computed rolling statistics across multiple windows:
      </p>
      <ul>
        <li><strong>Short-term (5-day):</strong> Intraweek patterns—mean, standard deviation, min, max, and exponential moving averages</li>
        <li><strong>Medium-term (21-day):</strong> Roughly monthly trends—rolling statistics, momentum indicators, and volatility-of-volatility</li>
        <li><strong>Long-term (63 and 126-day):</strong> Quarterly and semi-annual patterns—extended rolling windows and trend strength measures</li>
      </ul>
      <p>
        For each raw feature and each window, I computed central tendency (mean, median), dispersion (std, range), and position (min, max, percentiles). This multi-scale approach captured both short-term noise and long-term signal.
      </p>

      <h3>Volatility Features</h3>
      <p>
        Volatility—the degree of price fluctuation—is fundamental to finance. It determines risk, affects option pricing, and signals regime changes. I extracted volatility in multiple ways:
      </p>
      <ul>
        <li><strong>Historical Volatility:</strong> Standard deviation of returns over windows of 5, 21, 63, and 126 days</li>
        <li><strong>EWMA Volatility:</strong> Exponentially-weighted moving average volatility, which gives more weight to recent observations</li>
        <li><strong>Range-Based Volatility:</strong> High-low ranges, which can be more robust than close-to-close volatility</li>
      </ul>
      <p>
        These volatility features would later become crucial for regime detection and risk-adjusted position sizing.
      </p>

      <h3>Regime Features</h3>
      <p>
        Markets don't behave the same way all the time. Low-volatility trending markets require different strategies than high-volatility choppy markets. I classified market conditions into regimes based on rolling volatility percentiles:
      </p>
      <ul>
        <li><strong>Low Volatility:</strong> Below 33rd percentile of 21-day rolling volatility</li>
        <li><strong>Medium Volatility:</strong> Between 33rd and 66th percentile</li>
        <li><strong>High Volatility:</strong> Above 66th percentile</li>
      </ul>
      <p>
        These regime labels allowed the model to learn different patterns for different market conditions and enabled regime-specific position sizing later in the pipeline.
      </p>

      <h3>Dimensionality Reduction</h3>
      <p>
        After generating all possible statistical transformations, I had over 2,000 features from 68 raw inputs. This created both computational and statistical challenges. More features mean more opportunities for overfitting, especially with limited training data.
      </p>
      <p>
        I implemented a three-stage reduction strategy:
      </p>
      <p>
        <strong>Stage 1: Correlation Clustering.</strong> Features with pairwise correlation above 0.95 were grouped together. These highly correlated features provide redundant information—knowing one tells you almost everything about the others. Clustering reduced multicollinearity while preserving the underlying information.
      </p>
      <p>
        <strong>Stage 2: PCA per Cluster.</strong> Within each correlation cluster, I applied Principal Component Analysis (PCA) to extract the main directions of variation. I retained enough components to explain 85% of each cluster's variance. This transformed groups of correlated features into uncorrelated principal components.
      </p>
      <p>
        <strong>Stage 3: Feature Selection by Importance.</strong> Finally, I trained a preliminary model (a Random Forest) to score feature importance and selected the top 150 features. This balanced model capacity against overfitting risk.
      </p>
      <p>
        The result was a refined feature set of 150 predictors that captured the essential patterns from the original 2,000+ engineered features.
      </p>

      <h2>The Modeling Pipeline: A Multi-Stage Framework</h2>
      <p>
        Rather than training a single model to predict allocations directly, I built a modular pipeline where specialized models handled different aspects of the problem. This separation of concerns made the system more interpretable, easier to debug, and ultimately more robust.
      </p>
      <p>
        The pipeline had three main components:
      </p>
      <ol>
        <li><strong>Return Prediction Model:</strong> Predicts expected return (μ)</li>
        <li><strong>Volatility Prediction Model:</strong> Predicts forecast uncertainty (σ)</li>
        <li><strong>Meta-Labeling Model:</strong> Predicts confidence in the return forecast</li>
      </ol>
      <p>
        These outputs were then combined through a regime-dependent position sizing framework to produce final allocation recommendations.
      </p>

      <h3>Return Prediction</h3>
      <p>
        The first model predicts 1-day forward excess returns using LightGBM, a gradient boosting framework that builds an ensemble of decision trees. I chose LightGBM for its efficiency with high-dimensional data, native handling of missing values, and strong regularization capabilities.
      </p>
      <p>
        The model was trained with careful regularization to prevent overfitting:
      </p>
      <ul>
        <li>31 leaves per tree (controls complexity)</li>
        <li>Learning rate of 0.05 (slower learning, better generalization)</li>
        <li>80% feature and row sampling (randomness to prevent overfitting)</li>
        <li>L1 and L2 regularization penalties</li>
        <li>Early stopping after 50 rounds without validation improvement</li>
      </ul>
      <p>
        The output is a point estimate of expected return, which forms the μ component in our position sizing formula.
      </p>

      <h3>Volatility Prediction</h3>
      <p>
        Knowing expected return isn't enough—we need to know how uncertain that prediction is. A predicted 1% gain with high uncertainty should warrant a smaller position than the same prediction with low uncertainty.
      </p>
      <p>
        I trained a second LightGBM model to predict the volatility of prediction errors. The target variable was the log-variance of residuals from the return model:
      </p>
      <BlockMath math="\text{target} = \log\left(\text{residuals}^2 + \epsilon\right)" />
      <p>
        This formulation models heteroskedasticity—the tendency for forecast errors to have time-varying variance. During calm markets, prediction errors might be small and consistent. During turbulent periods, they can be large and erratic.
      </p>
      <p>
        The volatility model used features specifically related to risk: historical volatility across multiple windows, GARCH-like features (lagged squared returns), volatility trends, and regime indicators.
      </p>
      <p>
        Raw predictions required calibration through a multi-stage pipeline:
      </p>
      <ol>
        <li><strong>Bias Correction:</strong> Transform from log-variance scale back to volatility with empirical bias adjustment</li>
        <li><strong>Clipping:</strong> Bound predictions to reasonable ranges (10th to 90th percentile of training distribution)</li>
        <li><strong>EWMA Smoothing:</strong> Apply exponential smoothing with decay parameter 0.9 to reduce noise</li>
      </ol>
      <p>
        This produced stable, well-calibrated volatility forecasts (σ) that accurately reflected forecast uncertainty.
      </p>

      <h3>Meta-Labeling: Separating Prediction from Position Sizing</h3>
      <p>
        Meta-labeling is one of the most powerful concepts I learned from López de Prado's work. The idea is elegant: instead of asking a single model to both predict returns and determine position sizes, separate these tasks.
      </p>
      <p>
        The primary model (the return predictor) determines direction and magnitude. The meta-model predicts a different question: "How confident should we be that the primary prediction is correct?"
      </p>
      <p>
        To train the meta-model, I created binary labels:
      </p>
      <BlockMath math="\text{meta-label} = \begin{cases} 1 & \text{if } \text{sign}(\hat{\mu}) = \text{sign}(\mu_{\text{actual}}) \\ 0 & \text{otherwise} \end{cases}" />
      <p>
        This transforms the problem into classification: predict whether the primary model's directional call (positive vs. negative) was correct.
      </p>
      <p>
        The meta-features included:
      </p>
      <ul>
        <li>All original market features (regime, volatility, economic indicators)</li>
        <li>Primary prediction magnitude |μ|</li>
        <li>Primary prediction sign</li>
        <li>Rolling statistics of past predictions (prediction volatility, recent accuracy)</li>
      </ul>
      <p>
        The meta-classifier (another LightGBM model) outputs a probability: P(primary prediction correct) ∈ [0, 1].
      </p>
      <p>
        Final allocations are scaled by this confidence:
      </p>
      <BlockMath math="\text{final allocation} = \text{base allocation} \times P(\text{correct})" />
      <p>
        This framework allows the system to reduce positions when uncertain and increase positions when confident, improving risk-adjusted returns without changing the underlying predictions.
      </p>

      <h2>Position Sizing: The Kelly Criterion</h2>
      <p>
        With predictions for return (μ), volatility (σ), and confidence in hand, the final step is determining actual position size. This is where the Kelly criterion comes in—a mathematical formula for optimal position sizing.
      </p>
      <p>
        The Kelly criterion answers the question: "What fraction of my capital should I allocate to maximize long-term growth?" The formula is remarkably simple:
      </p>
      <BlockMath math="f^* = \frac{\mu}{\sigma^2}" />
      <p>
        where <InlineMath math="f^*" /> is the optimal fraction, μ is expected return, and <InlineMath math="\sigma^2" /> is variance.
      </p>
      <p>
        The intuition is clear: allocate more when expected return is high, and less when volatility is high.
      </p>
      <p>
        However, raw Kelly can be aggressive—it assumes your return estimates are perfect and can recommend extremely large positions. In practice, I applied several modifications:
      </p>

      <h3>Regime-Dependent Kelly Fractions</h3>
      <p>
        Markets exhibit time-varying risk characteristics. What worked in a calm, trending market might fail catastrophically in a volatile, mean-reverting regime. I applied regime-specific scaling factors:
      </p>
      <ul>
        <li><strong>Low Volatility Regime:</strong> k = 0.99 (near-full Kelly, aggressive)</li>
        <li><strong>Medium Volatility Regime:</strong> k = 0.71 (moderate)</li>
        <li><strong>High Volatility Regime:</strong> k = 0.50 (conservative, half-Kelly)</li>
      </ul>
      <p>
        These scaling factors were calibrated through cross-validation to optimize risk-adjusted returns in each regime.
      </p>

      <h3>Final Allocation Formula</h3>
      <p>
        The complete position sizing formula incorporates all components:
      </p>
      <BlockMath math="\text{allocation} = \text{clip}\left(k \cdot \tanh\left(\frac{\mu}{\sigma^2 \times s + \epsilon}\right) \times P(\text{correct}), \, 0, \, 2\right)" />
      <p>
        Where:
      </p>
      <ul>
        <li><strong>μ / (σ² × s + ε):</strong> Kelly ratio with scale parameter s = 1.15 and small ε to prevent division by zero</li>
        <li><strong>tanh(·):</strong> Smooth bounding function that asymptotically approaches -1 and +1, preventing extreme allocations</li>
        <li><strong>k:</strong> Regime-dependent Kelly fraction (0.5 to 1.0)</li>
        <li><strong>× P(correct):</strong> Meta-labeling confidence scaling</li>
        <li><strong>clip(·, 0, 2):</strong> Hard constraints enforcing 0% to 200% allocation range</li>
      </ul>
      <p>
        This formula balances mathematical optimality with practical risk management, adapting to market conditions while respecting competition constraints.
      </p>

      <h2>Implementation and Training</h2>
      <p>
        Building the models was one challenge; orchestrating their training was another. The three models had dependencies—the volatility model needed return predictions, and the meta-model needed both—which required careful coordination. This was a challenge during pipeline iterations.
      </p>
      <p>
        The training script handled dependencies explicitly, training models in sequence: returns first, then volatility and meta-labeling in parallel (both depend on returns but not on each other).
      </p>
      <p>
        Hyperparameter tuning was done using Optuna, a Bayesian optimization framework. Rather than grid search (which explores a fixed grid) or random search (which samples randomly), Bayesian optimization learns from previous trials to intelligently explore the hyperparameter space.
      </p>
      <p>
        For deployment, I implemented a stateful predictor that maintains rolling history across prediction batches—essential for computing features that require lookback windows. The system handles missing features gracefully and applies regime detection using historical volatility.
      </p>

      <h2>Performance and Results</h2>
      <p>
        Evaluating the system across 5 purged cross-validation folds revealed consistent risk-adjusted performance:
      </p>
      <table style={{margin: '20px 0', borderCollapse: 'collapse', width: '100%'}}>
        <thead>
          <tr style={{borderBottom: '2px solid #333'}}>
            <th style={{padding: '10px', textAlign: 'left'}}>Fold</th>
            <th style={{padding: '10px', textAlign: 'right'}}>Adjusted Sharpe Ratio</th>
            <th style={{padding: '10px', textAlign: 'right'}}>Direction Accuracy</th>
          </tr>
        </thead>
        <tbody>
          <tr style={{borderBottom: '1px solid #ddd'}}>
            <td style={{padding: '10px'}}>1</td>
            <td style={{padding: '10px', textAlign: 'right'}}>1.32</td>
            <td style={{padding: '10px', textAlign: 'right'}}>49.0%</td>
          </tr>
          <tr style={{borderBottom: '1px solid #ddd'}}>
            <td style={{padding: '10px'}}>2</td>
            <td style={{padding: '10px', textAlign: 'right'}}>1.81</td>
            <td style={{padding: '10px', textAlign: 'right'}}>52.9%</td>
          </tr>
          <tr style={{borderBottom: '1px solid #ddd'}}>
            <td style={{padding: '10px'}}>3</td>
            <td style={{padding: '10px', textAlign: 'right'}}>1.88</td>
            <td style={{padding: '10px', textAlign: 'right'}}>52.9%</td>
          </tr>
          <tr style={{borderBottom: '1px solid #ddd'}}>
            <td style={{padding: '10px'}}>4</td>
            <td style={{padding: '10px', textAlign: 'right'}}>0.99</td>
            <td style={{padding: '10px', textAlign: 'right'}}>53.3%</td>
          </tr>
          <tr style={{borderBottom: '1px solid #ddd'}}>
            <td style={{padding: '10px'}}>5</td>
            <td style={{padding: '10px', textAlign: 'right'}}>1.47</td>
            <td style={{padding: '10px', textAlign: 'right'}}>53.5%</td>
          </tr>
          <tr style={{fontWeight: 'bold', borderTop: '2px solid #333'}}>
            <td style={{padding: '10px'}}>Mean</td>
            <td style={{padding: '10px', textAlign: 'right'}}>1.49</td>
            <td style={{padding: '10px', textAlign: 'right'}}>51.7%</td>
          </tr>
        </tbody>
      </table>

      <h3>Understanding the Metrics</h3>
      <p>
        The competition used a sophisticated adjusted Sharpe ratio as its evaluation metric. To understand it, let's start with the standard Sharpe ratio and build up to the adjustments.
      </p>
      <p>
        The <strong>Sharpe ratio</strong> measures risk-adjusted returns—how much return you earn per unit of volatility:
      </p>
      <BlockMath math="\text{Sharpe} = \frac{\mu_{\text{strategy}} - r_f}{\sigma_{\text{strategy}}} \times \sqrt{252}" />
      <p>
        where <InlineMath math="\mu_{\text{strategy}}" /> is the strategy's mean daily return, <InlineMath math="r_f" /> is the risk-free rate, <InlineMath math="\sigma_{\text{strategy}}" /> is daily volatility, and the <InlineMath math="\sqrt{252}" /> factor annualizes the ratio (252 trading days per year).
      </p>
      <p>
        However, the competition added two penalty terms to discourage certain undesirable behaviors:
      </p>
      
      <h4>Volatility Penalty</h4>
      <p>
        The first penalty targets strategies that take on excessive volatility relative to the market. It's calculated as:
      </p>
      <BlockMath math="\text{Volatility Penalty} = 1 + \max\left(0, \frac{\sigma_{\text{strategy}}}{\sigma_{\text{market}}} - 1.2\right)" />
      <p>
        This allows your strategy to be up to 20% more volatile than the underlying market without penalty. But if you exceed that threshold—say through aggressive leverage or wild tactical swings—the penalty grows linearly with the excess volatility.
      </p>

      <h4>Return Penalty</h4>
      <p>
        The second penalty punishes strategies that underperform the market. If your strategy earns less than simply holding the index, you face a quadratic penalty:
      </p>
      <BlockMath math="\text{Return Penalty} = 1 + \frac{\max(0, \mu_{\text{market}} - \mu_{\text{strategy}})^2}{100}" />
      <p>
        The quadratic nature means small underperformance has minimal impact, but large gaps are severely penalized. This discourages overly conservative strategies that reduce risk but fail to capture market returns.
      </p>

      <h4>Final Adjusted Sharpe</h4>
      <p>
        The final score combines these penalties:
      </p>
      <BlockMath math="\text{Adjusted Sharpe} = \frac{\text{Sharpe}}{\text{Volatility Penalty} \times \text{Return Penalty}}" />
      <p>
        This metric balances multiple objectives: generate returns, manage risk, avoid excessive volatility, and don't underperform the market. It rewarded strategies that beat the market with controlled risk, while punishing both reckless volatility and overly timid allocations.
      </p>
      <p>
        My system achieved adjusted Sharpe ratios between 0.99 and 1.88 across validation folds, with a mean of 1.49. In the context of this metric, a ratio above 1.0 is considered strong, as it indicates not only positive risk-adjusted returns but also successful navigation of both penalty mechanisms. It's worth noting tht the "do nothing" baseline—simply holding the S&P 500 with a 100% allocation—yields an adjusted Sharpe of approximately 0.46 across the competition dataset.
      </p>
      <p>
        <strong>Directional accuracy</strong> of 51.7% might seem barely better than random (50%), but this is actually realistic and valuable in financial markets. Even a 51-52% directional edge, combined with proper position sizing and risk management, can generate substantial risk-adjusted returns. The key isn't being right most of the time—it's making more when you're right than you lose when you're wrong.
      </p>

      <h3>Key Insights</h3>
      <p>
        <strong>Consistency Across Regimes:</strong> The system performed well across all 5 validation folds, which represented different historical periods and market conditions. This suggests the approach generalizes rather than overfitting to specific market environments.
      </p>
      <p>
        <strong>Meta-Labeling Impact:</strong> While meta-labeling showed mixed results across folds (improvements ranging from -2.75% to +0.65%), it provides a robust framework for confidence-based position sizing that can be refined with additional meta-features or different confidence calibration approaches.
      </p>
      <p>
        <strong>Regime Adaptation:</strong> The performance variance across folds validated the regime-dependent position sizing. Markets behave differently under different conditions, and the system adapted accordingly.
      </p>

      <h2>Challenges and Lessons Learned</h2>
      
      <h3>The Obfuscation Challenge</h3>
      <p>
        Working with completely anonymized features was initially frustrating—I couldn't apply any financial domain knowledge. But it forced me to think more rigorously about statistical relationships and temporal patterns. In some ways, this constraint made me a better ML practitioner. I had to trust the data and the methodology rather than relying on intuition that might be wrong.
      </p>
      <p>
        The massive feature engineering effort (2,000+ features from 68 inputs) would have been impossible with manual feature selection. The quantitative approach—generating everything possible, then reducing intelligently—worked precisely because I couldn't cherry-pick based on financial theory.
      </p>

      <h3>Pipeline Complexity</h3>
      <p>
        As the system evolved, managing model dependencies became increasingly complex. The volatility model needed return predictions. The meta-model needed both. Each required different features and preprocessing. Keeping track of what depended on what, ensuring consistent data splits, and avoiding subtle forms of leakage required careful architecture.
      </p>
      <p>
        I learned to treat each model as a separate module with explicit input/output contracts. This modular design made debugging easier and allowed me to iterate on individual components without breaking the entire pipeline.
      </p>

      <h3>Label Leakage Prevention</h3>
      <p>
        Implementing purged cross-validation was crucial but tricky. It's easy to accidentally create temporal overlaps, especially with rolling features that look backwards. I had to carefully verify that no validation sample's label overlapped with any training sample's label, accounting for both the prediction horizon (1 day) and feature lookback windows (up to 126 days).
      </p>
      <p>
        The effort was worth it. Models that looked great with standard CV often degraded significantly with proper purged CV—revealing overfitting that would have failed in production.
      </p>

      <h2>Future Directions</h2>
      <p>
        While the current system performs well, several enhancements could improve it further:
      </p>

      <h3>Online Learning</h3>
      <p>
        The current models are trained on historical data and remain static. Markets evolve—relationships that held in the past may weaken or reverse. Online learning would allow the models to adapt continuously as new data arrives, potentially catching regime shifts earlier and maintaining performance as market dynamics change.
      </p>
      <p>
        This could be implemented through incremental updates to the gradient boosting models or by using algorithms specifically designed for online learning, such as online gradient descent or adaptive boosting methods.
      </p>

      <h3>Ensemble Methods</h3>
      <p>
        Currently, I use LightGBM for all prediction tasks. Combining multiple model types—XGBoost, neural networks, linear models—could capture different aspects of the data and improve robustness. Ensemble diversity often provides better out-of-sample performance than any single model. Ultimately, the pipeline complexity and competition deadlines discouraged me from pursuing this, but it's a promising avenue.
      </p>

      <h3>Enhanced Meta-Features</h3>
      <p>
        The meta-labeling framework showed promise but mixed results. Adding more sophisticated meta-features could help:
      </p>

      <h3>Multi-Asset Extension</h3>
      <p>
        The current system focuses on S&P 500 allocation. Extending it to multi-asset portfolio optimization—balancing stocks, bonds, commodities, currencies—would require portfolio-level constraints and correlation modeling but could significantly improve diversification and risk-adjusted returns.
      </p>

      <h3>Transaction Costs and Constraints</h3>
      <p>
        The competition didn't penalize trading frequency, but real-world trading involves costs: commissions, bid-ask spreads, market impact. Incorporating these into the allocation model would make it more production-ready, potentially by adding turnover penalties to the objective function or implementing minimum holding periods.
      </p>

      <h2>Reflections</h2>
      <p>
        This project taught me that good machine learning in finance isn't about finding a magic model—it's about rigorous methodology, careful handling of temporal data, and thoughtful separation of concerns. The obfuscated features forced me to develop a systematic, quantitative approach that I now apply even when domain knowledge is available.
      </p>
      <p>
        Reading chapters in "Advances in Financial Machine Learning" was transformative. Concepts like purged cross-validation, meta-labeling, and regime-dependent sizing turned what could have been a naive regression problem into a sophisticated decision-making framework. These techniques are now permanent parts of my ML toolkit.
      </p>
      <p>
        Perhaps most importantly, I learned to respect the difficulty of financial prediction. A 51.7% directional accuracy sounds unimpressive until you realize that even this modest edge, combined with proper position sizing and risk management, can generate Sharpe ratios above 1.0. The key isn't being right all the time—it's being systematically right often enough, and managing risk when you're wrong.
      </p>
      <p>
        Financial machine learning sits at the intersection of statistics, computer science, and economics. It's technically demanding, intellectually rich, and deeply practical. This project was my introduction to that world, and it's one I'm excited to explore further.
      </p>
    </PageTemplate>
  );
}
