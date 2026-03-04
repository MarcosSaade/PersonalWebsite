# Projects Portfolio

A comprehensive overview of my technical projects spanning machine learning, software engineering, computational modeling, and AI research.

---

## 🧠 Machine Learning & AI Research

### Detecting Alzheimer's from Voice Recordings
**Technologies:** Python, Scikit-learn, OpenCV, Praat/Parselmouth, Signal Processing

My first research project where I developed a machine learning system to detect Alzheimer's Disease from simple voice recordings using the Spanish-language Ivanova dataset from DementiaBank (361 recordings: 74 AD, 287 healthy/MCI).

**Key Achievements:**
- Designed a comprehensive audio preprocessing pipeline with noise reduction, amplitude normalization, and voice activity detection (VAD)
- Engineered 100+ features across three domains:
  - **Acoustic:** MFCCs, formants (F1-F4), spectral centroid/slope, jitter, shimmer, CPPS, and HNR
  - **Temporal:** pause patterns, articulation rate, speech-to-pause ratio, and nPVI (rhythm variability)
  - **Complexity:** Higuchi's Fractal Dimension to quantify waveform irregularity
- Implemented SMOTE-ENN for handling class imbalance
- Applied purged k-fold cross-validation to prevent data leakage
- Achieved strong classification performance using SVM with recursive feature elimination
- Published findings demonstrating early detection potential for a non-invasive, low-cost diagnostic tool

**Impact:** This work addresses a critical need for accessible dementia screening, particularly in low-resource settings where expensive neuroimaging is unavailable.

---

### Predicting S&P 500 Allocations with Machine Learning
**Technologies:** Python, XGBoost, Random Forest, Pandas, NumPy, Time Series Analysis

A quantitative finance project from the Hull Tactical Market Prediction Kaggle competition, where I predicted optimal daily S&P 500 allocations (0-200%) using 68 obfuscated features across economic, market, monetary, price, sentiment, and volatility categories.

**Key Achievements:**
- Engineered 2,000+ features from 68 raw inputs using rolling statistics, volatility measures, and regime detection
- Implemented three-stage dimensionality reduction: correlation clustering → PCA per cluster → feature importance selection
- Built a modular pipeline with specialized models:
  - Return prediction model (expected μ)
  - Volatility prediction model (forecast uncertainty σ)
  - Meta-labeling model (confidence estimation)
- Applied purged k-fold cross-validation to prevent look-ahead bias in time series
- Implemented regime-dependent position sizing based on López de Prado's "Advances in Financial Machine Learning"
- Achieved consistent outperformance over buy-and-hold baseline with improved Sharpe ratios

**Technical Highlights:**
- Handled missing values (some features 70%+ missing) with availability masks and forward-filling
- Used Median Absolute Deviation (MAD) with feature-specific winsorization thresholds for outlier detection
- Combined multiple predictions into risk-managed positions based on market volatility regimes

---

### SugarZero: Teaching AI to Master a Simple Game through Self-Play
**Technologies:** Python, PyTorch, NumPy, Monte Carlo Tree Search (MCTS), Reinforcement Learning

Inspired by AlphaZero, I built a self-learning AI for the abstract strategy game "Sugar" (3×3 board with stackable pieces) that learns optimal play through pure self-play, starting only with the game rules.

**Key Achievements:**
- Implemented Monte Carlo Tree Search with PUCT (Predictor + Upper Confidence bounds for Trees) formula to balance exploration-exploitation
- Designed convolutional neural network architecture with:
  - Shared convolutional layers for spatial pattern recognition
  - Separate policy head (move probabilities) and value head (position evaluation)
  - Batch normalization and dropout for stable training
- Created efficient game state representation with fast cloning for MCTS simulations
- Built complete self-play training loop generating synthetic training data
- Achieved superhuman performance through iterative improvement

**Technical Insights:**
- The PUCT formula elegantly solves the multi-armed bandit problem by combining exploitation (Q-values) with exploration (priors and visit counts)
- Convolutional layers capture spatial patterns better than fully-connected networks for board games
- Self-play creates a curriculum where the AI continually faces appropriately challenging opponents

---

## 💻 Full-Stack Software Development

### BanorTeach: Personalized Financial Literacy Powered by AI
**Technologies:** React Native, Node.js, Express, Google Cloud Vertex AI (Gemini API), PostgreSQL

Built during Hackathon Banorte 2024, BanorTeach is a mobile app delivering personalized financial education using AI-generated interactive stories. I developed the entire platform (frontend, backend, AI integration) while teammates supported design and pitch preparation.

**Key Features:**
- **Personalized Learning:** Content adapted to user profile (age, occupation, financial goals)
- **AI-Generated Scenarios:** Interactive choose-your-own-adventure stories with branching outcomes based on financial decisions
- **Test-Out Quizzes:** Users can skip modules they're confident in
- **Goal Tracking:** Monitor progress toward concrete financial objectives
- **Gamification:** Points, achievements, and progress tracking

**Technical Architecture:**
- React Native frontend for cross-platform mobile experience
- Node.js/Express backend handling user management and progress storage
- Google Cloud Vertex AI integration with carefully engineered prompts for pedagogically sound, realistic financial scenarios
- Modular architecture enabling easy expansion to new topics

**Vision:** Could be offered by banks as a free app with redeemable rewards, creating value for customers while encouraging better financial behavior.

---

### NeuroCapture: Desktop App for Multimodal Neurological Studies
**Technologies:** Electron, React, FastAPI, PostgreSQL, Alembic, Tailwind CSS, Audio Processing

A full-stack desktop application for collecting rich, longitudinal, multimodal datasets to power dementia research. Designed for busy clinics with a focus on data integrity and usability.

**Core Functionality:**
- **Patient & Study Management:** Create, search, edit participants with unique study IDs; CSV export with referential integrity
- **Cognitive Assessments:** Research-grade entry for MMSE/MoCA with per-item subscores, validation, and clinical notes
- **Speech Capture:** Record audio in-app or attach existing files (WAV/FLAC) with checksums and normalization
- **One-Click Export:** Tidy CSVs joining demographics, assessments, and 150+ engineered speech features

**Technical Architecture:**
- Electron + React UI with Tailwind styling
- FastAPI backend with versioned, self-documenting API (Swagger/ReDoc)
- PostgreSQL database with Alembic migrations
- Transactional writes and queued background jobs for resilience
- Automated audio pipeline: noise reduction, normalization, VAD, feature extraction (prosodic, spectral, MFCCs, jitter/shimmer, formants)

**Design Principles:**
- Clinic-friendly UX with fast keyboard workflows
- Reproducibility through deterministic preprocessing
- Privacy with participant IDs and optional PHI redaction
- Extensibility for new modalities via typed tables

---

### Orto-Flex Scanner: Clinical Image Processing for Orthopedics
**Technologies:** Python, OpenCV, Pygame, WIA Interface, Computer Vision, PDF Generation

A production-grade desktop application for orthopedics clinics, digitizing patient evaluation workflows with hardware-integrated scanning and automated report generation. Currently serves ~4,000 patients/year across two clinics.

**Problem Solved:**
- Eliminated inefficient manual processes (generic scanner software, hand-drawn measurements)
- Reduced patient wait times during peak hours
- Professionalized reports for better patient and referral partner experience

**Computer Vision Pipeline:**
- **Image Acquisition:** 300 DPI scanning via WIA interface with orientation correction
- **Background Removal:** HSV color space segmentation with Gaussian blur, morphological closing, and contour detection
- **Convex Hull:** Shape regularization for biological plausibility
- **Feature Extraction:** Anatomical landmarks (heel center, toe tip), metatarsal width measurement
- **Diagnostic Heatmaps:** Pressure point visualization

**Technical Details:**
- Implemented Gaussian blur for noise reduction: `G(x,y) = (1/2πσ²)e^(-(x²+y²)/2σ²)`
- Applied morphological closing: `A•B = (A⊕B)⊖B` to fill gaps
- Automated PDF report generation with clinic branding
- Hardware abstraction layer supporting multiple scanner models
- Configuration panel for staff preferences

---

## 🔬 Computational Physics & Modeling

### Innovative Electric Field-Based Solar Panel Cleaning System
**Technologies:** MATLAB, Numerical Methods, Electrostatics, Computational Physics

A computational simulation for a sustainable solar panel cleaning system designed for desert environments like the Sahara, using electric fields to remove sand and dust with minimal water usage.

**Physical Challenge:**
- Desert solar panels lose 10-40% efficiency within weeks due to dust
- Traditional water-based cleaning is unsustainable in water-scarce regions
- Mechanical brushing damages panels over time

**Solution Design:**
- Negatively charged bars above panel surface
- Positively charged bars creating opposing electric field
- Controlled rotation sweeping field across entire panel
- Leverages natural electrostatic charge on desert sand particles

**Mathematical Framework:**
- **Electric Field Calculation:** 3D implementation of Coulomb's law: `E⃗(r⃗) = (1/4πε₀)(q/|r⃗|³)r⃗`
- **Force Analysis:** Net force `F⃗ₙₑₜ = qE⃗ + mg⃗` combining electric and gravitational forces
- **Numerical Integration:** Euler's method for particle trajectory simulation
- **Critical Lift Threshold:** `Eₘᵢₙ = mg/q ≈ 1.23×10⁶ N/C` to overcome gravity
- **Bar Optimization:** Derived optimal length `L = (1/2)√(b²+h²)` for complete coverage

**Simulation Details:**
- 1,000 charge points (10 rings × 100 points) modeling spinning bars
- 50-second simulations with 1,000 time steps
- Particles removed when reaching 0.1m height
- Vector field visualization in 2D and 3D

**Results:** Demonstrated feasibility of water-free cleaning using strategically positioned electric fields optimized through computational modeling.

---

### Modeling Epidemics: The SIR Dynamic System
**Technologies:** Python, Pygame, Differential Equations, Agent-Based Modeling, Numerical Analysis

Explored variations of the classic SIR (Susceptible-Infected-Recovered) epidemiological model to analyze disease spread under different assumptions including population dynamics, vaccination, and spatial effects.

**Mathematical Models:**
- **Classical SIR:** Three coupled differential equations capturing disease transmission
- **Dynamic Population:** Extended with birth rate `b` and death rate `μ` for endemic equilibria
- **Vaccination:** Analyzed herd immunity threshold: `v > 1 - 1/R₀`

**Key Findings:**
- **Basic Reproduction Number:** `R₀ = β/γ` determines epidemic occurrence
- **Infection Peak:** Occurs when `S = γN/β` (new infections balance recoveries)
- **Herd Immunity:** For `R₀ = 4`, requires 75% vaccination coverage
- **Early Growth Phase:** Time to halve susceptible population: `t = (N ln 2)/(βI₀)`

**Agent-Based Simulation:**
- Implemented spatial dynamics in Pygame with moving particles
- Color-coded states (blue=susceptible, red=infected, gray=recovered)
- Central attraction simulating urban centers (stores, schools)
- Distance-based infection transmission
- Probabilistic recovery

**Insights:**
- Urban centers accelerate disease spread through concentrated contact
- Spatial heterogeneity affects epidemic timing and magnitude
- Agent-based models complement differential equation approaches by capturing local interactions

---

## 🎮 Game Development & Algorithms

### Building a Chess Engine with Minimax and Alpha-Beta Pruning
**Technologies:** Python, Pygame, Minimax Algorithm, Alpha-Beta Pruning, Pickle

My first serious programming project (summer after high school): a fully functional chess engine supporting all standard rules with AI opponent using Minimax search and Alpha-Beta pruning.

**Core Features:**
- **Board Representation:** 8×8 2D list with string notation (e.g., "wP" for white pawn)
- **Move Generation:** Custom logic for each piece type (pawns, knights, bishops, rooks, queens, kings)
- **Special Rules:** Castling, pawn promotion, checkmate detection
- **Opening Book:** Hardcoded popular openings (Sicilian, Italian Game, London System) indexed by FEN strings

**AI Implementation:**
- **Minimax Algorithm:** Recursive decision-making simulating all moves up to depth limit
- **Alpha-Beta Pruning:** Dramatically reduces search space by pruning branches that can't improve outcome
  - Alpha: best score for maximizing player
  - Beta: best score for minimizing player
  - Prune when `β ≤ α`
- **Evaluation Function:** Combines material count with heuristics:
  - Piece development (centralizing pieces)
  - Pawn structure (penalizing doubled/isolated pawns)
  - Castling safety
  - Rooks on open files
  - King safety

**Technical Optimizations:**
- **Move Ordering:** Prioritizes promising moves for better pruning
- **Persistent Memory:** Stores strong moves using Pickle for reuse in future games
- **UI:** Pygame-based interactive chessboard with click-based input

**Personal Impact:** Solidified Python skills and built confidence for tackling more complex projects. The engine plays decent games and taught me fundamental AI search algorithms.

---

## 🏥 Healthcare Technology

### Clinical Applications
Multiple projects focused on healthcare technology:

1. **Dementia Detection System** - Voice-based early diagnosis
2. **NeuroCapture Platform** - Multimodal data collection for neurological research
3. **Orto-Flex Scanner** - Computer vision for orthopedics workflow automation

These projects demonstrate my commitment to leveraging technology for meaningful healthcare impact, from research tools to production systems serving thousands of patients.

---

## Skills Summary

**Languages:** Python, JavaScript, MATLAB, SQL

**ML/AI:** PyTorch, Scikit-learn, XGBoost, Random Forest, Neural Networks, Reinforcement Learning, MCTS

**Data Science:** Pandas, NumPy, Signal Processing, Time Series Analysis, Feature Engineering

**Computer Vision:** OpenCV, Image Processing, Morphological Operations

**Web/Mobile:** React, React Native, Node.js, Express, FastAPI, Electron

**Databases:** PostgreSQL, SQL, Alembic

**Scientific Computing:** Numerical Methods, Differential Equations, Physics Simulation

**Algorithms:** Minimax, Alpha-Beta Pruning, Search Algorithms, Optimization

**Tools:** Git, VS Code, Jupyter, Kaggle

---

## Project Themes

My projects reflect several consistent themes:

1. **Healthcare Innovation:** Multiple projects addressing real medical challenges through technology
2. **Machine Learning Rigor:** Strong focus on preventing overfitting, data leakage, and maintaining scientific validity
3. **End-to-End Development:** Comfortable building complete systems from data processing to user interfaces
4. **Mathematical Foundations:** Deep understanding of underlying mathematical principles in ML and physics
5. **Real-World Impact:** Projects deployed in production serving thousands of users

---

## Contact

- **GitHub:** [MarcosSaade](https://github.com/MarcosSaade)
- **Email:** Available on website

For detailed technical writeups, code samples, and visualizations, visit the individual project pages on my website.
