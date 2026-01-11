import React from 'react';
import 'katex/dist/katex.min.css';
import { BlockMath, InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';


import PageTemplate from '../components/PageTemplate';
import dementiaImage from '../images/dementia.jpeg';

import graphicalAbstract from '../images/dementia/graphical-abstract.png';
import preprocessingDiagram from '../images/dementia/preprocessing-pipeline.png';
import silenceComparison from '../images/dementia/silence-comparison.png';
import pipelineDiagram from '../images/dementia/pipeline-diagram.png';
import classDistributionChart from '../images/dementia/class-distribution.png';
import confusionMatrix from '../images/dementia/confusion-matrix.png';
import featureImportanceChart from '../images/dementia/feature-importance.png';
import modelComparisonChart from '../images/dementia/model-comparison.png';
import dementiaUI from '../images/dementia/ui.png';

export default function DementiaPage() {
  return (
    <PageTemplate title="Detecting Alzheimer's from Voice Recordings" image={dementiaImage}>
      {/* 1. Introduction */}
      <p>
  This was my very first research project—and it shaped how I think about AI, medicine, and what good machine learning can do for the world. It started in the first week of my third semester, when I ran into an old friend who mentioned that a biomedical research team was looking for someone with data science skills. As it happened, I was actively searching for a research project where I could work with Machine Learning.
</p>
<p>
  The project sounded fascinating: build a machine learning system to detect Alzheimer's Dementia from simple voice recordings. Not only was it technically rich, but it also had the potential for real-world clinical impact. Within a few weeks, the team was reduced to just me and the professor who was mentoring me. I took over end-to-end responsibility: audio processing, feature engineering, modeling, analysis, and writing.
</p>
<p>
  Humans can often perceive when someone’s voice changes due to cognitive decline—there’s something subtle in the rhythm, tone, or clarity that gives it away. That means there are real, quantifiable differences in the signal. Machine learning models excel at detecting such patterns, and in many cases, they can outperform humans in doing so. This made the idea of an automatic detection system not only feasible, but extremely promising.
</p>

      <h2>Graphical Abstract</h2>
      <p>
      Below is <a href="/dementiaDetectionML.pdf" target="_blank" rel="noopener noreferrer">the paper's</a> graphical abstract, which summarizes our approach.
      </p>
      <img src={graphicalAbstract} alt="Graphical Abstract" className="page-image" />

      {/* 2. Why This Matters */}
      <h2>Why This Matters</h2>
      <p>
        Dementia, including Alzheimer's disease, is one of the greatest public health challenges of our time. Today, more than 55 million people worldwide live with dementia—and that number is expected to rise to over 150 million by 2050 (or around 1 in 65!). This surge is driven mostly by aging populations and shifting demographics across the globe.
      </p>
      <p>
        Despite its growing impact, dementia still has no cure. But early diagnosis can make a huge difference. It allows patients to start treatments that slow the disease's progression, gives families more time to plan and adapt, and improves access to support services and clinical trials. 
      </p>
      <p>
        Unfortunately, current diagnostic tools are far from ideal. Many rely on expensive neuroimaging techniques, like MRI or PET scans, which require specialized equipment and trained personnel. Others depend on invasive procedures like lumbar punctures, which extract cerebrospinal fluid from the spine to look for biomarkers. These methods are not only costly and uncomfortable—they're also inaccessible to millions of people worldwide, especially in low-resource settings.
      </p>
      <p>
        That's where this kind of work comes in. Voice is a natural, familiar, and non-invasive signal—something most people can provide with nothing more than a phone or microphone. By analyzing subtle changes in speech patterns, we can potentially detect Alzheimer's years before traditional symptoms become obvious. And because it's fast, low-cost, and scalable, voice-based screening could help bring early diagnosis to the people and places that need it most.
      </p>

      <p>
        In a world where dementia is rising fast, making early detection widely accessible is not just a technical challenge—it's a moral one. With this motivation in mind, let's explore the technical approach I took.
      </p>

      {/* 3. Dataset */}
      <h2>Dataset</h2>
      <p>
        For this study, I used the Ivanova Spanish-language dataset from DementiaBank, which contains 361 audio recordings of adults reading a fixed paragraph. Of these recordings, 74 were from participants with Alzheimer's Disease, while 287 were from healthy controls or those with Mild Cognitive Impairment (MCI). This made it one of the few high-quality Spanish datasets available for this task.
      </p>

      {/* 4. Preprocessing */}
      <h2>Preprocessing</h2>
      <p>
        Before I could extract any meaningful features, I had to clean the audio recordings thoroughly. Raw voice data is often messy—filled with background noise, inconsistent volumes, and long pauses.
      </p>
      <p>
        I designed a multi-step preprocessing pipeline to ensure the quality and comparability of all recordings. The process began by loading the raw audio and converting it to mono to simplify analysis. Then, I resampled everything to a uniform 16 kHz sampling rate to standardize time resolution.
      </p>
      <p>
        Next, I applied amplitude normalization to scale all recordings to the same average volume level, followed by noise reduction using spectral subtraction and Wiener filtering. To handle occasional spikes—like sudden microphone bumps—I implemented peak smoothing using a statistical threshold (so I removed any peaks that were some standard deviations above mean energy).
      </p>
      <p>
        Finally, I applied voice activity detection (VAD) to segment actual speech from silence or irrelevant noise. This allowed me to isolate only the parts of the audio that contained human speech, which helped a lot on the feature engineering phase.
      </p>

      <img src={preprocessingDiagram} alt="Audio Preprocessing" className="page-image" />

      {/* 5. Feature Engineering */}
      <h2>Feature Engineering: Listening for Cognitive Decline</h2>
      <p>
        Like any beginner, my first approach was a little naive. I tried feeding the raw audio files directly into a machine learning model, hoping it could "just learn" the difference. Unsurprisingly, this didn't work—training was slow, and accuracy hovered near chance levels. That's when I turned to the literature and had conversations with my advisor, which led me to a far more robust strategy: feature engineering.
      </p>
      <p>
        Instead of working with raw waveform data, I would extract a rich set of numerical features from the audio and train the model on that. This switch made all the difference. Let's explore the features that eventually helped us detect Alzheimer's on the audios.
      </p>

      <h3>Acoustic Features</h3>
      <p>
        The acoustic domain provided rich indicators of vocal quality, which often deteriorates with neurodegeneration:
      </p>
      <ul>
        <li><strong>MFCCs (Mel-Frequency Cepstral Coefficients)</strong>: These coefficients describe the short-term power spectrum of sound, using a perceptual scale that mimics how the human ear works. Each MFCC captures different aspects of vocal quality. They're computed through a multi-step process: windowing the signal, applying the Fourier Transform to extract the frequency spectrum, mapping frequencies to the Mel scale using <InlineMath math="m(f) = 2595 \cdot \log_{10} \left(1 + \frac{f}{700} \right)" />, taking the log amplitude, and finally applying the Discrete Cosine Transform (DCT).

        <p>Each frame produces a vector of 13 coefficients, which I aggregated across time using statistical descriptors like mean and standard deviation. This yielded 26 final features: mfcc_1_mean, mfcc_1_std through mfcc_13_mean, mfcc_13_std.</p></li>
        <li><strong>Formants F1–F4</strong>: Formants are resonance frequencies of the vocal tract that define vowel quality. They are calculated from peaks in the spectral envelope of speech. I used Parselmouth (a Python interface for Praat) to extract them, computing the mean and range of F1–F4 across time. Their variation reflects articulation characteristics that often degrade in Alzheimer's, helping quantify slurring, imprecise articulation, and motor symptoms common in neurodegenerative speech.</li>
        <li><strong>Spectral centroid & slope</strong>: The spectral centroid represents the center of mass of the sound spectrum, often referred to as the "brightness" of the sound. The spectral slope captures the tilt or decay of the spectrum. Alzheimer's can subtly shift speech energy distribution, which is captured by these frequency-domain statistics.</li>
        <li><strong>Voice quality metrics</strong>: I extracted jitter, shimmer, cepstral peak prominence (CPPS), and harmonics-to-noise ratio (HNR) using Praat through Parselmouth. These features reflect micro-instabilities and noise in speech that increase with vocal deterioration. Lower HNR and CPPS values, or higher jitter/shimmer, can all signal impaired vocal fold control or neuromotor irregularity.</li>
      </ul>

      <h3>Temporal Features</h3>
      <p>
        When listening to recordings of people with Alzheimer's, one of the most noticeable differences was in their rhythm and timing. Hesitations, prolonged pauses, and irregular speech cadence were common. This led me to explore temporal features that could quantify these changes.
      </p>
      <p>
        I used voice activity detection (VAD) to segment the audio signal into speech and non-speech regions. This enabled the calculation of several timing-related features that are highly relevant in Alzheimer's detection, particularly given that the audios consisted of participants reading the same paragraph of text.
      </p>
      <p>
        I implemented a hybrid energy threshold and silence duration method to detect pauses longer than 0.5 seconds. The algorithm analyzes the audio signal frame by frame, classifying each segment as either speech or silence based on energy levels and voice activity patterns. From this segmentation, I computed several key metrics:
      </p>
      <ul>
        <li><strong>Total duration</strong> (how long did the subject take to read the text)</li>
        <li><strong>Total speech duration</strong> (sum of active speech segments)</li>
        <li><strong>Number of pauses</strong></li>
        <li><strong>Max / mean / std pause duration</strong></li>
        <li><strong>Speech-to-pause ratio</strong></li>
        <li><strong>Articulation rate</strong> (syllables per second excluding pauses)</li>
      </ul>
      <p>
        Additionally, I extracted rhythm features from phonetic literature:
      </p>
      <ul>
        <li>
          <strong>nPVI (Normalized Pairwise Variability Index):</strong>
          <br />
          Measures variability in successive speech durations (e.g., syllables):
          <BlockMath math={'\\text{nPVI} = \\frac{100}{N - 1} \\sum_{k=1}^{N-1} \\frac{|d_k - d_{k+1}|}{(d_k + d_{k+1}) / 2}'} />
        </li>
      </ul>
      <p>
  Where:
  <ul>
    <li><code>N</code> is the number of speech intervals (e.g., syllables).</li>
    <li><code>dₖ</code> is the duration of the <em>k</em>-th interval.</li>
    <li>The formula computes the average relative difference between adjacent durations, scaled by 100.</li>
  </ul>
  A higher nPVI means more rhythmic variability, often associated with stress-timed languages like English. Lower values are typical of syllable-timed languages like Spanish.
</p>

      
      <p>
        The differences in pause patterns between healthy controls and those with Alzheimer's were striking:
      </p>
      <img src={silenceComparison} alt="AD vs HC speech segmentation" className="page-image" />

      <p>
      In the image we can see two audio waveforms: one from a patient with Alzheimer’s (top), and one from a healthy control (bottom). Green regions show detected voice, red regions are silences, and white gaps were too short to classify (we only care about long enough silences, not micro-pauses). Notice how the AD subject has way more pauses and scattered speech, while the healthy speaker talks more fluently. This kind of difference is what I wanted to capture with the features.
    </p>



      <h3>Complexity Features</h3>
      <p>
        Perhaps the most conceptually fascinating features came from analyzing the complexity and fractal nature of speech waveforms:
      </p>
      <p>
        One of the most interesting discoveries I came across in the literature was Higuchi's Fractal Dimension (HFD). This method estimates the complexity of a waveform by evaluating its fractal geometry — essentially, how "rough" or irregular a signal is over different scales.
      </p>
      <p>
        The basic idea is to estimate the fractal dimension <InlineMath math="D" /> of the waveform, which lies between 1 (a smooth curve) and 2 (a highly jagged, noisy curve). Higuchi's Fractal Dimension (HFD) quantifies the complexity of a time series like speech: higher values indicate more irregular, unpredictable patterns.
      </p>

      <p>
        To compute it, we calculate the total curve length <InlineMath math="L(k)" /> for different time intervals <InlineMath math="k" />, and then fit a line to the relationship in log-log space:
      </p>

      <BlockMath math={'\\log L(k) = -D \\cdot \\log k + C'} />

      <p>
        I extracted statistics like <code>HFD_mean</code>, <code>HFD_min</code>, and <code>HFD_std</code> across the cleaned waveform.
      </p>
      
      <p>
        Together, these acoustic, temporal, and complexity features gave us a multidimensional view of speech changes associated with Alzheimer's Disease. Now, let's see how I integrated them into the classification system.
      </p>

      {/* 6. System Overview */}
      <h2>System Overview: From Audio to Interpretability</h2>
      <p>
        After preprocessing, the cleaned audio recordings were ready for feature extraction and classification. I built a modular, end-to-end pipeline to handle everything from raw features to evaluation and visualization.
      </p>
      <p>
        The pipeline begins with feature and metadata extraction, where I computed both acoustic features (like MFCCs, jitter, shimmer) and temporal features (such as pause duration and speech rate), along with metadata like patient age. Because the dataset was highly imbalanced—with far fewer Alzheimer's cases than controls—I applied SMOTE-ENN for balancing.
      </p>
      <p>
        Next came feature selection using recursive feature elimination (RFE), which helped reduce dimensionality and eliminate noisy or redundant inputs. I then trained a support vector machine (SVM) classifier and used grid search to tune its hyperparameters, such as the regularization strength and kernel type, for optimal performance.
      </p>
      <p>
        Finally, I evaluated the model using k-fold cross-validation, which allowed me to assess generalization by training and testing across different partitions of the dataset. I also generated performance reports and visualizations to better interpret the model's behavior and outcomes.
      </p>
      <img src={pipelineDiagram} alt="Machine Learning Pipeline" className="page-image" />

      {/* 7. Modeling and Evaluation */}
      <h2>Modeling and Evaluation</h2>
      
      <h3>Class Balancing with SMOTE-ENN</h3>
      <p>
        The first challenge was addressing the dataset's inherent imbalance—Alzheimer's recordings made up only about 20% of the dataset. Imbalanced datasets can lead to biased models that prioritize the majority class at the expense of the minority class. To counter this, I used a hybrid method known as Synthetic Minority Oversampling Technique with Edited Nearest Neighbors (SMOTE-ENN), which combines oversampling and data cleaning.
      </p>
      <h4>SMOTE</h4>
      <p>
        SMOTE generates new synthetic minority samples by interpolating between existing ones. Given an AD sample <InlineMath math="x" /> and one of its nearest neighbors <InlineMath math="x_{NN}" />, a new sample <InlineMath math="x_{new}" /> is created as:
      </p>
      <BlockMath math={'x_{\\text{new}} = x + \\delta \\cdot (x_{\\text{NN}} - x), \\quad \\delta \\sim \\mathcal{U}(0, 1)'} />
      <p>
        This increases the number of AD cases without simply duplicating them.
      </p>
      <h4>ENN (Edited Nearest Neighbors)</h4>
      <p>
        After oversampling, ENN removes ambiguous or noisy samples. If a point is misclassified by its 3 nearest neighbors, it is dropped. This helps eliminate class overlaps and mislabeled data. This dual strategy produced a dataset that was both balanced and clean, dramatically improving model generalization and reducing false positives.
      </p>
      <img src={classDistributionChart} alt="Class Distribution Before/After" className="page-image" />
      
      <h3>Feature Selection</h3>
      <p>
        Having extracted more than 80 features, I then had to reduce dimensionality for generalization and interpretability. I used Recursive Feature Elimination (RFE) with a Random Forest classifier as the estimator. RFE works by recursively fitting the model, ranking features by importance, and removing the least important ones until a target number is reached. After extensive experimentation, this yielded a subset of 15 features that balanced predictive power with model simplicity.
      </p>
      <ul>
        <li><strong>Acoustic (MFCC): </strong>mfcc_2_mean, mfcc_3_mean, mfcc_5_mean, mfcc_6_mean, mfcc_7_mean/std, mfcc_8_mean, mfcc_11_mean</li>
        <li><strong>Acoustic (others): </strong>F2_range, spectral_centroid, hnr_mean</li>
        <li><strong>Temporal: </strong>total_duration, total_speech_duration, speech_duration_CV</li>
        <li><strong>Complexity: </strong>HFD_min</li>
      </ul>

      <h3>Model Training</h3>
      <p>
        With our balanced dataset and optimized feature set ready, I moved on to training several classification models. I trained a Support Vector Machine (SVM) using the selected features. SVMs are well-suited for small, high-dimensional datasets and allow non-linear classification via kernels. I used an RBF kernel with hyperparameter tuning over regularization strength <InlineMath math="C" /> and the kernel coefficient <InlineMath math="\gamma" />. The best model had <InlineMath math="C=10" /> and <InlineMath math="\gamma" /> set to 'scale'. Cross-validation and full evaluation showed strong performance, especially considering the initial class imbalance.
      </p>
      
      <p>
        To verify that the selected features generalized well and weren't overly optimized for the SVM architecture, I also trained and tuned two additional classifiers: an Artificial Neural Network (ANN) and an XGBoost model. Both were tested using the same 15 selected features.
      </p>
      <h4>Artificial Neural Network (ANN)</h4>
      <p>
        I implemented a shallow fully connected feedforward network with one hidden layer of 64 neurons, using ReLU activation and L2 regularization. Despite the small sample size, the ANN achieved an F1 score of 0.79—competitive with the SVM and indicative of strong signal in the features.
      </p>
      <h4>XGBoost</h4>
      <p>
        XGBoost is an ensemble of decision trees trained using gradient boosting. I tuned key hyperparameters including tree depth, learning rate, and the number of estimators to optimize performance while avoiding overfitting.
      </p>

      <h3>Performance & Feature Importance</h3>

      <p>
        All three models showed promising results, with the SVM slightly outperforming the others:
      </p>
      <ul>
        <li><strong>SVM:</strong> F1 = 0.81, Accuracy = 0.80</li>
        <li><strong>ANN:</strong> F1 = 0.79, Accuracy = 0.78</li>
        <li><strong>XGBoost:</strong> F1 = 0.77, Accuracy = 0.75</li>
      </ul>

      <img src={modelComparisonChart} alt="Model Comparison Chart" className="page-image" />
     
      <p>
        The consistency in performance across SVM, ANN, and XGBoost confirmed the reliability and discriminative power of our 15-feature subset, suggesting that we had indeed captured meaningful audio biomarkers of Alzheimer's Disease.
      </p>

      <p>
        Fitting the model into the entire dataset, the SVM reached 86% accuracy and 87% weighted F1 score.
      </p>
      <img src={confusionMatrix} alt="Confusion Matrix" className="page-image" />
      

      <h2>Interpretability</h2>
      <p>
        To interpret the model's decisions and better understand what was driving its predictions, I used permutation feature importance. This technique measures the increase in model error when a single feature's values are randomly shuffled. For each feature, the importance is computed as the difference between the model's performance on the original data versus the shuffled version. Features that, when shuffled, cause a large drop in performance are considered highly informative.
      </p>

      <p>
        Top-ranked features included <code>mfcc_8_mean</code>, <code>F2_range</code>, <code>mfcc_2_mean</code>, and <code>spectral_centroid</code>. These results aligned well with what we saw across multiple classifiers, reinforcing the interpretability and robustness of the selected set.
      </p>
      <img src={featureImportanceChart} alt="Feature Importance" className="page-image" />
      
      <p>
        In the image above, each blue bar represents how important a feature was for the SVM model — specifically, how much the model’s performance dropped when that feature was randomly shuffled. Longer bars mean the feature was more critical. The black lines are error bars showing variability across different shuffles, so smaller black bars mean more consistent importance. As we can see, features like <code>mfcc_8_mean</code> and <code>F2_range</code> stood out as especially reliable indicators.
      </p>

            <p>
        With the model trained and evaluated, I wanted to make it accessible beyond the research setting. That’s where the web application came in—a practical step toward real-world use.
      </p>

      <h2>Bringing It to Life: Web Application for Real-World Use</h2>
      <p>
        To make the model more usable by clinicians, researchers, or even individuals, I built a webapp using React for the frontend and Flask for the backend. While this wasn’t part of the original research project, it was a natural and important extension—turning a promising model into a functional tool.
      </p>
      <p>
        The app lets users either record their voice directly or upload an existing audio file. Once submitted, the backend runs the full preprocessing pipeline: it normalizes the audio, denoises it, performs voice activity detection, extracts all the relevant features, and feeds them to the trained SVM model.
      </p>
      <p>
        The user receives a prediction within seconds. The interface is mobile-friendly and designed for simplicity—no technical background needed. My hope is that, in the future, such tools can support healthcare providers or even be offered as a preliminary self-assessment tool.
      </p>

      <img src={dementiaUI} alt="Web Application Screenshot" className="page-image" />

      {/* 8. Reflections and Future Work */}
      <h2>Reflections and Future Work</h2>
      <p>
        This was more than just a machine learning project — it was my first immersion into real research. I learned to navigate the ambiguity that comes with working on open-ended problems, to build complete pipelines from raw, noisy data to interpretable results, and to balance technical rigor with creativity.
      </p>
      <p>
        I developed skills in signal processing, statistical analysis, and applied machine learning, while also gaining a deeper appreciation for how data-driven tools can support healthcare. I discovered just how much information can be hidden in something as human as voice.
      </p>
      <p>
        We're collaborating with Mexican institutions to gather more diverse data — crucial for testing cross-population generalization. Ultimately, my goal is to help build non-invasive, low-cost tools that expand early dementia screening to places where traditional diagnostics are hard to access.
      </p>
    </PageTemplate>
  );
}