# National-Quant-Finance-Olympiad-2026

# Implied Volatility Surface Reconstruction

### National Quant Finance Olympiad 2026

A quantitative finance project for reconstructing missing implied volatilities on a NIFTY50-style option surface.

## 📄 Methodology

The complete methodology, including the modelling approach, feature engineering, validation strategy, ensemble design, and arbitrage-aware post-processing, is provided in:

**[Methodology PDF](./methodology.pdf)**

The document covers:

* SVI Surface Calibration
* Market Regime Detection using K-Means
* Machine Learning Ensemble
* Feature Engineering
* Time-Ordered Validation
* Model Blending
* Put-Call Parity Adjustment
* Calendar Monotonicity Enforcement
* Validation Results
* Reproducibility Details

## Project Overview

The objective is to reconstruct missing implied volatility values from an option volatility surface where approximately 30–45% of the observations are missing.

The methodology combines a quantitative SVI model with machine-learning models and arbitrage-aware post-processing to produce the final volatility estimates.

## Methodology Pipeline

```text
Observed Option Data
        │
        ▼
SVI Surface Calibration
        │
        ▼
Market Regime Detection
        │
        ▼
Feature Engineering
        │
        ▼
ML Ensemble
 ┌──────┼──────────┐
 ▼      ▼          ▼
HistGBM ExtraTrees Ridge
 └──────┼──────────┘
        ▼
    Model Blend
        │
        ▼
Arbitrage-Aware
Post-Processing
        │
        ▼
Final IV Predictions
```

## Validation Results

| Model                | Validation RMSE |
| -------------------- | --------------: |
| SVI                  |           0.714 |
| Ridge                |           0.620 |
| ExtraTrees           |           0.635 |
| HistGradientBoosting |           0.565 |
| Blended Ensemble     |       **0.560** |

The validation methodology uses a time-ordered split, with earlier dates used for training and later dates used for validation.

## Repository Structure

```text
project/
│
├── methodology.pdf
├── solution.py
├── requirements.txt
└── README.md
```

## Technologies

* Python
* NumPy
* Pandas
* SciPy
* Scikit-learn

## Documentation

For the detailed technical explanation of the complete approach, see:

📄 **[Read the Methodology](./methodology.pdf)**

---

**National Quant Finance Olympiad 2026**

*Implied Volatility Surface Reconstruction*
