Markdown
# QuantModeller: Systematic Trading Research Framework.

### **A Technical Workbench for Strategy Development and ML Validation**

## **Overview**
**QuantModeller** is a Python-based research environment designed to bridge the gap between raw financial data and validated trading logic. It focuses on the **engineering challenges** of quantitative finance: handling temporal data leakage, managing high-dimensional feature sets, and ensuring reproducible ML experiments.

This project was built to explore the integration of Large Language Models (LLMs) in the strategy ideation phase and to automate the boilerplate code required for rigorous backtesting.

## **Key Engineering Features**

* **Modular Strategy Schema**: Built on `Pydantic` to ensure strict data validation for trading rules, entry/exit logic, and universe selection.
* **Leakage-Resistant Pipeline**: Implements **Purged Cross-Validation** and **Walk-Forward Analysis** to ensure that models do not "peek" into the future—a common pitfall in financial ML.
* **Extensible Model Factory**: A unified interface for testing various estimators, ranging from traditional linear regressions to ensemble methods (XGBoost, LightGBM).
* **Experiment Management**: Integrated with `MLflow` to track hyperparameters, metrics, and model versions for full research reproducibility.
* **Automated Feature Engineering**: A library of 100+ technical and statistical features built using `NumPy` and `Pandas` vectorization for performance.

---

## **Technical Architecture**
The project structure emphasizes **Separation of Concerns**, making it easy for developers to swap out individual components:

```text
Quant_Modeller/
├── src/
│   ├── core/               # Validation logic & Metric definitions
│   ├── features/           # Vectorized technical indicator library
│   ├── data_engine/        # API connectors (Alpha Vantage, Yahoo Finance)
│   ├── models/             # Wrapper classes for Scikit-learn & Gradient Boosting
│   ├── backtest/           # Event-simulating engine with slippage & cost models
│   └── ai_bridge/          # LLM integration for natural-language-to-schema mapping
├── tests/                  # Unit tests for core financial calculations
└── notebooks/              # Research examples and visualization demos
What Makes This Project Unique
Robustness Focus: Instead of focusing on "beating the market," this tool focuses on validation. It includes Monte Carlo resampling and noise-feature injection to test if a strategy is statistically significant.

Environment Agnostic: Developed and tested on Pop!_OS (Linux), ensuring compatibility with standard high-performance computing (HPC) environments.

LLM-Assisted Ideation: Uses the Gemini API to translate human-readable ideas into a structured JSON/Pydantic schema, allowing for human-in-the-loop strategy development.

Quick Start for Developers
1. Environment Setup
This project uses pip -e . for editable installs.

Bash
# Clone and setup virtual environment
git clone [https://github.com/shivansh023023/Quant_modeller.git](https://github.com/shivansh023023/Quant_modeller.git)
cd Quant_modeller
pip install -r requirements.txt
2. Running the Validation Suite
Before running any strategy, verify the core engine:

Bash
pytest tests/
Why I Built This
As a Computer Science student, I wanted to understand the technical difficulties of processing time-series data at scale. This project is an exercise in software architecture, API integration, and data integrity. It represents my journey into building modular, testable Python systems.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Contact
Shivansh - @shivansh023023
