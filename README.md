# ROC-Benchmark for IBF Models

This Python script performs a comparative ROC analysis and statistical evaluation of two classification models using the Bootstrap method (10,000 resamples). 

It compares a baseline model (`W_bar`) against an advanced model (`IBF_full`) to check if the performance improvement ($\Delta$AUC) is statistically significant.

## 📌 Features
* Calculates ROC curves and AUC.
* Finds the optimal threshold using Youden's Index ($J$).
* Computes 95% Confidence Intervals (CI) via Bootstrapping.
* Calculates $\Delta$AUC and a two-sided $p$-value.
* Automatically saves a text report (`text_report.txt`) and a plot (`roc_ibf_benchmark.png`).

## 🚀 Usage
1. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib scikit-learn scipy
   Place your data in the root folder as your_data.csv. The file must contain the following columns: W_bar, IBF_full, and collapse.

Run the script:
python roc_benchmark.py
If your_data.csv is not found, the script will automatically run on synthetic demo data.
