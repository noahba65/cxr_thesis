# Truncated EfficientNet for Tuberculosis Classification

## 📖 Description

This repository contains the implementation and analysis for my Master's thesis, which investigates truncated versions of EfficientNet-B0 for binary classification of Tuberculosis (TB) from chest X-rays (CXRs).

The core finding: **you can remove up to 3 blocks from EfficientNet-B0 and still achieve performance that matches—or even exceeds—the full model**, while slashing parameter count by 13×. This has significant implications for deploying TB screening tools in resource-limited settings.

Key contributions:
- 🔬 Systematic truncation of EfficientNet-B0 (from -1 to -4 blocks)
- 📊 Rigorous evaluation on internal (Kaggle) and external (Mendeley) datasets
- 📈 Bootstrap analysis with 95% confidence intervals
- ⚖️ Trade-off analysis between accuracy and model efficiency

---

## 🏆 Key Findings

- ✅ **100% accuracy** on internal Kaggle test set with all models
- 🌍 **97.4% external accuracy** with B0(-3), including:
  - Sensitivity: **98.96%**
  - Specificity: **95.68%**
- ⚡ B0(-3) uses only **308K parameters**, compared to **4.1M** in the full B0
- 📉 63× smaller than prior DenseNet-201–based approaches
- 🚀 Real-world potential for clinical use in low-resource settings

---

## 📁 Repository Structure

```
📁 cxr_thesis/
├── custom_lib/
│   ├── data_prep.py
│   ├── eval_tools.py
│   └── custom_models/
│       ├── truncated_b0.py, truncated_b0_leaky.py, etc.
│
├── run_model.py                  # Training script (full + truncated models)
├── run_model.ipynb              # Jupyter training variant
├── run_experiments.sh           # Batch experiment runner
├── run_bootstraps.ipynb         # Bootstrap CI evaluation
├── explore_model.ipynb          # Early exploration / architecture tests
├── replot_training_loss.ipynb   # Plot regeneration for report
├── results/                     # Saved metrics, checkpoints
├── external_bootstrap_results/  # External test bootstrap metrics
├── paper_figs/                  # Final figure exports for thesis
├── results_figs.ipynb           # Plot generation scripts
├── plots_presentation.pptx      # Presentation slides
├── requirements.txt             # Dependency list
└── thesis.pdf                   # Final paper
```

---

## 🛠️ Requirements

- Python 3.8+
- PyTorch ≥ 1.10
- torchvision
- scikit-learn
- pandas, matplotlib, seaborn

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Prepare Data
Download and organize the datasets:
- [Kaggle TB Dataset](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset)
- [Mendeley Pakistan TB Dataset](https://data.mendeley.com/datasets/jctsfj2sfn/1)

Place the images into a `data/` directory using the format expected by `data_prep.py`.

---

### 2. Train Model
```bash
python run_model.py --model b0_minus_3
```

### 3. Evaluate on External Data
```bash
python run_model.py --model b0_minus_3 --eval_only --external
```

### 4. Bootstrap Confidence Intervals
```bash
jupyter notebook run_bootstraps.ipynb
```

---

## 📊 Results Summary

| Model     | Params | Internal Acc | External Acc | Sensitivity | Specificity |
|-----------|--------|--------------|--------------|-------------|-------------|
| B0(-0)    | 4.1M   | 100%         | 97.26%       | 98.72%      | 95.68%      |
| B0(-3) 🔥 | 308K   | 100%         | 97.38%       | 98.96%      | 95.68%      |

> 🔥 B0(-3) is 13× smaller than B0(-0), with overlapping performance and better efficiency.

---

## 📚 References

1. Rahman et al. (2020) — Kaggle TB dataset  
2. Ke et al. (2021) — Model truncation and bootstrap evaluation  
3. WHO (2021) — TB triage standards and diagnostic guidelines  

---

