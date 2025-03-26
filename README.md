# Truncated EfficientNet for Tuberculosis Classification

## Description
This repository contains the implementation and analysis for my Master's thesis investigating truncated versions of EfficientNet-B0 for binary Tuberculosis (TB) classification in Chest X-Rays (CXRs). The project demonstrates that significant model size reductions (up to 13× smaller) can be achieved while maintaining diagnostic accuracy that meets WHO guidelines.

Key features:
- Systematic truncation of EfficientNet-B0 architecture
- Evaluation on both internal (Kaggle) and external (Mendeley) datasets
- Performance analysis including bootstrap confidence intervals
- Comparative study against state-of-the-art models

## Key Findings
- Achieved **100% accuracy** on internal test set with truncated models
- External validation showed **98.71% sensitivity** and **98.97% specificity** (B0(-3))
- Models **13-63× more efficient** than comparable approaches
- Demonstrated potential for clinical deployment in resource-constrained settings

## Repository Structure
```
cxr_thesis/
├── custom_lib/
│   ├── __init__.py
│   ├── data_prep.py
│   ├── eval_tools.py
│   └── custom_models/
│       ├── __init__.py
│       ├── basic_nn.py
│       ├── spatial_sep.py
│       ├── truncated_b0.py
│       ├── truncated_b0_leaky.py
│       ├── truncated_b0_leaky2.py
│       └── truncated_b0_act1.py
│
├── results/
├── external_bootstrap_results/
├── results_figs.ipynb
├── explore_model.ipynb
├── freeze/
├── run_bootstraps.ipynb
├── run_experiments.sh
├── run_model.ipynb
├── run_model.py
├── plots_presentation.pptx Results of results_figs.ipynb in a powerpoint
├── paper_figs/ # Folder for figures used in my thesis
├── requirements.txt # Required packages and versions
├── myenv/  # (virtual environment - usually excluded from version control)
```

## Requirements
- Python 3.8+
- PyTorch 1.10+
- torchvision
- scikit-learn
- pandas
- matplotlib
- seaborn

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. **Data Preparation**:
   - Download datasets from [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) and [Mendeley](https://data.mendeley.com/datasets/jctsfj2sfn/1)
   - Place in `data/` directory following the preprocessing scripts

2. **Training**:
```bash
python src/training.py --config configs/b0_minus_3.yaml
```

3. **Evaluation**:
```bash
python src/evaluation.py --model_path models/b0_minus_3.pt --dataset external
```

4. **Visualization**:
```bash
python src/visualization.py --model_path models/b0_minus_3.pt
```

## Results Summary
| Model | Params | Test Acc | External Acc | Sensitivity | Specificity |
|-------|--------|----------|--------------|-------------|-------------|
| B0    | 4.01M  | 100%     | 97.26%       | 0.9872      | 0.9568      |
| B0(-3)| 308K   | 100%     | 97.38%       | 0.9896      | 0.9573      |

## References
1. Rahman et al. (2020) - Kaggle TB dataset
2. Ke et al. (2021) - Model truncation study
3. WHO TB diagnostic guidelines

## License
MIT License - See [LICENSE](LICENSE) for details
