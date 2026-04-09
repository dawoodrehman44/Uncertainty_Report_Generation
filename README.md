# Uncertainty-Conditioned Classification and Report Generation. A Bayesian Framework for Interpretable Diagnosis

<p align="center">
  <img src="Figures/Framework Architecture Plot - Phase 1.png" alt="Phase 1 Architecture Plot" width="1000"/>
</p>
---

## 🔥 Highlights

- **Uncertainty Decomposition**: Explicitly separates epistemic (model) and aleatoric (data) uncertainty through variational Bayesian inference
- **Uncertainty-Conditioned Reports**: First framework to generate medical reports with linguistically appropriate hedging based on quantified uncertainty
- **State-of-the-Art Performance**: 
  - Classification: **AUC 0.889**, **ECE 0.013** (83% reduction vs. baselines)
  - Report Generation: **BLEU-4 0.229**, **Clinical F1 0.511**
- **Two-Phase Training**: Prevents gradient conflicts, maintains calibration while learning report generation
- **Multi-Dataset Validation**: Evaluated on CheXpert, MIMIC-CXR, ChestX-ray14, IU-Xray (700K+ images)

---

## 📋 Abstract

This repository contains the implementation of our **Uncertainty-Conditioned Classification and Report Generation. A Bayesian Framework for Interpretable Diagnosis** for uncertainty-aware chest X-ray classification.  

Our approach addresses the critical need for **reliable uncertainty quantification and report generation in medical AI** by combining:  

- **Bayesian Classification**  
- **Uncertainty Quantification**  
- **Consistency Validation**  
- **Adaptive Calibration**
- **Report Generation with Integrated Uncertainty**  

**Key Innovation**: Unlike traditional deep learning models that output point predictions and report generation, our framework provides *accurate predictions with uncertainty estimates and integrate those uncertainties into report generation* that help clinicians understand **when the model is uncertain and why** — crucial for high-stakes medical decisions.

---

## 🏗️ Architecture Overview

Our framework consists of four main components:

1. **Hierarchical Bayesian Encoder**: Multi-scale feature extraction with variational layers  
2. **Disease Classification Agent**: Bayesian classifier with uncertainty quantification  
3. **Consistency Validation Agent**: Cross-validation of predictions and uncertainties  
4. **Adaptive Calibration Module**: Dynamic temperature and Platt scaling
5. 

---

## 📁 Repository Structure
```bash
├── Configuration/
│   └── model_configuration.json           # Model configuration
├── Baesian Model/
│   └── enhanced_multiagent_bayesian_model.py
├── data_Pre-processing/
│   ├── IU_Xray_cxr_class.py
│   └── mimic_cxr_class.py              
├── Figures/
├── Framework_Components/
│   └── bayesian_components.py
├── Evaluation_Metrics/
│   └── evaluation_metrics_calculator.py   # Custom evaluation metrics
├── Experiments/
│   ├── Comparative_Case_Study_Analysis.py
│   └── Uncertainty_Analysis.py
├── Figures/
│   └── Architecture_Plot.png
├── Model/
│   ├── model_integration.py
│   ├── model_train.py
│   ├── model_valid.py                     # Framework & uncertainty visualizations
├── Model Components/
│   ├── bayesian_encoder.py                # Hierarchical Bayesian encoder
│   ├── calibration.py                     # Adaptive calibration
│   ├── classification_network.py          # Disease classification agent
│   ├── consistency_validation.py          # Consistency validation agent
│   ├── enhanced_bayesian_framework.py     # Full model integration
│   ├── multi_objective_loss.py            # Multi-objective loss
│   └── variational_linear.py              # Variational linear layers
├── main.py                                # Entry point for training/testing
├── requirements.text                      # Environment dependencies
└── README.md                              # Project documentation


```

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/dawoodrehman44/ICASSP-2026.git
cd ICASSP-2026

```
### Create environment
```bash
conda create -n bayesian_med python=3.8
conda activate bayesian_med

# Install dependencies
pip install -r requirements.txt
```

## Training
### Train the Enhanced Bayesian Framework
```bash
python main.py \
    --model train \
    --config configuration/model_configuration.json \
    --data_path /path/to/training \

```

## Testing
### Perform comprehensive uncertainty analysis
```bash
python Experiments/uncertainty_Analysis.py \
    --Experiments/Comparative_Case_Study_Analysis.py \
    --data_path /path/to/validation \
    --mc_samples 1000
```

## 🤝 Acknowledgments
- **Datasets**: CheXpert (Stanford), MIMIC-CXR (MIT), ChestX-ray14 (NIH), IU-Xray (Indiana University)
- **Pre-trained Models**: CXR-CLIP, BioBart
- **Frameworks**: PyTorch, Hugging Face Transformers

## Contact
For questions or collaborations, please contact: 
Dawood Rehman – [dawoodrehman1297@gapp.nthu.edu.tw]