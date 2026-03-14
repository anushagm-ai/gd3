# GlaucomaVision Pro V2

A professional-grade clinical decision support system for Glaucoma detection via fundus images, designed for clinical transparency and scientific accuracy.

## Project Architecture
- **backend/**: AI logic and API server.
  - `data_engine/`: Scripts for ophthalmological preprocessing (Green channel, CLAHE) and dataset splitting.
  - `model_logic/`: EfficientNet-B0 fine-tuning and inference.
- **frontend/**: React-based clinical dashboard.
- **data/**:
  - `raw/`: Original downloaded dataset (Arnav Jain / Kaggle).
  - `processed/`: Medical-grade preprocessed images (Augmented & Split).

## Scientific Approach
- **Backbone**: EfficientNet-B0 (Transfer Learning for medical imaging).
- **Clinical Evidence**: Optic Cup-to-Disc Ratio (CDR) calculation and segmentation.
- **Transparency**: Heatmaps (XAI) and Risk Grading (Low/Moderate/High).
- **Metrics**: Sensitivity, Specificity, ROC-AUC, and Confusion Matrix.

