# Multi-modal-deep-learning-for-fetal-MRI-corpus-callosum-segmentation
The corpus callosum (CC) plays a fundamental role in interhemispheric integration, supporting motor coordination, language, and sensory processing. Accurate prenatal assessment of CC morphology could enable the establishment of normative trajectories of neurodevelopment.
Here, we propose automated CC segmentation methods for fetal Magnetic Resonance Imaging (MRI) data. Both unimodal and multimodal deep learning models were developed by combining anatomical (T2-weighted, T2w) and microstructural (fractional anisotropy, FA) information.
Our dual-channel approach, evaluated across a gestational age range of 21–38 weeks, significantly outperformed state-of-the-art methods. The dual-channel model (T2w + FA) achieved a mean Dice score of 0.74 ± 0.07, a volume similarity of 0.92 ± 0.06, and perfect topological consistency (Euler Difference = 0).
Morphometric analysis of the CC revealed robust gestational age-dependent growth trajectories for most extracted parameters, including height, length, rostrum–splenium distance, skeleton length, area, volume, and thickness. In contrast, mean FA profile remained stable throughout gestation (Spearman ρ = −0.02), suggesting limited variation in the microstructural properties underlying callosal anisotropy during this developmental period.
<img width="3085" height="1752" alt="Picture_1" src="https://github.com/user-attachments/assets/f1243a93-b15f-43d3-92c4-c9605679e9f5" />
*Figure: Flow chart illustrating the processing pipeline for CC characterization.*
## Pipeline
| Stage | What it does | Code |
| :--- | :--- | :--- |
| 1. Segmentation | 3D nnU-Net inference on fetal brain reconstructions | `scripts/predict.sh` |
| 2. Characterization | Shape, regional and FA measures from the predicted masks | `scripts/cc-morphometry.py` |
## Input Data
3D fetal brain reconstructions meeting the following criteria:
- T2w image (e.g., 0.5 × 0.5 × 0.5 mm<sup>3</sup>) / FA map from dMRI data (e.g., 15 b = 0 s/mm<sup>2</sup>, 46 b = 400 s/mm<sup>2</sup>, 80 b = 1000 s/mm<sup>2</sup> at 2 mm isotropic resolution)
- Adequate signal-to-noise ratio (SNR) and overall image quality
- Full coverage of the CC region
- Good-quality 3D reconstruction
- Gestational age: 21–38 weeks
- No significant shading artifacts
- No severe structural anomalies
- Acquired using 1.5T or 3T MRI
## Stage 1: Segmentation
Three separate 3D nnU-Net architectures were trained on the dHCP fetal cohort:

| Dataset | Input | Channels |
| :--- | :--- | :--- |
| `Dataset030_CC_T2w` | T2w only | `_0000` |
| `Dataset031_CC_FA` | FA only | `_0000` |
| `Dataset032_CC_T2w_FA` | T2w + FA (dual-channel) | `_0000`, `_0001` |

Follow the [nnU-Net](https://github.com/mic-dkfz/nnunet) instructions for installation and environment setup, then run inference:
`./scripts/predict.sh dual INPUT_FOLDER OUTPUT_FOLDER`

Equivalently, in full: 
```bash
nnUNetv2_predict -d Dataset030_CC_T2w -i INPUT_FOLDER -o OUTPUT_FOLDER -f 0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans 
nnUNetv2_predict -d Dataset031_CC_FA -i INPUT_FOLDER -o OUTPUT_FOLDER -f 0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans 
nnUNetv2_predict -d Dataset032_CC_T2w_FA -i INPUT_FOLDER -o OUTPUT_FOLDER -f 0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans 
```
## Stage 2: Characterization
```bash
python cc_morphometry.py --masks OUTPUT_FOLDER --fa INPUT_FOLDER --output results 
```

### Extracted parameters

| Parameter | Columns |
| :--- | :--- |
| Area | `area_mm2`, `genu_area_mm2`, `body_area_mm2`, `splenium_area_mm2` |
| Volume | `cc_volume_mm3`, `genu_volume_mm3`, `body_volume_mm3`, `splenium_volume_mm3` |
| Height and length | `height_mm`, `length_mm` |
| Rostrum–splenium distance | `rostrum_splenium_distance_mm` |
| Perimeter | `perimeter_length_mm`, `perimeter_curvature_mean`, `perimeter_curvature_max` |
| Skeleton | `skeleton_length_mm`, `skeleton_curvature_mean` |
| Thickness | `thickness_{mean,std,min,max}_mm`, `thickness_{genu,body,splenium}_mm` |
| FA | `fa_mean`, `fa_{genu,body,splenium}` |

### Outputs

| File | Contents |
| :--- | :--- |
| `cc_morphometry.csv` | one row per subject, all scalar measures |
| `cc_thickness_profiles.csv` | thickness in mm at each midline position |
| `cc_fa_profiles.csv` | FA at each midline position |
| `<subject>_split.nii.gz` | regional label map (1 genu, 2 body, 3 splenium) |
| `qc/<subject>.png` | six-panel overview for visual inspection |

## Requirements
Python ≥ 3.9 — NumPy, SciPy, scikit-image, nibabel, OpenCV, pandas, Matplotlib. Stage 1 additionally requires [nnU-Net](https://github.com/mic-dkfz/nnunet) v2 and its own dependencies.

## How to cite
@inproceedings{  
title={Multi-modal deep learning for fetal corpus callosum segmentation and characterization in MRI},  
author={M. Di Stefano, D. Peruzzo, F. Montano, R. Licandro, A. De Luca, S. MC De Zwarte, A. Leemans and T. Ciceri},  
booktitle={MICCAI PIPPI},  
year={2026},  
doi={}  
}
