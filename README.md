# Mars Terrain Segmentation with CNNs

Semantic segmentation of Mars rover imagery using convolutional neural networks. The project focuses on pixel-level identification of four terrain classes: **soil, bedrock, sand, and large rock**, supporting visual analysis of the Martian surface.

## Goal

Develop and train a terrain segmentation model in PyTorch to identify different Martian terrain types and evaluate its ability to support rover navigation and environmental perception.

## Dataset

The project uses AI4MARS Curiosity NavCam images and terrain masks.

## Project workflow

Open [Terrain_Segmentation.ipynb](Terrain_Segmentation.ipynb) to check the project. It contains the workflow from model definitions and data preparation to training, validation, and results comparison. The notebook runs independently of the `.py` files. The original scripts are kept in `src/` for reference.

## Current stage

The project is at the progress-report stage. Data preparation and command-sequence splits are complete. The Random Forest baseline and the initial CNN have been trained and evaluated on the same fixed 100-image validation subset. The CNN has completed one training epoch.

Skip-connection comparisons and final test-set evaluation are not completed yet.

Data and saved models are kept locally and are not included in Git.

## Repository structure

```text
Terrain_Segmentation.ipynb   Main project workflow
src/                        Original Python scripts
tests/                      Code checks
data/                       Local data and frozen splits
outputs/                    Local models and results
```
