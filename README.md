# Mars Terrain Segmentation with CNNs

A deep learning project exploring how convolutional neural networks can label terrain in Mars rover images. Rather than assigning one category to an entire image, the model will predict a terrain class for each pixel: **soil, bedrock, sand, or large rock**.

## Goal

Design and train an encoder–decoder CNN in PyTorch, then examine how different skip-connection configurations affect terrain segmentation. A Random Forest classifier using prepared pixel features will provide a baseline comparison.

## Current status

The project is in the design and data-preparation stage. The dataset has been downloaded and inspected locally. Model implementation, training experiments, and baseline evaluation are planned; this repository does not yet contain training code or performance results.

## Planned approach

### Data preparation

- Check that images and labels are readable and have matching dimensions.
- Check for duplicates before assigning images to training, validation, and testing sets.
- Prepare 256 × 256 image crops and ignore unlabeled pixels during training and evaluation.
- Apply paired geometric augmentation, such as horizontal flipping, and training-only brightness adjustments.

### CNN design

The encoder will learn spatial features through convolution and pooling. The decoder will use upsampling and convolution to restore spatial resolution and produce four terrain scores at each pixel.

The design comparison will examine:

| Configuration | Feature transfer |
| --- | --- |
| No skip connections | Decoder uses the encoded representation without direct encoder-stage connections. |
| Standard skip connections | Earlier encoder features are passed to corresponding decoder stages. |
| Compressed skip connections | Fewer feature channels are transferred between the corresponding stages. |

Validation results will guide model selection. Final comparisons will use the same held-out testing data. These are planned experiments, not claims that one configuration will outperform the others.

### Random Forest baseline

The baseline will use prepared pixel intensity or color values and local neighborhood information. It will use the same source-image splits as the CNN, providing a comparison between prepared features and learned convolutional features.

## Dataset

The project uses **AI4MARS**, a public dataset of labeled Mars rover imagery. The local working release contains Curiosity/MSL data.

- [AI4MARS paper and citation](https://openaccess.thecvf.com/content/CVPR2021W/AI4Space/html/Swan_AI4MARS_A_Dataset_for_Terrain-Aware_Autonomous_Driving_on_Mars_CVPRW_2021_paper.html)
- [NASA dataset catalog](https://data.nasa.gov/dataset/ai4mars-a-dataset-for-terrain-aware-autonomous-driving-on-mars)
- [Public mirror used for the local download](https://huggingface.co/datasets/hassanjbara/AI4MARS)

Dataset files are stored separately and are not included in this repository. Image data, weights, and generated experiment outputs are excluded by `.gitignore`. Download and environment instructions will be added alongside the implementation.

## Repository structure

```text
src/        Model, preparation, training, and evaluation code (planned)
notebooks/  Exploration and experiment notebooks (planned)
reports/    Project documentation and reports
```

These directories currently contain placeholders.

## Limitations

Terrain classes are unevenly represented, particularly large rocks. Results on the selected images will not establish performance across other rover missions or operating conditions. Incorrect terrain predictions could mislead navigation decisions; this project is an offline investigation and is not a validated rover-navigation system.

## Research foundations

The architecture and experiments draw on established segmentation methods, with attribution to the original work:

- [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [UNet++: A Nested U-Net Architecture for Medical Image Segmentation](https://arxiv.org/abs/1807.10165)
- [Mars Terrain Segmentation with Less Labels](https://arxiv.org/abs/2202.00791)

The project's contribution will be its implementation, justified design decisions, preparation pipeline, and experiments. It does not claim to invent skip connections or establish a new benchmark.
