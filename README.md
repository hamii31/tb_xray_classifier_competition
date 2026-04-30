# TB Chest X-Ray Classifier

A binary classifier for tuberculosis detection on chest X-rays, built for the DataCamp "Detecting Tuberculosis in X-Rays" competition.

## Problem

Distinguish healthy lungs from TB-affected lungs in chest X-ray images. Built on a 400-image sample of the Sakha-TB dataset (300 train, 100 test, perfectly balanced classes). The interesting question isn't peak accuracy — with 240 effective training images that's noise-bound — it's how training strategy and feature learning interact on a small medical-imaging dataset.

## Approach

Three-way model comparison in PyTorch:

- **Small CNN** — 4-block conv net, ~100k parameters, trained end-to-end on the X-rays.
- **ResNet18 (frozen)** — ImageNet-pretrained backbone frozen, only a fresh classifier head trained.
- **ResNet18-FT** — same as above but with the last residual block (`layer4`) unfrozen and fine-tuned at lr=1e-4.

All three evaluated on the same held-out test set with the four metrics that matter for a screening task — sensitivity, specificity, PPV, NPV — plus ROC and threshold tuning.

## Key result

Test AUC: Small CNN 0.951, ResNet18-FT 0.918, ResNet18 (frozen) 0.667.

The frozen-features model was barely above chance. Unfreezing one residual block raised AUC by 0.25, putting it within noise of the from-scratch CNN. The takeaway: on small medical-imaging datasets, *training strategy* (frozen vs. fine-tuned vs. end-to-end) mattered more than *architecture* (100k-param custom CNN vs. 11M-param ResNet18). Consistent with Raghu et al. (2019), *Transfusion: Understanding Transfer Learning for Medical Imaging*.

## Stack

PyTorch · torchvision · scikit-learn · matplotlib

## Notebook

Full analysis published on DataLab: [link](https://www.datacamp.com/datalab/w/73558e81-6544-4e04-a78f-ef2aa26854b9)
