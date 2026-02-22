# 🦷 Bits2Bites

This repository is a customized fork of [Pointcept](https://github.com/Pointcept/Pointcept) tailored for multi-task classification on 3D dental point cloud data. It includes several major extensions and changes that may affect the default behavior of Pointcept. We **recommend using this fork only to explore our implementation**.

## 📌 What's in this Fork

We implemented:

* ✅ **Custom model**: `MultiTaskClassifier` with multiple classification heads
  → Located in: `pointcept/models/multi_task_classifier`

* ✅ **Custom configs**
  → Located in: `configs/`

* ✅ **Custom data loader** for the `Bits2Bites` dataset
  → Located in: `pointcept/datasets/`
  → Dataset info: [Bits2Bites Dataset](https://ditto.ing.unimore.it/bits2bites)

* ✅ **Custom inference scripts**
  → Located in: `tools/`

* ✅ **Custom evaluation hook** for multi-task learning
  → Located in: `pointcept/engines/hooks/evaluator`

* ⚠️ **Additional modifications** to core components
  → This may impact compatibility with upstream Pointcept. Please use this fork only for reproducing our results.

## 📄 Citation & Paper

This repository supports the methods described in our MICCAI 2025 paper:
📝 *"Bits2Bites: Intra-oral Scans Occlusal Classification"*
**Authors**: L. Borghi, L. Lumetti, et al.
📍 To be presented at [ODIN - MICCAI 2025 – South Korea](https://odin-workshops.org)
📄 [Paper](https://openreview.net/forum?id=6KfC6LkKF4)
🖼️ [Poster](https://odin-workshops.org/2025/posters/bits2bites_occlusal_classification.pdf)

## 📦 Getting Started

Please refer to [Pointcept's installation guide](https://github.com/Pointcept/Pointcept) and ensure you clone this repository instead of the original.
