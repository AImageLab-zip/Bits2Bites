# 🦷 ToothFairy4 Multi-Task Fork of Pointcept

This repository is a customized fork of [Pointcept](https://github.com/Pointcept/Pointcept) tailored for multi-task classification on 3D dental point cloud data. It includes several major extensions and changes that may affect the default behavior of Pointcept. We **recommend using this fork only to explore our implementation**.

## 📌 What's in this Fork

We implemented:

* ✅ **Custom model**: `MultiTaskClassifier` with multiple classification heads
  → Located in: `pointcept/models/multi_task_classifier`

* ✅ **Custom configs**
  → Located in: `configs/`

* ✅ **Custom data loader** for the `ToothFairy4` dataset
  → Located in: `pointcept/datasets/`
  → Dataset info: [ToothFairy4 Dataset (link coming soon)](#)

* ✅ **Custom inference scripts**
  → Located in: `tools/`

* ✅ **Custom evaluation hook** for multi-task learning
  → Located in: `pointcept/engines/hooks/evaluator`

* ⚠️ **Additional modifications** to core components
  → This may impact compatibility with upstream Pointcept. Please use this fork only for reproducing our results.

## 📄 Citation & Paper

This repository supports the methods described in our MICCAI 2025 paper:
📝 *"Multi-task Classification on 3D Dental Point Clouds with Task-aware Transformers"*
**Authors**: Lorenzo Borghi et al., AImageLab
📍 To be presented at [MICCAI 2025 – South Korea](#)
📄 [Paper (link coming soon)](#)

## 📦 Getting Started

Please refer to [Pointcept's installation guide](https://github.com/Pointcept/Pointcept) and ensure you clone this repository instead of the original.