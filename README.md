# AI-Assisted Early Characterization of Corpus Callosum Development

<p align="center">
  <strong>Quantifying Corpus Callosum Morphology from Infant Brain MRI for Early Neurodevelopmental Assessment</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Domain-Medical%20AI-blue" />
  <img src="https://img.shields.io/badge/Deep%20Learning-U--Net-orange" />
  <img src="https://img.shields.io/badge/Imaging-Infant%20MRI-purple" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-red" />
  <img src="https://img.shields.io/badge/Backend-FastAPI-green" />
  <img src="https://img.shields.io/badge/Frontend-Next.js-black" />
  <img src="https://img.shields.io/badge/Status-Research%20Prototype-yellow" />
</p>

---

## Overview

Early brain development is a highly dynamic process, and structural abnormalities in infancy may provide important clues for earlier clinical evaluation and monitoring.

The **corpus callosum (CC)** is a major white-matter structure connecting the cerebral hemispheres. Its size, shape, thickness, and regional morphology change throughout early development.

This project aims to develop an **AI-assisted neuroimaging system** capable of automatically identifying, segmenting, and quantitatively characterizing the corpus callosum from infant brain MRI.

The system combines:

**3D Infant MRI → Preprocessing → Mid-Sagittal Localization → CC Segmentation → Morphological Analysis → Age-Matched Reference Analysis → Explainable Report**

The objective is not to replace clinical diagnosis.

Instead, the system is designed to provide **objective quantitative measurements that may support earlier clinical evaluation, monitoring, and research into neurodevelopmental abnormalities.**

---

# Problem Statement

Current assessment of early neurodevelopmental abnormalities often depends on clinical examination, developmental milestones, and expert interpretation of neuroimaging.

Although infant MRI contains valuable structural information, extracting quantitative anatomical measurements from MRI can be:

- time-consuming
- operator-dependent
- difficult to standardize
- challenging to scale across large datasets

In particular, the corpus callosum is an anatomically important structure whose morphology can potentially provide measurable information about early brain development.

### Our problem

> **Develop an AI-assisted system that automatically extracts and analyzes the corpus callosum from infant brain MRI, quantifies its structural morphology, and compares those measurements against age-matched developmental reference patterns to support earlier clinical evaluation and longitudinal monitoring.**

---

# Core Idea

Instead of attempting to diagnose a neurological condition directly, this project focuses on a much more specific and measurable problem:

```text
                 INFANT BRAIN MRI
                         │
                         ▼
              ┌─────────────────────┐
              │     PREPROCESSING    │
              │                     │
              │ Orientation         │
              │ Normalization       │
              │ Brain extraction    │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ MID-SAGITTAL        │
              │ LOCALIZATION        │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ CC SEGMENTATION     │
              │                     │
              │       U-Net         │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ MORPHOLOGY          │
              │                     │
              │ Area                │
              │ Length              │
              │ Thickness           │
              │ Regional structure  │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ AGE-AWARE ANALYSIS  │
              │                     │
              │ Expected morphology │
              │ Deviation           │
              │ Reference range     │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ EXPLAINABLE REPORT  │
              └─────────────────────┘
