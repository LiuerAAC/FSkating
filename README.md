## Figure Skating Spin Analysis & Video Feature Toolkit ⛸️

---

A comprehensive toolkit for figure skating motion analysis, temporal action segmentation, and 3D pose-based view analysis, leveraging state-of-the-art models including VideoMAE, MS-TCN++, and custom pipelines for fine-grained skating jump and spin analysis.

---
## Features
+ **Figure Skating Video Feature Extraction**
  + Extract spatiotemporal features from skating videos using VideoMAE/VideoMAE v2.
  + Segment-level representation for jumps, spins, and step sequences.
+ **Temporal Action Segmentation (TAS)**
  + Precisely segment skating routines into technical elements.
  + Support multi-class labeling for jumps, spins, and footwork sequences.
+ **3D Pose & View Analysis**
  + Triangulate 3D joint locations from multi-view recordings.
  + Automated view judgment for rotations along XYZ axes in jumps and spins.
+ **Data Processing & Statistical Utilities**
  + Parsing of JSON, EAF, and other annotation formats.
  + Bootstrapping, sampling, and statistical analysis for performance evaluation.
+ **Experimental Tools**
  + Custom fitness functions for analyzing skating elements.
  + Mock pipelines for multi-view video augmentation and action assessment.
