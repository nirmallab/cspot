---
hide:
  - toc        
  - navigation
---

[![Downloads](https://static.pepy.tech/badge/cspot)](https://pepy.tech/project/cspot)
[![docs](https://github.com/nirmallab/cspot/actions/workflows/docs.yml/badge.svg)](https://github.com/nirmallab/cspot/actions/workflows/docs.yml)


## CELL SPOTTER (CSPOT): A scalable machine learning framework for automated processing of highly multiplexed tissue images  

Highly multiplexed tissue imaging and in situ spatial profiling aims to extract single-cell data from specimens containing closely packed cells having diverse morphologies. This is a challenging problem due to the difficulty of accurately assigning boundaries between cells (the process of segmentation) and then integrating per-cell staining intensities. In addition, existing methods use gating to assign positive and negative scores to individual scores, a common approach in flow cytometry but one that is restrictive in high-resolution imaging. In contrast, human experts identify cells in crowded environments using morphological, neighborhood, and intensity information. Here we describe a computational approach (Cell Spotter or CSPOT) that uses supervised machine learning in combination with classical segmentation to combine human visual review and computation for automated cell type calling.  The end-to-end Python implementation of CSPOT can be integrated into cloud-based image processing pipelines and substantially improves the speed, accuracy, and reproducibility of single-cell spatial data.