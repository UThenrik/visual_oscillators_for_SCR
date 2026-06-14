# visual_oscillators_for_SCR

<p align="center">
  <a href="https://ieeexplore.ieee.org/document/11560904" target="_blank">
    <img src="https://img.shields.io/badge/Publication-IEEE%20RA--L-blue?style=for-the-badge&logo=ieee" alt="IEEE RA-L Badge"/>
  </a>
  <a href="https://arxiv.org/abs/2511.18322" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv-blueviolet?style=for-the-badge&logo=arxiv" alt="arXiv Badge"/>
  </a>
</p>

<h2 align="center"><b>Learning Visually Interpretable Oscillator Networks
for Soft Continuum Robots from Video</b></h2>

<p align="center">
  <b>Published in IEEE Robotics and Automation Letters (RA-L), 2026</b><br>
  <a href="https://ieeexplore.ieee.org/document/11560904">IEEE Xplore</a> &nbsp;|&nbsp;
  DOI: <a href="https://doi.org/10.1109/LRA.2026.3703241">10.1109/LRA.2026.3703241</a>
</p>

<p align="center">
  <a href="https://www.youtube.com/watch?v=i80H8erVISM" target="_blank">
    <img src="https://img.youtube.com/vi/i80H8erVISM/maxresdefault.jpg" alt="YouTube Video" style="width: 80%; max-width: 720px; aspect-ratio: 16/9;">
  </a>
  <br>
  <a href="https://www.youtube.com/watch?v=i80H8erVISM" target="_blank">
    ▶️ Watch the supplemental video on YouTube
  </a>
</p>

# Content

- **`configs/`** - Configuration files for different model variants (harmonic/koopman, with/without attention, 1-segment/2-segment)
- **`data/`** - SCR video datasets (to be downloaded separately)
- **`Latent_dynamics_learning.ipynb`** - Main notebook for training Koopman and oscillator networks
- **`models.py`** - Model architecture definitions and functions
- **`utils.py`** - Utility functions
- **`results/`** - Trained model checkpoints, loss histories, and visualization figures
- **`SCR_data_processing.ipynb`** - Notebook for preparing SCR datasets from videos and raw data

# How to get the dataset

Processed data (compatible with this repo) and raw data are available on Zenodo:  
https://zenodo.org/records/17812071

# How to cite

If you use this code or refer to our work, please cite:

```bibtex
@ARTICLE{11560904,
  author={Krauss, Henrik and Licher, Johann and Takeishi, Naoya and Raatz, Annika and Yairi, Takehisa},
  journal={IEEE Robotics and Automation Letters},
  title={Learning Visually Interpretable Oscillator Networks for Soft Continuum Robots from Video},
  year={2026},
  pages={1-8},
  doi={10.1109/LRA.2026.3703241}
}
```

Paper: [https://ieeexplore.ieee.org/document/11560904](https://ieeexplore.ieee.org/document/11560904)
