# visual_oscillators_for_SCR

<p align="center">
  <a href="https://arxiv.org/abs/2511.18322" target="_blank">
    <img src="https://img.shields.io/badge/Paper-Learning%20Visually%20Interpretable%20Oscillator%20Networks%20for%20SCR-blueviolet?style=for-the-badge&logo=arxiv" alt="Paper Badge"/>
  </a>
</p>

<h2 align="center"><b>Learning Visually Interpretable Oscillator Networks
for Soft Continuum Robots from Video</b></h2>


<p align="center">
  <a href="https://www.youtube.com/watch?v=nbGgmY9bI-w" target="_blank">
    <img src="https://img.youtube.com/vi/nbGgmY9bI-w/hqdefault.jpg" alt="YouTube Video" style="width: 60%; max-width: 480px;">
  </a>
  <br>
  <a href="https://www.youtube.com/watch?v=nbGgmY9bI-w" target="_blank">
    ▶️ Watch the supplemental animations on YouTube
  </a>
</p>

# Content

- **`configs/`** - Configuration files for different model variants (harmonic/koopman, with/without attention, 1-segment/2-segment)
- **`data/`** - SCR video datasets (to be downloaded seperately)
- **`Latent_dynamics_learning.ipynb`** - Main notebook for training koopman and oscillator networks
- **`models.py`** - Model architecture definitions and functions
- **`utils.py`** - Utility functions
- **`results/`** - Trained model checkpoints, loss histories, and visualization figures
- **`SCR_data_processing.ipynb`** - Notebook for preparing SCR datasets from videos and raw data
# How to get the dataset

Processed data (compatible with this repo) and raw data are available on Zenodo:  
https://zenodo.org/records/17812071

# How to Cite
If you use this code or refer to our work, please cite the following:

@misc{krauss2025learningvisuallyinterpretableoscillator,
      title={Learning Visually Interpretable Oscillator Networks for Soft Continuum Robots from Video}, 
      author={Henrik Krauss and Johann Licher and Naoya Takeishi and Annika Raatz and Takehisa Yairi},
      year={2025},
      eprint={2511.18322},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2511.18322}, 
}

