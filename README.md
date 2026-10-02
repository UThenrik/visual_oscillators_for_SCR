# visual_oscillators_for_SCR

<p align="center">
  <a href="https://arxiv.org/abs/2603.19655" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv-blueviolet?style=for-the-badge&logo=arxiv" alt="arXiv Badge"/>
  </a>
  <a href="https://youtu.be/WKF82YBOH-Q" target="_blank">
    <img src="https://img.shields.io/badge/Video-YouTube-red?style=for-the-badge&logo=youtube" alt="YouTube Badge"/>
  </a>
</p>

<h2 align="center"><b>Accurate Open-Loop Control of a Soft Continuum Robot Using Visually Learned Latent Dynamics</b></h2>

<p align="center">
  <a href="https://arxiv.org/abs/2603.19655">arXiv:2603.19655</a>
</p>

<p align="center">
  <a href="https://youtu.be/WKF82YBOH-Q" target="_blank">
    <img src="https://img.youtube.com/vi/WKF82YBOH-Q/maxresdefault.jpg" alt="YouTube Video" style="width: 80%; max-width: 720px; aspect-ratio: 16/9;">
  </a>
  <br>
  <a href="https://youtu.be/WKF82YBOH-Q" target="_blank">
    Watch the supplemental video on YouTube
  </a>
</p>

This repository provides code, pretrained models, and processed data to replicate the **open-loop latent control** experiments: train / load latent dynamics models (Koopman, MLP, oscillator / VON, with and without ABCD), design targets in a live simulator, and optimize open-loop pressure trajectories in latent space.

The architectures (ABCD, VON) and original video-dynamics learning setup were introduced in our IEEE RA-L 2026 paper (see citation note below).

# Content

- **`configs/`** — Training configs for control-paper models (`*_control.yaml`) and VON ablations (`*_ablation*.yaml`), plus shared `base.yaml`
- **`data/scr_match2/`** — Processed training / step / static NPZs (included in this repo)
- **`Latent_dynamics_learning.ipynb`** — Train Koopman / MLP / oscillator networks on the control dataset
- **`Latent_control.ipynb`** — Open-loop single-shooting control in latent space
- **`live_simulation_pyqt.py`** — Interactive SCR live simulator for static / dynamic / extrapolated targets
- **`control_utils.py`** — Open-loop optimizer and control I/O helpers
- **`models.py`** — Model architecture definitions and losses
- **`utils.py`** — Config loading and checkpoint helpers
- **`Process_scr_match2_dataset.ipynb`** — Process raw H5 recordings into training NPZs
- **`results/models/V02_CDC/`** — Pretrained 2-segment control-paper checkpoints (6 main models + 7 VON ablations)
- **`results/control_states/2seg/`** — Simulator-recorded target pickles used by `Latent_control.ipynb`

# Dependencies

- Python with: `torch`, `numpy`, `pyyaml`, `matplotlib`, `opencv-python`, `scipy`, `pandas`
- Live simulator additionally: `PyQt5`, `pyqtgraph`

# Dataset

Processed NPZs for training and evaluation are included under `data/scr_match2/`:

- `2segments_smooth_input_rand_compressed_processed.npz` (main training data)
- `2segments_step_input_rand_18s_compressed_processed.npz`
- `2segments_step_input_rand_18s_compressed_static_processed.npz`
- Matching 1-segment NPZs (optional; configs provided)

Simulator-recorded control targets are included under `results/control_states/2seg/`.

Raw H5 recordings (and a fuller archive) are also available on Zenodo:

https://doi.org/10.5281/zenodo.22790365

# Quick start

1. Load a pretrained run from `results/models/V02_CDC/`, or retrain with `Latent_dynamics_learning.ipynb` using a `*_control.yaml` config (uses `data/scr_match2/`).
2. (Optional) Design / re-record targets with `python live_simulation_pyqt.py`.
3. Run `Latent_control.ipynb` using the shipped `results/control_states/2seg/` pickles (or newly recorded ones).

# How to cite

**If you use this repository for open-loop control, the control experiments, or the control-paper dataset, please cite the control paper:**

```bibtex
@article{krauss2026accurate,
  title={Accurate Open-Loop Control of a Soft Continuum Robot Using Visually Learned Latent Dynamics},
  author={Krauss, Henrik and Licher, Johann and Takeishi, Naoya and Raatz, Annika and Yairi, Takehisa},
  journal={arXiv preprint arXiv:2603.19655},
  year={2026}
}
```

Paper: [https://arxiv.org/abs/2603.19655](https://arxiv.org/abs/2603.19655)

**If you refer to the general methods (ABCD, VON / visually interpretable oscillator networks) or the original RA-L dataset, please cite the RA-L paper:**

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

- IEEE Xplore: [https://ieeexplore.ieee.org/document/11560904](https://ieeexplore.ieee.org/document/11560904)
- arXiv: [https://arxiv.org/abs/2511.18322](https://arxiv.org/abs/2511.18322)
- Original dataset: [https://zenodo.org/records/17812071](https://zenodo.org/records/17812071)
