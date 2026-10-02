"""
Live simulation of the soft robot with PyQtGraph.
Supports 1-segment (2 pressure channels) and 2-segment (4 pressure channels) models.
Runs at consistent 50 Hz. Starts at rest (steady-state z and z_dot from dataset).

Dependencies:
    pip install pyqtgraph PyQt5

Usage:
    python live_simulation_pyqt.py [--model MODEL_NAME_OR_SUFFIX]
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from PyQt5.QtWidgets import QSlider, QPushButton, QRadioButton, QButtonGroup, QVBoxLayout, QHBoxLayout, QLabel, QWidget, QShortcut, QListWidget, QListWidgetItem, QMenu, QFileDialog, QAction, QComboBox, QStyle, QStyleOptionSlider, QCheckBox
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QRectF, QPointF
from PyQt5.QtGui import QFont, QKeySequence, QPainter, QPainterPath, QPen, QBrush, QColor
import threading

import models

# Dataset paths for steady-state rest (same as live_simulation.py)
DATASET_PATHS = {
    "scr_match_1seg": "scr_match2/1segment_smooth_input_rand_compressed_processed.npz",
    "scr_match_2seg": "scr_match2/2segments_smooth_input_rand_compressed_processed.npz",
}

# Dataset paths for max-deformation overlay by segment (static vs dynamic)
OVERLAY_DATASET_PATHS = {
    "1seg": {
        "static": "scr_match2/1segment_step_input_rand_compressed_static_processed.npz",
        "dynamic": "scr_match2/1segment_smooth_input_rand_compressed_processed.npz",
    },
    "2seg": {
        "static": "scr_match2/2segments_step_input_rand_18s_compressed_static_processed.npz",
        "dynamic": "scr_match2/2segments_smooth_input_rand_compressed_processed.npz",
    },
}

# Segment dropdown labels and radio variants (no "1 Seg"/"2 Seg" prefix on radios)
SEGMENT_KEYS = ("1seg", "2seg")
SEGMENT_DISPLAY = {"1seg": "1 Segment SCR", "2seg": "2 Segment SCR"}
MODEL_VARIANTS = [
    ("Koopman", {"1seg": "scr_match_1seg_koopman", "2seg": "scr_match_2seg_koopman"}),
    ("Koopman + ABCD", {"1seg": "scr_match_1seg_koopman_attn", "2seg": "scr_match_2seg_koopman_attn"}),
    ("MLP", {"1seg": "scr_match_1seg_mlp", "2seg": "scr_match_2seg_mlp"}),
    ("MLP + ABCD", {"1seg": "scr_match_1seg_mlp_attn", "2seg": "scr_match_2seg_mlp_attn"}),
    ("Oscillator", {"1seg": "scr_match_1seg_harmonic1d", "2seg": "scr_match_2seg_harmonic1d_main"}),
    ("Oscillator + ABCD", {"1seg": "scr_match_1seg_harmonic2d_vel_attn", "2seg": "scr_match_2seg_harmonic2d_vel_attn_main"}),
]
# Manuscript Fig. 3 uses $u_i$; GUI pressure channels use the same subscript style
PRESSURE_LABELS = ["p₁", "p₂", "p₃", "p₄"]
ORANGE_EXTRAP = (230, 126, 34)  # outside dataset range
DATASET_P_LOW = 0.0  # commanded rest; measured min is slightly below 0
_DATASET_P_HIGH_CACHE = {}

# CLI / old GUI names -> (segment_key, variant)
_OLD_MODEL_NAME_MAP = {
    "1 Seg Koopman": ("1seg", "Koopman"),
    "1 Seg Koopman + Attention": ("1seg", "Koopman + ABCD"),
    "1 Seg MLP": ("1seg", "MLP"),
    "1 Seg MLP + Attention": ("1seg", "MLP + ABCD"),
    "1 Seg Oscillator": ("1seg", "Oscillator"),
    "1 Seg Oscillator + Attention": ("1seg", "Oscillator + ABCD"),
    "2 Seg Koopman": ("2seg", "Koopman"),
    "2 Seg Koopman + Attention": ("2seg", "Koopman + ABCD"),
    "2 Seg Koopman\n+ ABCD": ("2seg", "Koopman + ABCD"),
    "2 Seg MLP": ("2seg", "MLP"),
    "2 Seg MLP + Attention": ("2seg", "MLP + ABCD"),
    "2 Seg MLP\n+ ABCD": ("2seg", "MLP + ABCD"),
    "2 Seg Oscillator": ("2seg", "Oscillator"),
    "2 Seg Oscillator + Attention": ("2seg", "Oscillator + ABCD"),
    "2 Seg Oscillator\n+ ABCD": ("2seg", "Oscillator + ABCD"),
}


# ---------------------------------------------------------------------------
# Model loading (same as live_simulation.py)
# ---------------------------------------------------------------------------

def find_latest_model_run(config_name):
    """Find latest results/models directory whose name ends with config_name."""
    results_dir = Path("results/models/V02_CDC")
    if not results_dir.exists():
        return None
    matching_dirs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir() and d.name.endswith(config_name)]
    )
    return matching_dirs[-1] if matching_dirs else None


def dataset_pressure_highs(dataset_name):
    """Per-channel upper pressure bounds from the training dataset."""
    if dataset_name in _DATASET_P_HIGH_CACHE:
        return list(_DATASET_P_HIGH_CACHE[dataset_name])
    rel = DATASET_PATHS.get(dataset_name)
    highs = [0.87]
    if rel:
        path = Path("data") / rel
        if path.exists():
            with np.load(path, mmap_mode="r") as dataset:
                keys = sorted(
                    [k for k in dataset.files if len(k) >= 2 and k[0] == "p" and k[1:].isdigit()],
                    key=lambda k: int(k[1:]),
                )
                if keys:
                    highs = [float(np.nanmax(dataset[k])) for k in keys]
    _DATASET_P_HIGH_CACHE[dataset_name] = highs
    return list(highs)


def load_model_and_config(model_run_suffix):
    """
    Find the run folder by suffix, load config from that run's config.json,
    build VAE and dynamics, load checkpoint. Returns (vae, dynamics, config, device).
    """
    run_dir = find_latest_model_run(model_run_suffix)
    if run_dir is None:
        raise FileNotFoundError(
            f"No trained run found ending with '{model_run_suffix}'. "
            "Train a model first or check results/models."
        )

    epoch_folders = sorted([d for d in run_dir.glob("epoch_*") if d.is_dir()])
    if not epoch_folders:
        raise FileNotFoundError(f"No epoch_* folders in {run_dir}")

    latest_epoch = epoch_folders[-1]
    config_file = latest_epoch / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Config not found: {config_file}")

    with open(config_file, "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build VAE from saved config
    vae = models.VAE(
        input_channels=1,
        latent_dim=config["latent_dim"],
        is_vae=config["use_vae"],
        use_attention_decoder=config["use_attention_decoder"],
        attention_feature_dim=config["attention_feature_dim"],
        attention_downsample_factor=config["attention_downsample_factor"],
        background_token_value=config["background_token_value"],
        image_size=config["resolution"],
        dynamics_type=config["dynamics_type"],
        gumbel_noise_strength=config["gumbel_noise_strength"],
    ).to(device)

    # Dynamics config: expand actuation_dim if delayed actuation is used
    dynamics_config = config.copy()
    num_delays = config.get("num_actuation_delays", 1)
    if num_delays > 1:
        dynamics_config["actuation_dim"] = config["actuation_dim"] * num_delays

    latent_dynamics = models.create_dynamics_model(config=dynamics_config, device=device).to(device)

    # Load checkpoint from same epoch folder
    ckpt_path = latest_epoch / "model_checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    vae.load_state_dict(checkpoint["vae_state_dict"])
    latent_dynamics.load_state_dict(checkpoint["latent_dynamics_state_dict"])

    vae.eval()
    latent_dynamics.eval()

    return vae, latent_dynamics, config, device


def build_models_on_device(config, device, source_vae=None, source_dynamics=None):
    """
    Build VAE and dynamics on the given device from config, optionally copying state
    from source models (so the new instances are safe to use from the current thread).
    Returns (vae, dynamics) in eval mode.
    """
    vae = models.VAE(
        input_channels=1,
        latent_dim=config["latent_dim"],
        is_vae=config["use_vae"],
        use_attention_decoder=config["use_attention_decoder"],
        attention_feature_dim=config["attention_feature_dim"],
        attention_downsample_factor=config["attention_downsample_factor"],
        background_token_value=config["background_token_value"],
        image_size=config["resolution"],
        dynamics_type=config["dynamics_type"],
        gumbel_noise_strength=config["gumbel_noise_strength"],
    ).to(device)
    dynamics_config = config.copy()
    num_delays = config.get("num_actuation_delays", 1)
    if num_delays > 1:
        dynamics_config["actuation_dim"] = config["actuation_dim"] * num_delays
    latent_dynamics = models.create_dynamics_model(config=dynamics_config, device=device).to(device)
    if source_vae is not None:
        vae.load_state_dict(source_vae.state_dict())
    if source_dynamics is not None:
        latent_dynamics.load_state_dict(source_dynamics.state_dict())
    vae.eval()
    latent_dynamics.eval()
    return vae, latent_dynamics


def get_rest_state_from_steady(vae, config, device):
    """
    Get (z_rest, z_dot_rest) from the steady-state portion of the dataset.
    Returns (z, z_dot) each of shape (1, latent_dim) on device.
    """
    dataset_name = config.get("dataset")
    if dataset_name not in DATASET_PATHS:
        raise FileNotFoundError(
            f"Unknown dataset '{dataset_name}' for rest state. "
            f"Known: {list(DATASET_PATHS.keys())}. Add path in DATASET_PATHS if needed."
        )
    data_path = Path("data") / DATASET_PATHS[dataset_name]
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Needed to compute steady-state rest (z, z_dot)."
        )

    with np.load(data_path, mmap_mode="r") as dataset:
        if "trajectory_time" not in dataset:
            raise ValueError("Dataset has no 'trajectory_time'; cannot determine steady-state frames.")
        n_steady = int(np.nonzero(dataset["trajectory_time"])[0][0])
        if n_steady <= 0:
            raise ValueError("No steady-state frames (trajectory_time nonzero from start).")

        o = dataset["images"]
        if o.max() > 1.0:
            o = o / 255.0
        if o.ndim == 3:
            o = o[:, None, :, :]
        elif o.shape[-1] == 3 and o.shape[1] != 3:
            o = np.transpose(o, (0, 3, 1, 2))
        # One steady-state frame (first)
        o_steady = o[0:1].copy().astype(np.float32)
        o_t = torch.from_numpy(o_steady).float().to(device)

    with torch.no_grad():
        if config["use_vae"]:
            mu, _ = vae.encode(o_t)
        else:
            mu = vae.encode(o_t)
        z_rest = mu
        #o_dot_zero = torch.zeros_like(o_t)
        #z_dot_rest = vae.latent_velocity_from_observation_velocity(o_t, o_dot_zero)
        z_dot_rest = torch.zeros_like(z_rest)

    return z_rest, z_dot_rest


def get_segment_key(config):
    """Return '1seg' or '2seg' from config['dataset'] for overlay/rest state."""
    ds = config.get("dataset", "")
    if "1seg" in ds:
        return "1seg"
    if "2seg" in ds:
        return "2seg"
    return "2seg"  # default


def load_max_deformation_overlay(segment_key, kind):
    """
    Load maximum deformation image from static or dynamic dataset.
    segment_key: '1seg' or '2seg'.
    kind: 'static' or 'dynamic'.
    Returns (H, W) float in [0, 1], or None if file missing.
    """
    if segment_key not in OVERLAY_DATASET_PATHS or kind not in OVERLAY_DATASET_PATHS[segment_key]:
        return None
    path = Path("data") / OVERLAY_DATASET_PATHS[segment_key][kind]
    if not path.exists():
        return None
    with np.load(path, mmap_mode="r") as ds:
        if "images" not in ds:
            return None
        o = np.array(ds["images"], copy=True)
    if o.max() > 1.0:
        o = o.astype(np.float32) / 255.0
    if o.ndim == 3:
        pass  # (T, H, W)
    elif o.ndim == 4:
        if o.shape[1] == 1:
            o = o[:, 0, :, :]
        elif o.shape[-1] == 3:
            o = np.transpose(o, (0, 3, 1, 2))[:, 0, :, :]
        else:
            o = o[:, 0, :, :]
    max_img = o.max(axis=0).astype(np.float32)
    if max_img.max() > 0:
        max_img = max_img / max_img.max()
    return max_img


# ---------------------------------------------------------------------------
# Simulation state and step
# ---------------------------------------------------------------------------

def build_u_from_sliders(slider_values, config, device):
    """Build control vector u from 4 slider values (single timestep, no delays)."""
    actuation_dim = config["actuation_dim"]
    u_now = np.array(slider_values[:actuation_dim], dtype=np.float32)
    return torch.from_numpy(u_now).float().unsqueeze(0).to(device)


def simulation_step(vae, dynamics, z, z_dot, u, dt, device):
    """One 50 Hz step: (z, z_dot) -> (z_next, z_dot_next), decode to image."""
    with torch.no_grad():
        z_next, zd_next = dynamics.forward(z, z_dot, u, dt)
        img = vae.decode(z_next)
    return z_next, zd_next, img


def oscillator_force_arrows_supported(vae, dynamics):
    """True for 2D oscillator + ABCD (attention COM Jacobian exists)."""
    osc = getattr(dynamics, "osc_net", None)
    decoder = getattr(vae, "decoder", None)
    if osc is None or getattr(osc, "control_to_state", None) is None:
        return False
    if decoder is None or getattr(decoder, "dim_per_attention", 1) != 2:
        return False
    return True


class ForceArrowItem(pg.GraphicsObject):
    """Filled arrow in image data coords: tail at (x0, y0), tip at (x0+dx, y0+dy).

    Avoids pg.ArrowItem, whose rotation is relative to the tip and fights invertY.
    """

    def __init__(self, color):
        super().__init__()
        if not isinstance(color, QColor):
            color = QColor(*color)
        self._color = color
        self._path = QPainterPath()
        self._rect = QRectF(-0.5, -0.5, 1.0, 1.0)
        self.setZValue(20)
        self.setVisible(False)

    def set_vector(self, x0, y0, dx, dy):
        length = float(np.hypot(dx, dy))
        if not np.isfinite(length) or length < 0.15:
            self.setVisible(False)
            return
        self.prepareGeometryChange()
        self.setPos(float(x0), float(y0))
        ux, uy = float(dx) / length, float(dy) / length
        px, py = -uy, ux
        head = min(2.4, max(0.8, 0.32 * length))
        if head > 0.85 * length:
            head = 0.85 * length
        half_head = max(0.45, 0.12 * length)
        half_tail = max(0.18, 0.045 * length)
        hx, hy = float(dx) - ux * head, float(dy) - uy * head
        path = QPainterPath()
        pts = [
            QPointF(px * half_tail, py * half_tail),
            QPointF(hx + px * half_tail, hy + py * half_tail),
            QPointF(hx + px * half_head, hy + py * half_head),
            QPointF(float(dx), float(dy)),
            QPointF(hx - px * half_head, hy - py * half_head),
            QPointF(hx - px * half_tail, hy - py * half_tail),
            QPointF(-px * half_tail, -py * half_tail),
        ]
        path.moveTo(pts[0])
        for p in pts[1:]:
            path.lineTo(p)
        path.closeSubpath()
        self._path = path
        pad = half_head + 0.5
        self._rect = QRectF(
            min(0.0, float(dx)) - pad,
            min(0.0, float(dy)) - pad,
            abs(float(dx)) + 2.0 * pad,
            abs(float(dy)) + 2.0 * pad,
        )
        self.setVisible(True)
        self.update()

    def boundingRect(self):
        return self._rect

    def paint(self, painter, option, widget=None):
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(self._color, 0))
        painter.setBrush(QBrush(self._color))
        painter.drawPath(self._path)


# ---------------------------------------------------------------------------
# PyQtGraph GUI
# ---------------------------------------------------------------------------

# Pressure slider range: allow over/undershoot; 0 = rest, [0,1] = dataset range
P_MIN, P_MAX = -1.0, 2.0
SLIDER_RESOLUTION = 300  # Steps for slider precision


class PressureSlider(QSlider):
    """Horizontal pressure slider with orange markers at dataset bounds."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p_low = DATASET_P_LOW
        self.p_high = 0.87

    def set_dataset_bounds(self, p_low, p_high):
        self.p_low = float(p_low)
        self.p_high = float(p_high)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        from PyQt5.QtGui import QPainter, QPen, QColor
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(*ORANGE_EXTRAP), 2))
        for p in (self.p_low, self.p_high):
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)
            opt.sliderPosition = int(round(p * SLIDER_RESOLUTION))
            opt.sliderValue = opt.sliderPosition
            handle = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self)
            groove = self.style().subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)
            x = handle.center().x()
            y0, y1 = groove.top(), min(self.height() - 2, groove.bottom() + 10)
            painter.drawLine(int(round(x)), y0, int(round(x)), y1)


def pressure_to_color(p):
    """Map pressure to RGB color (0=light blue, 1=dark blue). Clamp to [0,1]."""
    p_clamp = max(0.0, min(1.0, p))
    # Light blue to dark blue gradient
    r = int(255 * (1.0 - 0.75 * p_clamp))
    g = int(255 * (1.0 - 0.25 * p_clamp))
    b = 255
    return (r, g, b)


class PressureDiagram(QWidget):
    """Widget showing 2 or 4 pressure chambers with color coding (1-seg vs 2-seg)."""
    
    def __init__(self, actuation_dim=4):
        super().__init__()
        self.actuation_dim = actuation_dim
        self.pressures = [0.0] * max(2, min(4, actuation_dim))
        self.p_low = DATASET_P_LOW
        self.p_highs = [0.87] * max(2, min(4, actuation_dim))
        self.setMinimumSize(200, 200)
        self.setMaximumSize(250, 250)

    def set_dataset_bounds(self, p_low, p_highs):
        self.p_low = float(p_low)
        if np.isscalar(p_highs):
            self.p_highs = [float(p_highs)] * 4
        else:
            self.p_highs = [float(x) for x in p_highs]
        self.update()
    
    def set_actuation_dim(self, actuation_dim):
        """Switch between 2 (1-seg) and 4 (2-seg) chambers."""
        self.actuation_dim = actuation_dim
        n = max(2, min(4, actuation_dim))
        self.pressures = (self.pressures + [0.0] * 4)[:n]
        self.update()
    
    def update_pressures(self, pressures):
        """Update pressure values and redraw."""
        n = min(len(pressures), len(self.pressures))
        for i in range(n):
            self.pressures[i] = pressures[i]
        self.update()
    
    def paintEvent(self, event):
        """Draw 2 or 4 colored rectangles representing pressure chambers."""
        from PyQt5.QtGui import QPainter, QColor, QPen
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()
        n = self.actuation_dim
        if n == 2:
            cell_w = w // 2
            cell_h = h
            labels = PRESSURE_LABELS[:2]
            positions = [(0, 0), (cell_w, 0)]
        else:
            cell_w = w // 2
            cell_h = h // 2
            labels = PRESSURE_LABELS[:4]
            positions = [(0, 0), (cell_w, 0), (0, cell_h), (cell_w, cell_h)]
        
        for i in range(n):
            if i >= len(positions):
                break
            x, y = positions[i]
            label = labels[i] if i < len(labels) else f"p{i+1}"
            p = self.pressures[i] if i < len(self.pressures) else 0.0
            r, g, b = pressure_to_color(p)
            
            painter.fillRect(x, y, cell_w, cell_h, QColor(r, g, b))
            p_high = self.p_highs[i] if i < len(self.p_highs) else self.p_highs[-1]
            if p > p_high or p < self.p_low:
                pen = QPen(QColor(*ORANGE_EXTRAP), 3)
            else:
                pen = QPen(QColor(0, 0, 0), 2)
            painter.setPen(pen)
            painter.drawRect(x, y, cell_w, cell_h)
            painter.setPen(QColor(0, 0, 0))
            painter.setFont(QFont("Arial", 13, QFont.Bold))
            text_rect = painter.boundingRect(x, y, cell_w, cell_h, Qt.AlignCenter, label)
            painter.drawText(text_rect, Qt.AlignCenter, label)


class ModelLoaderThread(QThread):
    """Load model and compute rest state in background to avoid freezing the GUI."""
    # Emits: label, vae, dynamics, config, device, z, z_dot (or None,... on error)
    model_loaded = pyqtSignal(str, object, object, object, object, object, object)
    
    def __init__(self, suffix, label):
        super().__init__()
        self.suffix = suffix
        self.label = label
    
    def run(self):
        try:
            vae, dynamics, config, device = load_model_and_config(self.suffix)
            z, z_dot = get_rest_state_from_steady(vae, config, device)
            self.model_loaded.emit(
                self.label, vae, dynamics, config, device, z, z_dot
            )
        except Exception as e:
            print(f"Failed to load '{self.label}': {e}")
            self.model_loaded.emit(
                self.label, None, None, None, None, None, None
            )


class SimulationThread(QThread):
    """Dedicated thread for running simulation at precise 50 Hz."""
    
    simulation_updated = pyqtSignal(np.ndarray, list, float, np.ndarray, np.ndarray)  # image, pressures, fps, z_history, z_dot_history
    
    def __init__(self, current_state, z, z_dot):
        super().__init__()
        self.current = current_state
        self.z = z
        self.z_dot = z_dot
        adim = current_state.get("actuation_dim", 4)
        self.slider_values = [0.0] * adim
        self.running = True
        self.paused = False
        self.fps_state = [None, 50.0]
        self.sim_count = 0
        self.sim_start_time = time.perf_counter()
        self._lock = threading.Lock()
        self.model_lock = threading.Lock()  # VAE/dynamics: sim step vs GUI Jacobian
        self._state_generation = 0  # bumped by update_state; thread skips overwriting if it changed mid-step
        
        # History for plotting (1 second at 50 Hz = 50 samples)
        self.history_length = 50
        latent_dim = current_state["config"]["latent_dim"]
        # Initialize history with current state (not zeros)
        z_init = z.detach().cpu().numpy().squeeze()
        z_dot_init = z_dot.detach().cpu().numpy().squeeze()
        self.z_history = np.tile(z_init, (self.history_length, 1))
        self.z_dot_history = np.tile(z_dot_init, (self.history_length, 1))
    
    def update_sliders(self, values):
        """Thread-safe update of slider values."""
        with self._lock:
            self.slider_values = values.copy()
    
    def update_state(self, current_state, z, z_dot):
        """Thread-safe update of simulation state (for model changes/resets)."""
        with self._lock:
            self._state_generation += 1
            self.current = current_state
            self.z = z
            self.z_dot = z_dot
            self.fps_state = [None, 50.0]
            self.sim_count = 0
            self.sim_start_time = time.perf_counter()
            # Reset history with new latent dimension
            latent_dim = current_state["config"]["latent_dim"]
            # Initialize with current state values
            z_init = z.detach().cpu().numpy().squeeze()
            z_dot_init = z_dot.detach().cpu().numpy().squeeze()
            self.z_history = np.tile(z_init, (self.history_length, 1))
            self.z_dot_history = np.tile(z_dot_init, (self.history_length, 1))
    
    def run(self):
        """Main simulation loop. Rate = 50 * speed_factor Hz (speed_factor 0..1)."""
        base_dt = self.current["dt"]  # 0.02 for 50 Hz
        
        while self.running:
            if self.paused:
                time.sleep(0.01)
                continue
            
            speed_factor = self.current.get("speed_factor", 1.0)
            if speed_factor <= 0:
                time.sleep(0.02)
                continue
            
            dt_target = base_dt / speed_factor
            t_start = time.perf_counter()
            
            # Track FPS
            t_last, fps_smooth = self.fps_state[0], self.fps_state[1]
            if t_last is not None:
                dt_actual = t_start - t_last
                if dt_actual > 0:
                    fps_inst = 1.0 / dt_actual
                    fps_smooth = 0.9 * fps_smooth + 0.1 * fps_inst
                    self.fps_state[1] = fps_smooth
            self.fps_state[0] = t_start
            
            # Track average FPS
            self.sim_count += 1
            if self.sim_count % 100 == 0:
                elapsed = t_start - self.sim_start_time
                if elapsed > 0:
                    avg_fps = self.sim_count / elapsed
                    target_hz = 50.0 * speed_factor
                    if abs(avg_fps - target_hz) > max(2.0, target_hz * 0.1):
                        print(f"Simulation thread: Average rate is {avg_fps:.1f} Hz (target: {target_hz:.1f} Hz)")
            
            # Copy state and build u under lock (update u_history in place when delayed)
            with self._lock:
                gen_at_start = self._state_generation
                vae = self.current["vae"]
                dynamics = self.current["dynamics"]
                config = self.current["config"]
                device = self.current["device"]
                dt = self.current["dt"]
                uh = self.current.get("u_history")
                adim = self.current["actuation_dim"]
                slider_vals = self.slider_values.copy()
                z = self.z
                z_dot = self.z_dot
                z_history = self.z_history.copy()
                z_dot_history = self.z_dot_history.copy()
                # Skip building u if state and model disagree (avoid advancing u_history)
                expected_latent = config["latent_dim"]
                state_ok = (z.shape[1] == expected_latent and z_dot.shape[1] == expected_latent)
                if state_ok:
                    if uh is not None:
                        u_now = np.array(slider_vals[:adim], dtype=np.float32)
                        uh[:] = np.roll(uh, 1, axis=0)
                        uh[0] = u_now
                        u = torch.from_numpy(uh.flatten()).float().unsqueeze(0).to(device)
                    else:
                        u = build_u_from_sliders(slider_vals, config, device)
            
            # Skip step if state and model disagree (e.g. race right after model switch).
            # Sleep so we don't spin at full CPU and starve the main thread (which runs update_state).
            if not state_ok:
                time.sleep(0.02)
                continue
            
            # Simulation step. model_lock serializes with GUI Jacobian (force arrows).
            with self.model_lock:
                z_next, zd_next, img_tensor = simulation_step(vae, dynamics, z, z_dot, u, dt, device)
            
            z_np = z_next.detach().cpu().numpy().squeeze()
            z_dot_np = zd_next.detach().cpu().numpy().squeeze()
            z_history = np.roll(z_history, -1, axis=0)
            z_history[-1] = z_np
            z_dot_history = np.roll(z_dot_history, -1, axis=0)
            z_dot_history[-1] = z_dot_np
            
            img = img_tensor.detach().cpu().numpy().squeeze()
            if img.ndim == 3:
                img = img[0]
            img = img.T  # Transpose to match PyQtGraph's axis convention
            
            # Update shared state under lock (brief). If main thread called update_state
            # (model switch) during this step, do not overwrite with our stale result.
            with self._lock:
                if self._state_generation != gen_at_start:
                    continue  # skip overwrite and emit; next iteration will use new state
                self.z = z_next
                self.z_dot = zd_next
                self.z_history = z_history
                self.z_dot_history = z_dot_history
            
            # Emit update to GUI thread
            self.simulation_updated.emit(img, slider_vals, fps_smooth, z_history.copy(), z_dot_history.copy())
            
            # Precise timing: sleep for remaining time to maintain target rate (50 * speed_factor Hz)
            t_elapsed = time.perf_counter() - t_start
            t_remaining = dt_target - t_elapsed
            if t_remaining > 0:
                time.sleep(t_remaining)
            elif t_elapsed > dt_target * 1.5:
                print(f"Warning: Simulation step took {t_elapsed*1000:.1f}ms (target: {dt_target*1000:.1f}ms)")
    
    def stop(self):
        """Stop the simulation thread."""
        self.running = False


class LiveSimulationWindow(QWidget):
    """Main window for live simulation with PyQtGraph."""
    
    def __init__(self, models_by_segment, default_segment="2seg", default_variant="Koopman"):
        super().__init__()
        
        if not models_by_segment:
            raise ValueError("No models available. Train at least one model.")
        
        self.models_by_segment = models_by_segment
        if default_segment not in models_by_segment:
            default_segment = next(iter(models_by_segment))
        variants = dict(models_by_segment[default_segment])
        if default_variant not in variants:
            default_variant = next(iter(variants))
        self.current_segment = default_segment
        self.current_variant = default_variant
        self.name_to_suffix = variants
        self._model_loader = None
        
        # Load initial model
        default_suffix = variants[default_variant]
        vae, dynamics, config, device = load_model_and_config(default_suffix)
        dt = config.get("delta_t", 0.02)
        num_delays = config.get("num_actuation_delays", 1)
        actuation_dim = config["actuation_dim"]
        u_history = np.zeros((num_delays, actuation_dim), dtype=np.float32) if num_delays > 1 else None
        
        self.current = {
            "vae": vae,
            "dynamics": dynamics,
            "config": config,
            "device": device,
            "dt": dt,
            "num_delays": num_delays,
            "actuation_dim": actuation_dim,
            "u_history": u_history,
            "speed_factor": 1.0,  # 1 = 50 Hz, 0.2 = 10 Hz, 0 = no steps
        }
        
        self.z, self.z_dot = get_rest_state_from_steady(self.current["vae"], self.current["config"], self.current["device"])
        self._dataset_p_low = DATASET_P_LOW
        self._dataset_p_highs = dataset_pressure_highs(self.current["config"].get("dataset"))
        self.slider_values = [0.0] * self.current["actuation_dim"]
        self.saved_states = []  # list of {"z": np, "z_dot": np, "image": np (W,H)}; cleared on model change
        self._last_z = self.z.detach().cpu().numpy().squeeze().copy()
        self._last_z_dot = self.z_dot.detach().cpu().numpy().squeeze().copy()
        self._last_z_prev = None   # previous step state (for movement overlay red channel)
        self._last_z_dot_prev = None
        self._last_img = None  # set each frame in on_simulation_update (W, H)
        
        self.init_ui()
        
        # Use dedicated thread for true 50 Hz simulation
        self.sim_thread = SimulationThread(self.current, self.z, self.z_dot)
        self.sim_thread.simulation_updated.connect(self.on_simulation_update)
        self.sim_thread.start()
        
        print(f"Loaded: {SEGMENT_DISPLAY[self.current_segment]} / {self.current_variant}")
    
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Live simulation")
        self.setGeometry(100, 100, 900, 600)
        self.setFont(QFont("Arial", 12))
        
        # Main layout
        main_layout = QHBoxLayout()
        
        # Left panel: model selection and pressure diagram
        left_panel = QVBoxLayout()
        
        model_group_label = QLabel("Model")
        model_group_label.setFont(QFont("Arial", 14, QFont.Bold))
        left_panel.addWidget(model_group_label)
        
        self.segment_combo = QComboBox()
        self.segment_combo.setFont(QFont("Arial", 12))
        for key in SEGMENT_KEYS:
            if key in self.models_by_segment:
                self.segment_combo.addItem(SEGMENT_DISPLAY[key], key)
        idx = self.segment_combo.findData(self.current_segment)
        if idx >= 0:
            self.segment_combo.setCurrentIndex(idx)
        self.segment_combo.currentIndexChanged.connect(self.on_segment_change)
        left_panel.addWidget(self.segment_combo)
        
        self.model_radio_widget = QWidget()
        self.model_radio_layout = QVBoxLayout(self.model_radio_widget)
        self.model_radio_layout.setContentsMargins(0, 4, 0, 0)
        self.model_button_group = QButtonGroup(self)
        left_panel.addWidget(self.model_radio_widget)
        self._rebuild_model_radios(self.current_variant)
        
        self.model_loading_label = QLabel("")
        self.model_loading_label.setStyleSheet("color: gray; font-style: italic;")
        left_panel.addWidget(self.model_loading_label)
        
        left_panel.addSpacing(20)
        
        # Pressure diagram (2 chambers for 1-seg, 4 for 2-seg)
        diagram_label = QLabel("Pressures")
        diagram_label.setFont(QFont("Arial", 14, QFont.Bold))
        left_panel.addWidget(diagram_label)
        
        self.pressure_diagram = PressureDiagram(actuation_dim=self.current["actuation_dim"])
        self.pressure_diagram.set_dataset_bounds(self._dataset_p_low, self._dataset_p_highs)
        left_panel.addWidget(self.pressure_diagram)
        
        left_panel.addStretch()
        
        # Right panel: image display and controls
        right_panel = QVBoxLayout()
        
        # Create horizontal layout for image and plot
        vis_layout = QHBoxLayout()
        
        # Image display with pyqtgraph
        self.image_widget = pg.GraphicsLayoutWidget()
        self.view_box = self.image_widget.addViewBox()
        self.view_box.setAspectLocked(True)
        self.view_box.invertY(True)
        
        # Get initial image
        with torch.no_grad():
            img0 = self.current["vae"].decode(self.z).detach().cpu().numpy().squeeze()
        if img0.ndim == 3:
            img0 = img0[0]
        
        # PyQtGraph displays images differently than matplotlib - transpose to fix rotation
        self.image_item = pg.ImageItem(img0.T)
        self.view_box.addItem(self.image_item)
        self._last_img = img0.T.copy()  # (W, H) display order
        
        # Overlay: max deformation from static/dynamic dataset for current segment (cyan, toggleable)
        segment_key = get_segment_key(self.current["config"])
        self._overlay_static = load_max_deformation_overlay(segment_key, "static")   # (H, W) or None
        self._overlay_dynamic = load_max_deformation_overlay(segment_key, "dynamic")
        self._overlay_alpha = 0.75   # opacity of cyan overlay (stronger for visibility)
        h, w = img0.shape[0], img0.shape[1]
        overlay_src = self._overlay_dynamic if self._overlay_dynamic is not None else self._overlay_static
        overlay_img = self._make_overlay_rgba(overlay_src, (h, w))
        self.overlay_item = pg.ImageItem(overlay_img)
        self.overlay_item.setVisible(False)
        self.view_box.addItem(self.overlay_item)
        
        # Movement overlay (red = current - prev, green = next - current); visible only when paused
        self.movement_overlay_item = pg.ImageItem(np.zeros((w, h, 4), dtype=np.uint8))
        self.movement_overlay_item.setVisible(False)
        self.view_box.addItem(self.movement_overlay_item)

        # Total excitation (red) / stiffness (blue) arrows via RA-L attention-COM Jacobian
        self._force_arrow_scale = 0.04  # same mapping as Fig. 4(b)
        self._force_J = None
        self._force_J_time = 0.0
        self._force_origin = pg.ScatterPlotItem(
            size=9, brush=pg.mkBrush(128, 0, 128), pen=pg.mkPen("k", width=1), pxMode=True
        )
        self._force_origin.setZValue(21)
        self._force_origin.setVisible(False)
        self.view_box.addItem(self._force_origin)
        self._force_arrow_exc = ForceArrowItem((220, 40, 40))
        self._force_arrow_stiff = ForceArrowItem((40, 80, 200))
        self.view_box.addItem(self._force_arrow_exc)
        self.view_box.addItem(self._force_arrow_stiff)
        
        # Create vertical layout for plots (state and velocity)
        plots_layout = QVBoxLayout()
        
        # Latent state plot
        self.plot_widget_z = pg.PlotWidget()
        self.plot_widget_z.setBackground('w')
        _pg_label = {'color': '#000000', 'font-size': '13pt'}
        self.plot_widget_z.setLabel('left', 'z', **_pg_label)
        self.plot_widget_z.setLabel('bottom', 't', units='s', **_pg_label)
        self.plot_widget_z.showGrid(x=True, y=True, alpha=0.3)  # Enable grid
        self.plot_widget_z.setYRange(-3, 3)
        
        # Add zero reference line
        zero_line_z = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('k', width=1, style=Qt.DashLine))
        self.plot_widget_z.addItem(zero_line_z)
        
        # Create plot curves for each latent dimension
        latent_dim = self.current["config"]["latent_dim"]
        self.latent_curves = []
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']  # Color palette
        for i in range(latent_dim):
            color = colors[i % len(colors)]
            pen = pg.mkPen(color=color, width=2)
            curve = self.plot_widget_z.plot(pen=pen)
            self.latent_curves.append(curve)
        
        # Latent velocity plot
        self.plot_widget_zdot = pg.PlotWidget()
        self.plot_widget_zdot.setBackground('w')
        self.plot_widget_zdot.setLabel('left', 'ż', **_pg_label)
        self.plot_widget_zdot.setLabel('bottom', 't', units='s', **_pg_label)
        self.plot_widget_zdot.setTitle('ż (1 s)')
        self.plot_widget_zdot.showGrid(x=True, y=True, alpha=0.3)  # Enable grid
        self.plot_widget_zdot.setYRange(-10, 10)
        
        # Add zero reference line
        zero_line_zdot = pg.InfiniteLine(pos=0, angle=0, pen=pg.mkPen('k', width=1, style=Qt.DashLine))
        self.plot_widget_zdot.addItem(zero_line_zdot)
        
        # Create plot curves for each latent velocity dimension
        self.latent_vel_curves = []
        for i in range(latent_dim):
            color = colors[i % len(colors)]
            pen = pg.mkPen(color=color, width=2)
            curve = self.plot_widget_zdot.plot(pen=pen)
            self.latent_vel_curves.append(curve)
        
        # Time axis for 1 second at 50 Hz (0 to 1 second, 50 samples)
        self.time_axis = np.linspace(-1.0, 0.0, 50)
        
        plots_layout.addWidget(self.plot_widget_z)
        plots_layout.addWidget(self.plot_widget_zdot)
        
        self.zdot_norm_label = QLabel("‖ż‖ = 0.000")
        self.zdot_norm_label.setFont(QFont("Monospace", 13))
        plots_layout.addWidget(self.zdot_norm_label)
        
        vis_layout.addWidget(self.image_widget, 2)  # Image gets 2/3 of space
        vis_layout.addLayout(plots_layout, 1)       # Plots get 1/3 of space
        
        # Title and FPS label
        title_layout = QHBoxLayout()
        title_label = QLabel("Live simulation")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_layout.addWidget(title_label)
        
        title_layout.addStretch()
        
        self.fps_label = QLabel("50.0 FPS")
        self.fps_label.setFont(QFont("Monospace", 14, QFont.Bold))
        self.fps_label.setStyleSheet("color: green;")
        title_layout.addWidget(self.fps_label)
        
        right_panel.addLayout(title_layout)
        right_panel.addLayout(vis_layout)
        
        # Control buttons (Reset and Pause/Resume)
        button_layout = QHBoxLayout()
        
        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.do_reset)
        button_layout.addWidget(self.reset_button)
        
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)
        button_layout.addWidget(self.pause_button)
        
        # Keyboard 'p' = Pause/Resume
        QShortcut(QKeySequence("p"), self, self.toggle_pause)
        
        right_panel.addLayout(button_layout)
        
        # Save image / saved states list
        save_layout = QHBoxLayout()
        self.save_image_button = QPushButton("Save state")
        self.save_image_button.clicked.connect(self.save_current_state)
        save_layout.addWidget(self.save_image_button)
        self.save_as_button = QPushButton("Save as...")
        self.save_as_button.clicked.connect(self.save_states_as)
        save_layout.addWidget(self.save_as_button)
        save_layout.addStretch()
        right_panel.addLayout(save_layout)
        
        saved_label = QLabel("Saved")
        saved_label.setFont(QFont("Arial", 13, QFont.Bold))
        right_panel.addWidget(saved_label)
        self.saved_states_list = QListWidget()
        self.saved_states_list.setMaximumHeight(52)
        self.saved_states_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.saved_states_list.customContextMenuRequested.connect(self._show_saved_states_context_menu)
        self.saved_states_list.itemClicked.connect(self.on_saved_state_clicked)
        right_panel.addWidget(self.saved_states_list)
        
        # Deformation overlay: toggle and static/dynamic choice (only if overlay data loaded)
        overlay_available = self._overlay_static is not None or self._overlay_dynamic is not None
        overlay_layout = QHBoxLayout()
        self.overlay_check = QtWidgets.QCheckBox("Max deform. overlay")
        self.overlay_check.setChecked(False)
        self.overlay_check.setEnabled(overlay_available)
        self.overlay_check.toggled.connect(self._on_overlay_toggled)
        overlay_layout.addWidget(self.overlay_check)
        self.overlay_static_rb = QRadioButton("Static")
        self.overlay_dynamic_rb = QRadioButton("Dynamic")
        self.overlay_dynamic_rb.setChecked(True)
        self.overlay_static_rb.setEnabled(self._overlay_static is not None)
        self.overlay_dynamic_rb.setEnabled(self._overlay_dynamic is not None)
        self.overlay_static_rb.toggled.connect(lambda c: self._on_overlay_source_changed() if c else None)
        self.overlay_dynamic_rb.toggled.connect(lambda c: self._on_overlay_source_changed() if c else None)
        overlay_layout.addWidget(QLabel("Source:"))
        overlay_layout.addWidget(self.overlay_static_rb)
        overlay_layout.addWidget(self.overlay_dynamic_rb)
        overlay_layout.addStretch()
        right_panel.addLayout(overlay_layout)

        force_layout = QHBoxLayout()
        self.force_arrows_check = QCheckBox("Force arrows (Σ excitation / stiffness)")
        self.force_arrows_check.setChecked(False)
        self.force_arrows_check.setToolTip(
            "Total excitation (red) and stiffness (blue) through the attention-COM Jacobian.\n"
            "Available for Oscillator + ABCD. At equilibrium the two arrows should oppose."
        )
        self.force_arrows_check.toggled.connect(self._on_force_arrows_toggled)
        force_layout.addWidget(self.force_arrows_check)
        force_hint = QLabel("red = excitation, blue = stiffness")
        force_hint.setStyleSheet("color: gray; font-size: 11px;")
        force_layout.addWidget(force_hint)
        force_layout.addStretch()
        right_panel.addLayout(force_layout)
        self._sync_force_arrows_enabled()
        
        # Simulation speed factor (0 = 0 Hz, 1 = 50 Hz)
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(0)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setValue(100)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.setTickInterval(25)
        self.speed_slider.valueChanged.connect(self.on_speed_slider_change)
        speed_layout.addWidget(self.speed_slider)
        self.speed_label = QLabel("1.00 (50 Hz)")
        self.speed_label.setMinimumWidth(80)
        self.speed_label.setFont(QFont("Monospace", 12))
        speed_layout.addWidget(self.speed_label)
        right_panel.addLayout(speed_layout)
        
        # Pressure sliders (2 for 1-seg, 4 for 2-seg); container so we can rebuild on model change
        self.slider_container = QWidget()
        self.slider_container_layout = QVBoxLayout(self.slider_container)
        self.slider_container_layout.setContentsMargins(0, 0, 0, 0)
        self.sliders = []
        self._build_sliders(self.current["actuation_dim"])
        right_panel.addWidget(self.slider_container)
        
        # Assemble main layout
        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 3)
        
        self.setLayout(main_layout)
    
    def _rebuild_model_radios(self, checked_variant):
        """Rebuild radio buttons for the current segment (short names, no 1/2 Seg prefix)."""
        while self.model_radio_layout.count():
            item = self.model_radio_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                self.model_button_group.removeButton(w)
                w.deleteLater()
        self.name_to_suffix = dict(self.models_by_segment[self.current_segment])
        for i, (variant, _) in enumerate(self.models_by_segment[self.current_segment]):
            rb = QRadioButton(variant)
            rb.setFont(QFont("Arial", 12))
            rb.blockSignals(True)
            rb.setChecked(variant == checked_variant)
            rb.blockSignals(False)
            rb.toggled.connect(lambda checked, n=variant: self.on_model_change(n) if checked else None)
            self.model_button_group.addButton(rb, i)
            self.model_radio_layout.addWidget(rb)

    def on_segment_change(self, index):
        """Switch 1-seg / 2-seg models; keep the same dynamics variant when available."""
        key = self.segment_combo.itemData(index)
        if key is None or key == self.current_segment:
            return
        current_variant = self.current_variant
        self.current_segment = key
        variants = [v for v, _ in self.models_by_segment[key]]
        if current_variant not in variants:
            current_variant = variants[0]
        self._rebuild_model_radios(current_variant)
        self.on_model_change(current_variant)

    def _rebuild_latent_curves(self, latent_dim):
        """Recreate plot curves when switching 1-seg (k=8) vs 2-seg (k=10)."""
        for curve in self.latent_curves:
            self.plot_widget_z.removeItem(curve)
        for curve in self.latent_vel_curves:
            self.plot_widget_zdot.removeItem(curve)
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
        self.latent_curves = []
        self.latent_vel_curves = []
        for i in range(latent_dim):
            color = colors[i % len(colors)]
            pen = pg.mkPen(color=color, width=2)
            self.latent_curves.append(self.plot_widget_z.plot(pen=pen))
            self.latent_vel_curves.append(self.plot_widget_zdot.plot(pen=pen))

    def _clear_layout(self, layout):
        """Recursively clear layout and delete child widgets."""
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                self._clear_layout(item.layout())
    
    def _build_sliders(self, actuation_dim):
        """Build 2 or 4 pressure sliders in the slider container (clears existing)."""
        self._clear_layout(self.slider_container_layout)
        self.sliders = []
        labels = PRESSURE_LABELS[:actuation_dim]
        for i, label in enumerate(labels):
            slider_row = QHBoxLayout()
            label_widget = QLabel(label)
            label_widget.setMinimumWidth(36)
            slider_row.addWidget(label_widget)
            slider = PressureSlider(Qt.Horizontal)
            p_high = self._dataset_p_highs[i] if i < len(self._dataset_p_highs) else self._dataset_p_highs[-1]
            slider.set_dataset_bounds(self._dataset_p_low, p_high)
            slider.setMinimum(int(P_MIN * SLIDER_RESOLUTION))
            slider.setMaximum(int(P_MAX * SLIDER_RESOLUTION))
            slider.setValue(0)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(SLIDER_RESOLUTION)
            slider.valueChanged.connect(lambda val, idx=i: self.on_slider_change(idx, val))
            slider_row.addWidget(slider)
            value_label = QLabel("0.00")
            value_label.setMinimumWidth(50)
            value_label.setFont(QFont("Monospace", 12))
            slider_row.addWidget(value_label)
            self.sliders.append((slider, value_label))
            self.slider_container_layout.addLayout(slider_row)
    
    def _reload_overlays(self, segment_key):
        """Load overlay images for the given segment (1seg or 2seg). Update UI availability."""
        self._overlay_static = load_max_deformation_overlay(segment_key, "static")
        self._overlay_dynamic = load_max_deformation_overlay(segment_key, "dynamic")
        overlay_available = self._overlay_static is not None or self._overlay_dynamic is not None
        self.overlay_check.setEnabled(overlay_available)
        self.overlay_static_rb.setEnabled(self._overlay_static is not None)
        self.overlay_dynamic_rb.setEnabled(self._overlay_dynamic is not None)
        if not overlay_available:
            self.overlay_check.setChecked(False)
    
    def _make_overlay_rgba(self, max_img, shape_hw, current_img_display=None):
        """
        Build (W, H, 4) uint8 overlay from max deformation (H, W) in [0,1].
        Cyan overlay: blend gray where current≈max, cyan where they differ.
          t = |max - current| → overlay = (1-t)*gray + t*cyan (no threshold).
        If current_img_display is None: overlay is solid cyan.
        """
        h, w = shape_hw[0], shape_hw[1]
        out = np.zeros((h, w, 4), dtype=np.uint8)
        if max_img is not None and max_img.size > 0:
            ih, iw = max_img.shape[0], max_img.shape[1]
            if ih != h or iw != w:
                pad_top = max(0, (h - ih) // 2)
                pad_left = max(0, (w - iw) // 2)
                crop_top = max(0, (ih - h) // 2)
                crop_left = max(0, (iw - w) // 2)
                fit_h, fit_w = min(h, ih), min(w, iw)
                canvas = np.zeros((h, w), dtype=np.float32)
                canvas[pad_top : pad_top + fit_h, pad_left : pad_left + fit_w] = max_img[crop_top : crop_top + fit_h, crop_left : crop_left + fit_w]
                max_img = canvas
            alpha = (np.clip(max_img, 0, 1) * 255 * self._overlay_alpha).astype(np.uint8)
            if current_img_display is not None:
                # current_img_display is (W, H) from PyQtGraph; we need (H, W) to match max_img
                cur = np.asarray(current_img_display, dtype=np.float32)
                if cur.shape == (w, h):
                    cur_hw = cur.T  # (W,H) -> (H,W)
                elif cur.shape == (h, w):
                    cur_hw = cur.copy()
                else:
                    cur_hw = np.zeros((h, w), dtype=np.float32)
                if cur_hw.max() > 1:
                    cur_hw = np.clip(cur_hw, 0, 1)
                t = np.clip(np.abs(max_img.astype(np.float32) - cur_hw), 0, 1)
                gray, cyan = 128, 255
                out[:, :, 0] = ((1 - t) * gray).astype(np.uint8)
                out[:, :, 1] = ((1 - t) * gray + t * cyan).astype(np.uint8)
                out[:, :, 2] = ((1 - t) * gray + t * cyan).astype(np.uint8)
            else:
                out[:, :, 0] = 0
                out[:, :, 1] = 255
                out[:, :, 2] = 255
            out[:, :, 3] = alpha
        return out.transpose(1, 0, 2)
    
    def _build_u_for_step(self):
        """Build control tensor u as the simulation thread would (current sliders, with u_history if delayed)."""
        config = self.current["config"]
        device = self.current["device"]
        uh = self.current.get("u_history")
        adim = self.current["actuation_dim"]
        slider_vals = self.slider_values
        if uh is not None:
            u_now = np.array(slider_vals[:adim], dtype=np.float32)
            uh_copy = np.roll(uh.copy(), 1, axis=0)
            uh_copy[0] = u_now
            u = torch.from_numpy(uh_copy.flatten()).float().unsqueeze(0).to(device)
        else:
            u = build_u_from_sliders(slider_vals, config, device)
        return u
    
    def _compute_prev_next_decoded(self):
        """Compute prev and next decoded images. Uses actual current state (_last_z, _last_z_dot); next from dynamics.forward; prev from stored previous state or Euler backward. Returns (prev_hw, next_hw) each (H, W)."""
        dt = self.current["dt"]
        vae = self.current["vae"]
        dynamics = self.current["dynamics"]
        device = self.current["device"]
        # Use actual current state from the thread (what the displayed image corresponds to), not self.z/self.z_dot which are only updated on reset/load
        z_cur = torch.from_numpy(self._last_z.astype(np.float32)).unsqueeze(0).to(device)
        z_dot_cur = torch.from_numpy(self._last_z_dot.astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            if self._last_z_prev is not None:
                prev_z = torch.from_numpy(self._last_z_prev.astype(np.float32)).unsqueeze(0).to(device)
                prev_img = vae.decode(prev_z).detach().cpu().numpy().squeeze()
            else:
                prev_z = z_cur - dt * z_dot_cur
                prev_img = vae.decode(prev_z).detach().cpu().numpy().squeeze()
            u = self._build_u_for_step()
            z_next, _ = dynamics.forward(z_cur, z_dot_cur, u, dt)
            next_img = vae.decode(z_next).detach().cpu().numpy().squeeze()
        if prev_img.ndim == 3:
            prev_img = prev_img[0]
        if next_img.ndim == 3:
            next_img = next_img[0]
        return prev_img.astype(np.float32), next_img.astype(np.float32)
    
    def _update_movement_overlay(self):
        """When paused: show RGB overlay as in Latent_control.ipynb — R=prev, G=next, B=obs (current). Each normalized to [0,1]."""
        if not self.sim_thread.paused or self._last_img is None:
            self.movement_overlay_item.setVisible(False)
            return
        try:
            prev_hw, next_hw = self._compute_prev_next_decoded()
        except Exception:
            self.movement_overlay_item.setVisible(False)
            return
        # current in (H, W); _last_img is (W, H)
        cur_hw = self._last_img.T.astype(np.float32).copy()
        if cur_hw.max() > 1.0 or cur_hw.min() < 0.0:
            cur_hw = np.clip((cur_hw - cur_hw.min()) / (cur_hw.max() - cur_hw.min() + 1e-8), 0, 1)
        prev_hw = prev_hw.astype(np.float32)
        next_hw = next_hw.astype(np.float32)

        def normalize(img):
            img = np.asarray(img, dtype=np.float32)
            if img.max() > 1.0 or img.min() < 0.0:
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            return np.clip(img, 0, 1)

        img_prev = normalize(prev_hw)
        img_next = normalize(next_hw)
        img_obs = normalize(cur_hw)
        # R=prev, G=next, B=obs (matches notebook overlay)
        overlay = np.stack([img_prev, img_next, img_obs], axis=-1)
        overlay = np.clip(overlay, 0, 1)
        out = (overlay * 255).astype(np.uint8)
        self.movement_overlay_item.setImage(out.transpose(1, 0, 2))
        self.movement_overlay_item.setVisible(True)
    
    def _on_overlay_toggled(self, checked):
        self.overlay_item.setVisible(checked)
        if checked:
            self._on_overlay_source_changed()
    
    def _on_overlay_source_changed(self):
        if not self.overlay_check.isChecked():
            return
        max_img = self._overlay_dynamic if self.overlay_dynamic_rb.isChecked() else self._overlay_static
        h, w = self.current["config"].get("resolution", 32), self.current["config"].get("resolution", 32)
        overlay_img = self._make_overlay_rgba(max_img, (h, w))
        self.overlay_item.setImage(overlay_img)

    def _sync_force_arrows_enabled(self):
        """Enable the force-arrow checkbox only for Oscillator + ABCD."""
        ok = oscillator_force_arrows_supported(self.current["vae"], self.current["dynamics"])
        self.force_arrows_check.setEnabled(ok)
        if not ok:
            self.force_arrows_check.blockSignals(True)
            self.force_arrows_check.setChecked(False)
            self.force_arrows_check.blockSignals(False)
            self._hide_force_arrows()

    def _on_force_arrows_toggled(self, checked):
        if not checked:
            self._hide_force_arrows()
            return
        self._force_J = None
        self._update_force_arrows(force_jacobian=True)

    def _hide_force_arrows(self):
        self._force_arrow_exc.setVisible(False)
        self._force_arrow_stiff.setVisible(False)
        self._force_origin.setVisible(False)

    def _set_one_force_arrow(self, item, x0, y0, dx, dy):
        item.set_vector(x0, y0, dx, dy)

    def _update_force_arrows(self, force_jacobian=False):
        """Draw total excitation (red) and stiffness (blue) at the mean attention COM."""
        if not getattr(self, "force_arrows_check", None) or not self.force_arrows_check.isChecked():
            return
        vae = self.current["vae"]
        dyn = self.current["dynamics"]
        if not oscillator_force_arrows_supported(vae, dyn):
            self._hide_force_arrows()
            return
        if self._last_z is None:
            return
        if not hasattr(self, "sim_thread"):
            return
        osc = dyn.osc_net
        device = self.current["device"]
        n_nodes = int(self.current["config"]["latent_dim"]) // 2
        now = time.perf_counter()
        paused = bool(getattr(self.sim_thread, "paused", False))
        j_interval = 0.05 if paused else 0.15
        need_J = (
            force_jacobian
            or self._force_J is None
            or (now - self._force_J_time) > j_interval
        )
        u = torch.as_tensor(self.slider_values[: self.current["actuation_dim"]], dtype=torch.float32, device=device).unsqueeze(0)
        z_t = torch.as_tensor(self._last_z, dtype=torch.float32, device=device)
        try:
            with self.sim_thread.model_lock:
                if need_J:
                    self._force_J = models.com_jacobian_per_node(vae, z_t, device=device)
                    self._force_J_time = now
                with torch.no_grad():
                    attn_out = vae.decoder.get_attention_weights(
                        z_t.unsqueeze(0), return_background_weights=True, return_peak_location=True
                    )
                    peak = attn_out[2] if isinstance(attn_out, tuple) and len(attn_out) == 3 else attn_out[-1]
                    com = peak[0, :n_nodes].detach().cpu().numpy()
                    F_exc = osc.control_to_state(u).detach().cpu().numpy()[0].reshape(n_nodes, 2)
                K_np = osc.give_Minv_KD()[1].detach().cpu().numpy()
                x0 = osc.x0.detach().cpu().numpy()
        except Exception as exc:
            print(f"Force arrows: {exc}")
            self._hide_force_arrows()
            return
        x_latent = np.asarray(self._last_z[: n_nodes * 2], dtype=np.float64)
        F_stiff = (-K_np @ (x_latent - x0)).reshape(n_nodes, 2)
        F_exc_img = models.latent_forces_to_image_jacobian(F_exc, self._force_J)
        F_stiff_img = models.latent_forces_to_image_jacobian(F_stiff, self._force_J)
        img_h = img_w = int(self.current["config"].get("resolution", 32))
        x_pix = ((com[:, 1] + 1.0) / 2.0) * (img_w - 1)
        y_pix = ((com[:, 0] + 1.0) / 2.0) * (img_h - 1)
        x0p, y0p = float(np.mean(x_pix)), float(np.mean(y_pix))
        dx_e, dy_e = models.image_force_arrow_xy(F_exc_img.sum(axis=0), img_h, img_w)
        dx_s, dy_s = models.image_force_arrow_xy(F_stiff_img.sum(axis=0), img_h, img_w)
        s = self._force_arrow_scale
        self._set_one_force_arrow(self._force_arrow_exc, x0p, y0p, float(dx_e) * s, float(dy_e) * s)
        self._set_one_force_arrow(self._force_arrow_stiff, x0p, y0p, float(dx_s) * s, float(dy_s) * s)
        self._force_origin.setData([x0p], [y0p])
        self._force_origin.setVisible(True)
    
    def on_speed_slider_change(self, value):
        """Handle speed factor slider (0–100 -> 0–1)."""
        factor = value / 100.0
        self.current["speed_factor"] = factor
        hz = 50.0 * factor if factor > 0 else 0.0
        self.speed_label.setText(f"{factor:.2f} ({hz:.0f} Hz)")
    
    def on_slider_change(self, idx, value):
        """Handle slider value change."""
        pressure = value / SLIDER_RESOLUTION
        self.slider_values[idx] = pressure
        self.sliders[idx][1].setText(f"{pressure:.2f}")
        if not hasattr(self, "sim_thread"):
            return
        self.sim_thread.update_sliders(self.slider_values)
        if self.force_arrows_check.isChecked():
            self._update_force_arrows()
    
    def do_reset(self):
        """Reset state to rest: z/z_dot from current model, sliders to 0."""
        vae = self.current["vae"]
        config = self.current["config"]
        device = self.current["device"]
        adim = self.current["actuation_dim"]
        self.z, self.z_dot = get_rest_state_from_steady(vae, config, device)
        
        # Reset sliders (number matches current model: 2 or 4)
        self.slider_values = [0.0] * adim
        for slider, value_label in self.sliders:
            slider.blockSignals(True)
            slider.setValue(0)
            slider.blockSignals(False)
            value_label.setText("0.00")
        
        # Reset u_history if using delayed actuation
        uh = self.current.get("u_history")
        if uh is not None:
            uh.fill(0)
        
        # Update pressure diagram
        self.pressure_diagram.update_pressures(self.slider_values)
        
        # Update image
        with torch.no_grad():
            img0 = vae.decode(self.z).detach().cpu().numpy().squeeze()
        if img0.ndim == 3:
            img0 = img0[0]
        self.image_item.setImage(img0.T)  # Transpose for PyQtGraph
        self._last_img = img0.T.copy()
        self._last_z = self.z.detach().cpu().numpy().squeeze().copy()
        self._last_z_dot = self.z_dot.detach().cpu().numpy().squeeze().copy()
        self._last_z_prev = None
        self._last_z_dot_prev = None
        
        # Update simulation thread
        self.sim_thread.update_state(self.current, self.z, self.z_dot)
        self.sim_thread.update_sliders(self.slider_values)
        if self.sim_thread.paused:
            self._update_movement_overlay()
        self.zdot_norm_label.setText(f"‖ż‖ = {float(np.linalg.norm(self.z_dot.detach().cpu().numpy())):.3f}")
        self._update_force_arrows(force_jacobian=True)
    
    def toggle_pause(self):
        """Toggle pause/resume of the simulation."""
        self.sim_thread.paused = not self.sim_thread.paused
        if self.sim_thread.paused:
            self.pause_button.setText("Resume")
            self._update_movement_overlay()
            self._update_force_arrows(force_jacobian=True)
        else:
            self.pause_button.setText("Pause")
            self.movement_overlay_item.setVisible(False)
    
    def save_current_state(self):
        """Append current decoded image, (z, z_dot), and u to the saved-states list."""
        if self._last_img is None:
            return
        adim = self.current["actuation_dim"]
        entry = {
            "z": self._last_z.copy(),
            "z_dot": self._last_z_dot.copy(),
            "image": self._last_img.copy(),
            "u": np.array(self.slider_values[:adim], dtype=np.float32),
        }
        self.saved_states.append(entry)
        self.saved_states_list.addItem(QListWidgetItem(f"State {len(self.saved_states)}"))
    
    def _show_saved_states_context_menu(self, position):
        """Show right-click menu: Delete for the selected state."""
        item = self.saved_states_list.itemAt(position)
        if item is None:
            return
        menu = QMenu(self)
        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(lambda: self._delete_saved_state_at_row(self.saved_states_list.row(item)))
        menu.addAction(delete_action)
        menu.exec_(self.saved_states_list.mapToGlobal(position))
    
    def _delete_saved_state_at_row(self, row):
        """Remove the saved state at the given row from the list and from saved_states."""
        if row < 0 or row >= len(self.saved_states):
            return
        self.saved_states.pop(row)
        self.saved_states_list.takeItem(row)
        for i in range(row, self.saved_states_list.count()):
            self.saved_states_list.item(i).setText(f"State {i + 1}")
    
    def save_states_as(self):
        """Save the current list of states to a file as a list of dicts: obs_decoded, z, z_dot, u, prev_decoded, next_decoded."""
        if not self.saved_states:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save states as", "", "Pickle (*.pkl);;All files (*)")
        if not path:
            return
        adim = self.current["actuation_dim"]
        dt = self.current["dt"]
        vae = self.current["vae"]
        dynamics = self.current["dynamics"]
        device = self.current["device"]
        uh = self.current.get("u_history")
        num_delays = 1 if uh is None else uh.shape[0]
        list_of_dicts = []
        with torch.no_grad():
            for entry in self.saved_states:
                z_t = torch.from_numpy(entry["z"].astype(np.float32)).unsqueeze(0).to(device)
                z_dot_t = torch.from_numpy(entry["z_dot"].astype(np.float32)).unsqueeze(0).to(device)
                prev_z = z_t - dt * z_dot_t
                prev_dec = vae.decode(prev_z).detach().cpu().numpy().squeeze()
                if prev_dec.ndim == 3:
                    prev_dec = prev_dec[0]
                u_np = entry.get("u", np.zeros(adim, dtype=np.float32))
                if np.asarray(u_np).shape != (adim,):
                    u_np = np.array(self.slider_values[:adim], dtype=np.float32)
                u_np = np.asarray(u_np, dtype=np.float32).reshape(-1)
                if num_delays > 1:
                    u_flat = np.tile(u_np[:adim], num_delays).astype(np.float32)
                else:
                    u_flat = u_np[:adim]
                u_t = torch.from_numpy(u_flat).float().unsqueeze(0).to(device)
                z_next, _ = dynamics.forward(z_t, z_dot_t, u_t, dt)
                next_dec = vae.decode(z_next).detach().cpu().numpy().squeeze()
                if next_dec.ndim == 3:
                    next_dec = next_dec[0]
                obs = entry["image"].T.copy().astype(np.float32)
                list_of_dicts.append({
                    "obs_decoded": obs,
                    "z": entry["z"].astype(np.float32),
                    "z_dot": entry["z_dot"].astype(np.float32),
                    "u": u_np[:adim].astype(np.float32),
                    "prev_decoded": prev_dec.astype(np.float32),
                    "next_decoded": next_dec.astype(np.float32),
                })
        with open(path, "wb") as f:
            pickle.dump(list_of_dicts, f)
        print(f"Saved {len(list_of_dicts)} states to {path}")
    
    def on_saved_state_clicked(self, item):
        """Load the clicked saved state and switch to pause mode."""
        row = self.saved_states_list.row(item)
        if row < 0 or row >= len(self.saved_states):
            return
        entry = self.saved_states[row]
        device = self.current["device"]
        self.z = torch.from_numpy(entry["z"].astype(np.float32)).unsqueeze(0).to(device)
        self.z_dot = torch.from_numpy(entry["z_dot"].astype(np.float32)).unsqueeze(0).to(device)
        self.image_item.setImage(entry["image"])
        self._last_z = entry["z"].copy()
        self._last_z_dot = entry["z_dot"].copy()
        self._last_z_prev = None
        self._last_z_dot_prev = None
        self._last_img = entry["image"].copy()
        if not self.sim_thread.paused:
            self.toggle_pause()
        self.sim_thread.update_state(self.current, self.z, self.z_dot)
        self.sim_thread.update_sliders(self.slider_values)
        # Update latent plots to show loaded state (flat history)
        n = len(self.time_axis)
        for i, curve in enumerate(self.latent_curves):
            if i < entry["z"].shape[0]:
                curve.setData(self.time_axis, np.full(n, entry["z"][i]))
        for i, curve in enumerate(self.latent_vel_curves):
            if i < entry["z_dot"].shape[0]:
                curve.setData(self.time_axis, np.full(n, entry["z_dot"][i]))
        self.zdot_norm_label.setText(f"‖ż‖ = {np.linalg.norm(entry['z_dot']):.3f}")
        self._update_movement_overlay()
        self._update_force_arrows(force_jacobian=True)
    
    def _set_model_loading(self, loading):
        """Enable/disable model selection and show loading state."""
        for btn in self.model_button_group.buttons():
            btn.setEnabled(not loading)
        if hasattr(self, "segment_combo"):
            self.segment_combo.setEnabled(not loading)
        self.model_loading_label.setText("Loading..." if loading else "")
        if loading:
            QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        else:
            QtWidgets.QApplication.restoreOverrideCursor()
    
    def _on_loader_finished(self, loader):
        """Only clear _model_loader if this loader is still the current one (avoids clearing a newer loader when an old loader's finished fires late)."""
        if self._model_loader is loader:
            self._model_loader = None

    def on_model_loaded(self, label, vae, dynamics, config, device, z, z_dot):
        """Apply loaded model on the GUI thread (called from loader thread signal)."""
        self._set_model_loading(False)
        # Do not set _model_loader = None here: the thread may still be shutting down; clearing the reference can cause "QThread: Destroyed while thread is still running". We clear it in _on_loader_finished when the thread has actually finished.
        if vae is None:
            return
        # Rebuild models on main thread so the sim thread uses main-thread-created
        # instances (avoids CUDA/thread-context hangs when sim thread runs forward).
        vae, dynamics = build_models_on_device(config, device, source_vae=vae, source_dynamics=dynamics)
        dt = config.get("delta_t", 0.02)
        nd = config.get("num_actuation_delays", 1)
        adim = config["actuation_dim"]
        uh = np.zeros((nd, adim), dtype=np.float32) if nd > 1 else None
        # Build a new current dict and update sim thread atomically. Do not mutate
        # self.current in place: the sim thread shares that dict and would then see
        # new vae/dynamics with old z/z_dot and crash (shape mismatch).
        new_current = {
            "vae": vae,
            "dynamics": dynamics,
            "config": config,
            "device": device,
            "dt": dt,
            "num_delays": nd,
            "actuation_dim": adim,
            "u_history": uh,
            "speed_factor": self.current.get("speed_factor", 1.0),
        }
        self.current = new_current
        self.z, self.z_dot = z, z_dot
        self.slider_values = [0.0] * adim
        # Update sim thread first so it never overwrites new state with a stale step result.
        self.sim_thread.update_state(self.current, self.z, self.z_dot)
        self.sim_thread.update_sliders(self.slider_values)
        if uh is not None:
            uh.fill(0)
        self.saved_states.clear()
        self.saved_states_list.clear()
        segment_key = get_segment_key(config)
        self._reload_overlays(segment_key)
        self._dataset_p_highs = dataset_pressure_highs(config.get("dataset"))
        self.pressure_diagram.set_actuation_dim(adim)
        self.pressure_diagram.set_dataset_bounds(self._dataset_p_low, self._dataset_p_highs)
        self._build_sliders(adim)
        self._rebuild_latent_curves(config["latent_dim"])
        for slider, value_label in self.sliders:
            slider.blockSignals(True)
            slider.setValue(0)
            slider.blockSignals(False)
            value_label.setText("0.00")
        self.pressure_diagram.update_pressures(self.slider_values)
        with torch.no_grad():
            img0 = vae.decode(self.z).detach().cpu().numpy().squeeze()
        if img0.ndim == 3:
            img0 = img0[0]
        self.image_item.setImage(img0.T)
        self._last_img = img0.T.copy()
        self._last_z = self.z.detach().cpu().numpy().squeeze().copy()
        self._last_z_dot = self.z_dot.detach().cpu().numpy().squeeze().copy()
        self._last_z_prev = None
        self._last_z_dot_prev = None
        if self.sim_thread.paused:
            self._update_movement_overlay()
        self.zdot_norm_label.setText(f"‖ż‖ = {float(np.linalg.norm(self.z_dot.detach().cpu().numpy())):.3f}")
        self._force_J = None
        self._sync_force_arrows_enabled()
        self._update_force_arrows(force_jacobian=True)
        print(f"Loaded: {SEGMENT_DISPLAY.get(self.current_segment, self.current_segment)} / {label}")
    
    def on_model_change(self, label):
        """Start loading the selected model in a background thread."""
        suffix = self.name_to_suffix.get(label)
        if suffix is None:
            return
        self.current_variant = label
        if self._model_loader is not None and self._model_loader.isRunning():
            return
        self._set_model_loading(True)
        loader = ModelLoaderThread(suffix, label)
        loader.model_loaded.connect(self.on_model_loaded)
        loader.finished.connect(lambda l=loader: self._on_loader_finished(l))
        self._model_loader = loader
        loader.start()
    
    def on_simulation_update(self, img, pressures, fps, z_history, z_dot_history):
        """Handle simulation update from thread (called in GUI thread)."""
        # Ignore stale emissions from the previous model (e.g. after 1seg -> 2seg switch
        # a queued signal can overwrite the new model's image with the old one).
        expected_latent = self.current["config"]["latent_dim"]
        if z_history.shape[1] != expected_latent or z_dot_history.shape[1] != expected_latent:
            return
        self._last_z_prev = self._last_z.copy() if self._last_z is not None else None
        self._last_z_dot_prev = self._last_z_dot.copy() if self._last_z_dot is not None else None
        self._last_z = z_history[-1].copy()
        self._last_z_dot = z_dot_history[-1].copy()
        self._last_img = img.copy()
        # Update image
        self.image_item.setImage(img)
        
        # Update overlay with gray/cyan blend from |max - current|
        if self.overlay_check.isChecked():
            max_img = self._overlay_dynamic if self.overlay_dynamic_rb.isChecked() else self._overlay_static
            if max_img is not None:
                h, w = self.current["config"].get("resolution", 32), self.current["config"].get("resolution", 32)
                overlay_img = self._make_overlay_rgba(max_img, (h, w), current_img_display=img)
                self.overlay_item.setImage(overlay_img)
        
        # Update pressure diagram
        self.pressure_diagram.update_pressures(pressures)
        
        # Update FPS display
        self.fps_label.setText(f"{fps:.1f} FPS")
        
        # Update latent state plot
        for i, curve in enumerate(self.latent_curves):
            if i < z_history.shape[1]:
                curve.setData(self.time_axis, z_history[:, i])
        
        # Update latent velocity plot
        for i, curve in enumerate(self.latent_vel_curves):
            if i < z_dot_history.shape[1]:
                curve.setData(self.time_axis, z_dot_history[:, i])
        
        # Update norm of z_dot (live)
        zdot_norm = float(np.linalg.norm(z_dot_history[-1]))
        self.zdot_norm_label.setText(f"‖ż‖ = {zdot_norm:.3f}")
        self._update_force_arrows()
    
    def closeEvent(self, event):
        """Clean up simulation thread when window closes."""
        self.sim_thread.stop()
        self.sim_thread.wait()
        event.accept()


def run_live_simulation(models_by_segment, default_segment="2seg", default_variant="Koopman"):
    """Run the live simulation GUI with PyQtGraph."""
    app = QtWidgets.QApplication([])
    window = LiveSimulationWindow(models_by_segment, default_segment, default_variant)
    window.show()
    app.exec_()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_available_models_by_segment():
    """Return {segment_key: [(variant, suffix), ...]} for trained runs."""
    by_seg = {}
    for variant, suffixes in MODEL_VARIANTS:
        for seg, suffix in suffixes.items():
            if find_latest_model_run(suffix) is None:
                continue
            by_seg.setdefault(seg, []).append((variant, suffix))
    return by_seg


def resolve_default_model(models_by_segment, model_arg):
    """Pick (segment, variant) from --model name/suffix, else prefer 2-seg Koopman."""
    suffix_to_key = {}
    for variant, suffixes in MODEL_VARIANTS:
        for seg, suffix in suffixes.items():
            suffix_to_key[suffix] = (seg, variant)

    if model_arg:
        if model_arg in _OLD_MODEL_NAME_MAP:
            seg, variant = _OLD_MODEL_NAME_MAP[model_arg]
            if seg in models_by_segment and any(v == variant for v, _ in models_by_segment[seg]):
                return seg, variant
        if model_arg in suffix_to_key:
            seg, variant = suffix_to_key[model_arg]
            if seg in models_by_segment and any(v == variant for v, _ in models_by_segment[seg]):
                return seg, variant
        for seg, pairs in models_by_segment.items():
            for variant, suffix in pairs:
                if model_arg in (variant, f"{SEGMENT_DISPLAY[seg]} {variant}", suffix):
                    return seg, variant
        print(f"Warning: '{model_arg}' not in available models; using default.")

    if "2seg" in models_by_segment:
        variants = [v for v, _ in models_by_segment["2seg"]]
        variant = "Koopman" if "Koopman" in variants else variants[0]
        return "2seg", variant
    seg = next(iter(models_by_segment))
    return seg, models_by_segment[seg][0][0]


def main():
    parser = argparse.ArgumentParser(
        description="Live soft robot simulation with PyQtGraph (consistent 50 Hz timing)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Default model: variant (e.g. 'Koopman + ABCD'), old name, or run suffix.",
    )
    args = parser.parse_args()

    models_by_segment = get_available_models_by_segment()
    if not models_by_segment:
        raise SystemExit(
            "No trained models found. Train at least one model (results/models/<run>/epoch_*/). "
            "Model suffixes are defined in MODEL_VARIANTS."
        )

    default_segment, default_variant = resolve_default_model(models_by_segment, args.model)
    print("Available models:")
    for seg in SEGMENT_KEYS:
        if seg in models_by_segment:
            print(f"  {SEGMENT_DISPLAY[seg]}: {[v for v, _ in models_by_segment[seg]]}")
    print(f"Default: {SEGMENT_DISPLAY[default_segment]} / {default_variant}")

    run_live_simulation(models_by_segment, default_segment, default_variant)


if __name__ == "__main__":
    main()