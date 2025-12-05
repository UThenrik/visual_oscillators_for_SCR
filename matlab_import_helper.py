"""
Helper functions for importing MATLAB data files from Simulink simulations.
Designed for SCR (Soft Continuum Robot) data structure.
"""

import scipy.io
import numpy as np
from typing import Dict, Tuple

def load_scr_matlab_data(file_path: str) -> Dict[str, Dict]:
    """Load and parse SCR MATLAB data file with simplified structure."""
    mat_data = scipy.io.loadmat(file_path)
    clean_data = {k: v for k, v in mat_data.items() if not k.startswith("__")}
    
    parsed_data = {}
    for group_name, group_data in clean_data.items():
        # Extract the struct data
        struct_data = group_data[0, 0]
        
        # Simple structure: time, p_is, p_set
        parsed_data[group_name] = {
            'time': struct_data['time'].flatten(),
            'signals': {
                'p_is': struct_data['p_is'],
                'p_set': struct_data['p_set'] if 'p_set' in struct_data.dtype.names else None
            }
        }
    
    return parsed_data

def get_scope_data(data: Dict[str, Dict]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Extract scope data from parsed data."""
    # Get the first (and typically only) group
    group_name = list(data.keys())[0]
    group_data = data[group_name]
    
    time = group_data['time']
    signals = {}
    
    for signal_name, signal_values in group_data['signals'].items():
        if signal_values is not None:
            signals[signal_name] = signal_values
    
    return time, signals

def print_data_summary(data: Dict[str, Dict]) -> None:
    """Print summary of loaded data."""
    print("=== SCR MATLAB Data Summary ===")
    
    for group_name, group_data in data.items():
        print(f"\n{group_name}:")
        print(f"  Time points: {len(group_data['time'])}")
        print(f"  Duration: {group_data['time'][-1] - group_data['time'][0]:.2f} seconds")
        print(f"  Signals: {list(group_data['signals'].keys())}")
        
        for signal_name, signal_values in group_data['signals'].items():
            if signal_values is not None:
                print(f"    {signal_name}: shape={signal_values.shape}, range=[{signal_values.min():.3f}, {signal_values.max():.3f}]")
