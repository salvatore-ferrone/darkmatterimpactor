#!/usr/bin/env python3
"""
Test script for the new mw_model.py module.

Reproduces the sanity check from compute_orbits.py:
- Load DM subhalo data
- Rescale positions and velocities
- Compute bound orbits
- Plot E-Lz diagram
"""

import numpy as np
import h5py
import os
from mw_model import get_mw_model

def test_mw_model():
    """Test the MW model with the existing DM subhalo data."""
    
    # Get MW model
    mw = get_mw_model()
    print(f"Testing MW model: {mw.name}")
    print(f"Parameters: ahalo={mw.ahalo:.1f}, rH={mw.rH:.1f}, zH={mw.zH:.1f}")
    
    # Load subhalo data (same as compute_orbits.py)
    infilename = "../inputdata/DMsubhaloMW634Norm26PB.hdf5"
    if not os.path.exists(infilename):
        print(f"Input file {infilename} not found. Skipping test.")
        return
    
    with h5py.File(infilename, 'r') as myfile:
        x = myfile['xh'][:]
        y = myfile['yh'][:]
        z = myfile['zh'][:]
        vx = myfile['vxh'][:]
        vy = myfile['vyh'][:]
        vz = myfile['vzh'][:]
        mh = myfile['mh'][:]
    
    print(f"Loaded {len(x)} subhalos")
    
    # Rescale positions and velocities (same as compute_orbits.py)
    ahalo = mw.ahalo
    xh, yh, zh = x * ahalo, y * ahalo, z * ahalo
    
    # Get circular velocity at each point
    fx, fy, fz, PHI = mw.potential(xh, yh, zh)
    gmag = np.sqrt(fx**2 + fy**2 + fz**2)
    R = np.sqrt(xh**2 + yh**2 + zh**2)
    vcirc = np.sqrt(gmag * R)
    
    # Rescale velocities
    vxh, vyh, vzh = vx * vcirc, vy * vcirc, vz * vcirc
    
    positions = (xh, yh, zh)
    velocities = (vxh, vyh, vzh)
    
    # Compute bound mask
    bound = mw.get_bound_mask(positions, velocities)
    n_bound = np.sum(bound)
    print(f"Bound subhalos: {n_bound} / {len(bound)}")
    
    # Plot E-Lz diagram
    os.makedirs("../plots", exist_ok=True)
    mw.plot_energy_angular_momentum(positions, velocities, 
                                   filename="../plots/test_E_Lz.png")
    print("Saved E-Lz plot to plots/test_E_Lz.png")
    
    # Test circular velocity curve
    r, vcirc_curve = mw.circular_velocity_curve(rmax=50)
    print(f"Circular velocity at R=8 kpc: {np.interp(8, r, vcirc_curve):.1f} km/s")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    test_mw_model()