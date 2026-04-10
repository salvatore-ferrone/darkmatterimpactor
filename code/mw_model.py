"""
Milky Way model module for the dark matter stream gap prediction project.

This module encapsulates the analytical Milky Way potential model and related utilities
for orbit calculations, energy-angular momentum analysis, and visualization.

Based on the existing code in compute_orbits.py, using tstrippy.pouliasis2017pii potential.
"""

import numpy as np
import matplotlib.pyplot as plt
import tstrippy
import astropy.units as u

# Set matplotlib parameters for consistent plotting
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 16
})

class MilkyWayModel:
    """
    Encapsulates the Milky Way potential model and utilities.
    
    Uses the Pouliasis et al. 2017 potential as implemented in tstrippy.
    """
    
    def __init__(self):
        """Initialize the MW model with default parameters."""
        self.params = tstrippy.Parsers.pouliasis2017pii()
        self.potential_func = tstrippy.potentials.pouliasis2017pii
        self.name = "pouliasis2017pii"
        
        # Extract key parameters
        self.ahalo = self.params[2]  # Scale radius
        self.rH = self.params[6]     # Disk scale radius
        self.zH = self.params[-4]    # Disk scale height
        
    def potential(self, x, y, z):
        """
        Compute the gravitational potential and forces.
        
        Parameters:
        x, y, z : float or array
            Positions in kpc
            
        Returns:
        fx, fy, fz, PHI : arrays
            Forces and potential
        """
        return self.potential_func(self.params, x, y, z)
    
    def circular_velocity_curve(self, rmax=300, npoints=1000):
        """
        Compute the circular velocity curve.
        
        Parameters:
        rmax : float
            Maximum radius in kpc
        npoints : int
            Number of points
            
        Returns:
        r : array
            Radii
        vcirc : array
            Circular velocities
        """
        r = np.linspace(0, rmax, npoints)
        x, y, z = r, np.zeros_like(r), np.zeros_like(r)
        fx, fy, fz, PHI = self.potential(x, y, z)
        gmag = np.sqrt(fx**2 + fy**2 + fz**2)
        vcirc = np.sqrt(gmag * r)
        return r, vcirc
    
    def energy_angular_momentum_curve(self, rmax=300, npoints=1000):
        """
        Compute the E-Lz curve for circular orbits.
        
        Parameters:
        rmax : float
            Maximum radius in kpc
        npoints : int
            Number of points
            
        Returns:
        E_circ, Lz_circ : arrays
            Energy and angular momentum for circular orbits
        """
        r, vcirc = self.circular_velocity_curve(rmax, npoints)
        x, y, z = r, np.zeros_like(r), np.zeros_like(r)
        fx, fy, fz, PHI = self.potential(x, y, z)
        E_circ = 0.5 * vcirc**2 + PHI
        Lz_circ = r * vcirc
        return E_circ, Lz_circ
    
    def compute_energy_angular_momentum(self, positions, velocities):
        """
        Compute energy and angular momentum for given positions and velocities.
        
        Parameters:
        positions : tuple of arrays (x, y, z)
            Positions in kpc
        velocities : tuple of arrays (vx, vy, vz)
            Velocities in km/s
            
        Returns:
        E, Lz : arrays
            Total energy and z-component of angular momentum
        """
        x, y, z = positions
        vx, vy, vz = velocities
        
        # Kinetic energy
        vs = np.sqrt(vx**2 + vy**2 + vz**2)
        T = 0.5 * vs**2
        
        # Potential energy
        fx, fy, fz, PHI = self.potential(x, y, z)
        
        # Total energy
        E = T + PHI
        
        # Angular momentum (z-component)
        Lz = x * vy - y * vx
        
        return E, Lz
    
    def plot_energy_angular_momentum(self, positions, velocities, filename=None, show_unbound=True):
        """
        Plot E-Lz diagram for given positions and velocities.
        
        Parameters:
        positions : tuple of arrays (x, y, z)
        velocities : tuple of arrays (vx, vy, vz)
        filename : str, optional
            If provided, save plot to this file
        show_unbound : bool
            Whether to show unbound particles
        """
        AXIS = {
            "xlabel": r'$L_z$ [$10^4$ kpc km/s]', 
            "ylabel": r'$E$ [$10^4$ km$^2$/s$^2$]',
            "title": r"$\Lambda$CDM subhalo orbits in the MW potential"
        }
        factor = 1e4
        
        E, Lz = self.compute_energy_angular_momentum(positions, velocities)
        bound = E < 0
        n_bound = np.sum(bound)
        n_total = len(E)
        
        # Get circular orbit curve
        E_circ, Lz_circ = self.energy_angular_momentum_curve()
        
        # Set axis limits
        xmax = np.max(Lz_circ / factor)
        xmin = -xmax
        ymin = np.min(E_circ / factor)
        ymax = np.max(E / factor)
        AXIS["xlim"] = (xmin, xmax)
        AXIS["ylim"] = (ymin, ymax)
        
        fig, axis = plt.subplots(1, 1, figsize=(8.1, 4))
        
        # Plot circular orbit curves
        axis.plot(Lz_circ / factor, E_circ / factor, color='k', ls='-')
        axis.plot(-Lz_circ / factor, E_circ / factor, color='k', ls='-')
        
        # Plot bound particles
        axis.scatter(Lz[bound] / factor, E[bound] / factor, s=10, label='Bound')
        
        # Plot unbound particles if requested
        if show_unbound:
            axis.scatter(Lz[~bound] / factor, E[~bound] / factor, s=1, alpha=0.5, label='Unbound')
        
        axis.set(**AXIS)
        axis.text(0.05, 0.05, f'Bound: {n_bound} / {n_total}', 
                 transform=axis.transAxes, va='bottom', ha='left')
        axis.legend()
        
        if filename:
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()
    
    def get_bound_mask(self, positions, velocities):
        """
        Get mask for bound orbits.
        
        Parameters:
        positions : tuple of arrays (x, y, z)
        velocities : tuple of arrays (vx, vy, vz)
            
        Returns:
        bound : boolean array
            True for bound orbits
        """
        E, Lz = self.compute_energy_angular_momentum(positions, velocities)
        return E < 0


# Convenience function for backward compatibility
def get_mw_model():
    """Get the default Milky Way model instance."""
    return MilkyWayModel()


if __name__ == "__main__":
    # Example usage
    mw = MilkyWayModel()
    print(f"MW model: {mw.name}")
    print(f"Scale radius: {mw.ahalo:.1f} kpc")
    print(f"Disk scale radius: {mw.rH:.1f} kpc")
    print(f"Disk scale height: {mw.zH:.1f} kpc")