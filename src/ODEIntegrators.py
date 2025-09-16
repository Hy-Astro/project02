"""ODE integrator class to compute N-body gravitational dynamics"""

import numpy as np

class ODEIntegrator:
    """
    Class to handle four ODE integration methods - Euler, RK2, RK4, Leapfrog
    """

    def __init__(self, masses, epsilon=1.0):
        """
        Parameters
        ----------
        masses : array-like
            Object mass in solar masses (M_sun)
        epsilon : float
            Softening parameter in astronomical units (AU)
        """

        # Attributes
        self.masses = np.asarray(masses, dtype=float)
        self.N = len(masses)

        self.epsilon = epsilon # AU
        self.G = 4 * np.pi**2  # AU^3 / (M_sun * yr^2)

    def compute_accelerations(self, positions):
        """
        Compute accelerations on each object due to gravity
        
        Parameters
        ----------
        positions : (N, 3) array
            Particle positions in AU

        Returns
        -------
        accelerations : (N, 3) array
            Particle accelerations in AU/yr^2
        """

        N = self.N
        accelerations = np.zeros_like(positions)

        # Formula for $\vec a_i$, computed for all i, j, i =/= j
        for i in range(N):
            for j in range(N):
                if i == j: continue # i =/= j

                r_vec = positions[j] - positions[i]
                r2_ep = np.dot(r_vec, r_vec) + self.epsilon**2

                accelerations[i] += \
                    self.G * self.masses[j] * r_vec / r2_ep**1.5

        # Return acceleration value for each object
        return accelerations

    def step_euler(self, current_positions, current_velocities, dt):
        """
        Advance one step from Euler's method
        
        Parameters
        ----------
        current_positions  : (N, 3) array
            Current positions  (r_n) in AU
        current_velocities : (N, 3) array
            Current velocities (v_n) in AU/yr
        dt                 : float
            Time step (Δt) in years
        
        Returns
        -------
        new_positions, new_velocities : (N, 3) arrays
            New positions (r_n+1) and new velocities (v_n+1)
        """
        
        # Algorithmic structure for Euler's method
        accelerations  = self.compute_accelerations(current_positions)
        new_positions  = current_positions  + dt * current_velocities
        new_velocities = current_velocities + dt * accelerations

        # Return new positions and new velocities for each object
        return new_positions, new_velocities