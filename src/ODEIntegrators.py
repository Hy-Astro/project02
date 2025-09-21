"""
ODE integrator class to calculate numerical steps in physical simulations
Currently supports interaction by gravity
"""

import numpy as np

class ODEIntegrator:
    """
    Class to handle four ODE integration methods - Euler, RK2, RK4, Leapfrog
    """

    def __init__(self, masses, epsilon=1.0):
        """
        Parameters
        ----------
        masses : (N, ) array
            Object masses in solar masses (M_sun)
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
        Advance one step using Euler's method
        
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
    
    def step_rk2(self, current_positions, current_velocities, dt):
        """
        Advance one step using 2nd-order Runge-Kutta (RK2) method

        Parameters
        ----------
        current_positions  : (N, 3) array
            Current positions (r_n) in AU
        current_velocities : (N, 3) array
            Current velocities (v_n) in AU/yr
        dt                 : float
            Time step (Δt) in years

        Returns
        -------
        new_positions, new_velocities : (N, 3) arrays
            New positions (r_n+1) and new velocities (v_n+1)
        """

        # Algorithmic structure for RK2 method
        k1_r = current_velocities
        k1_v = self.compute_accelerations(current_positions)

        midpoint_positions  = current_positions  + 0.5 * dt * k1_r
        midpoint_velocities = current_velocities + 0.5 * dt * k1_v

        k2_r = midpoint_velocities
        k2_v = self.compute_accelerations(midpoint_positions)

        new_positions  = current_positions  + dt * k2_r
        new_velocities = current_velocities + dt * k2_v

        # Return new positions and new velocities for each object
        return new_positions, new_velocities

    def step_rk4(self, current_positions, current_velocities, dt):
        """
        Advance one step using 4th-order Runge-Kutta (RK4) method

        Parameters
        ----------
        current_positions  : (N, 3) array
            Current positions (r_n) in AU
        current_velocities : (N, 3) array
            Current velocities (v_n) in AU/yr
        dt                 : float
            Time step (Δt) in years

        Returns
        -------
        new_positions, new_velocities : (N, 3) arrays
            New positions (r_n+1) and new velocities (v_n+1)
        """

        # Algorithmic structure for RK4 method
        k1_r = current_velocities
        k1_v = self.compute_accelerations(current_positions)

        pos2 = current_positions  + 0.5 * dt * k1_r
        vel2 = current_velocities + 0.5 * dt * k1_v
        k2_r = vel2
        k2_v = self.compute_accelerations(pos2)

        pos3 = current_positions  + 0.5 * dt * k2_r
        vel3 = current_velocities + 0.5 * dt * k2_v
        k3_r = vel3
        k3_v = self.compute_accelerations(pos3)

        pos4 = current_positions  + dt * k3_r
        vel4 = current_velocities + dt * k3_v
        k4_r = vel4
        k4_v = self.compute_accelerations(pos4)

        new_positions  = current_positions  + \
            (dt/6) * (k1_r + 2*k2_r + 2*k3_r + k4_r)
        new_velocities = current_velocities + \
            (dt/6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

        # Return new positions and new velocities for each object
        return new_positions, new_velocities
    
    def step_leapfrog(self, current_positions, current_velocities, dt):
        """
        Advance one step using the Leapfrog method

        Parameters
        ----------
        current_positions  : (N, 3) array
            Current positions (r_n) in AU
        current_velocities : (N, 3) array
            Current velocities (v_n) in AU/yr
        dt                 : float
            Time step (Δt) in years

        Returns
        -------
        new_positions, new_velocities : (N, 3) arrays
            New positions (r_n+1) and new velocities (v_n+1)
        """

        # Algorithmic structure for Leapfrog method
        current_accelerations = self.compute_accelerations(current_positions)
        halfway_velocities = current_velocities + 0.5 * dt * current_accelerations

        new_positions     = current_positions + dt * halfway_velocities
        new_accelerations = self.compute_accelerations(new_positions)
        new_velocities    = halfway_velocities + 0.5 * dt * new_accelerations
        
        # Return new positions and new velocities for each object
        return new_positions, new_velocities