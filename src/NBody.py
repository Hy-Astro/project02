"""
ODE integrator module to compute and plot gravitational dynamics for N-body system

Table of Contents
-----------------
NBodyIntegrator

EarthSun_test()
EarthSun_plot()

EarthJupiterSun_test()
EarthJupiterSun_plot()

ClusterUnifMass_test()
ClusterUnifMass_plot()

Cluster_loop_test()
Cluster_loop_energy_plot()
Cluster_loop_KroupaPlummer_plot()
Cluster_loop_snapshots()

Cluster_vecs_test()
Cluster_vecs_plot()
"""

import numpy as np
import matplotlib.pyplot as plt
import ODEIntegrators as odeint

class NBodyIntegrator:
    """
    Class to manage simulation procedures for an N-body system
    """

    def __init__(self, masses, positions, velocities,
                 epsilon=1.0, method='leapfrog'):
        """
        Parameters
        ----------
        masses    : (N, ) array
            Object mass in solar masses (M_sun)
        positions : (N, 3) array
            Initial positions in astronomical units (AU)
        velocities: (N, 3) array
            Initial velocities in AU/yr
        epsilon   : float, optional
            Softening parameter in AU
        method    : {'euler', 'rk2', 'rk4', 'leapfrog'}, optional
            ODE integration method
        """

        # Attributes
        self.masses     = np.array(masses, dtype=float)
        self.positions  = np.array(positions, dtype=float)
        self.velocities = np.array(velocities, dtype=float)
        self.G          = 4 * np.pi**2  # AU^3 / (M_sun * yr^2)
        self.N          = len(self.masses)

        self.epsilon = epsilon
        self.method  = method.lower()

        # Create ODEIntegrator toolbox instance
        self.integrator = odeint.ODEIntegrator(self.masses, self.epsilon)

    def compute_accelerations(self, positions):
        return self.integrator.compute_accelerations(positions)

    def step(self, dt):
        """
        Advance one step using the specified method
        """
        
        match self.method:
            # Euler's method (1st order)
            case 'euler':
                self.positions, self.velocities = self.integrator.step_euler(
                    self.positions, self.velocities, dt)

            # RK2 method (2nd order)
            case 'rk2':
                self.positions, self.velocities = self.integrator.step_rk2(
                    self.positions, self.velocities, dt)

            # RK4 method (4th order)
            case 'rk4':
                self.positions, self.velocities = self.integrator.step_rk4(
                    self.positions, self.velocities, dt)
                
            # Leapfrog method (2nd order, symplectic)
            case 'leapfrog':
                self.positions, self.velocities = self.integrator.step_leapfrog(
                    self.positions, self.velocities, dt)

            # Invalidate all other cases
            case _:
                raise ValueError(f"Unknown integration method: {self.method}. "
                                  "Options: 'euler', 'rk2', 'rk4', 'leapfrog'")

    def compute_energy(self):
        """
        Compute kinetic, potential, total energy of the system
        """

        # Kinetic energy
        KE = 0.5 * np.sum(self.masses[:, None] * self.velocities**2)

        # Potential energy
        PE = 0.0
        G = self.G
        for i in range(self.N):
            for j in range(i+1, self.N):
                r_vec = self.positions[j] - self.positions[i]
                r     = np.sqrt(np.dot(r_vec, r_vec) + self.epsilon**2)

                PE   -= G * self.masses[i] * self.masses[j] / r

        # Total energy
        TE = KE + PE

        # Return energy values in M_sun * AU^2 / yr^2
        return KE, PE, TE

    def run(self, dt, n_steps):
        """
        Run the simulation for n_steps timesteps with step size dt
        """

        # Compute and record energy at each step
        energies = []
        for _ in range(n_steps):
            self.step(dt)
            energies.append(self.compute_energy())

        # Return energy values in M_sun * AU^2 / yr^2
        return np.array(energies)
    
def EarthSun_test(years=10, dt=0.01, method='leapfrog'):
    """
    Simulate Earth-Sun system with chosen integrator
    
    Parameters
    ----------
    years  : float, optional
        Simulation duration in years (default: 10)
    dt     : float, optional
        Time step (Δt) in years (default: 0.01)
    method : {'euler', 'rk2', 'rk4', 'leapfrog'}, optional
        ODE integration method (default: 'leapfrog')

    Returns
    -------
    earth_positions : (n_steps, 3) array
        Position of Earth at each step in AU
    energies        : (n_steps, 3) array
        Energy at each step in M_sun * AU^2 / yr^2
        Includes kinetic, potential, total energy

    Notes
    -----
    n_steps = int(years/dt)
    """
    
    # Initial parameters
    masses = np.array(
        # Sun, Earth masses in M_sun
        [1.0, 1/333000]
    )
    positions = np.array([
        # [x, y, z] in AU, Sun at origin
        [0.0, 0.0, 0.0],  # Sun
        [1.0, 0.0, 0.0]   # Earth
    ])  # 1 AU on the x-axis
    v_0 = np.sqrt(
        (4 * np.pi**2) * masses[0] / np.linalg.norm(positions[1])
    )   # √ GM/r
    velocities = np.array([
        # [vx, vy, vz] in AU/yr, Sun at rest
        [0.0, 0.0, 0.0], # Sun
        [0.0, v_0, 0.0]  # Earth, one revolution per year
    ])  # 1 AU/yr counterclockwise, parallel to y-axis

    # Create and run simulation
    sim = NBodyIntegrator(masses, positions, velocities,
                          epsilon=1e-3, method=method)
    
    n_steps = int(years/dt)
    earth_positions, energies = [], []
    for _ in range(n_steps):
        sim.step(dt)
        KE, PE, TE = sim.compute_energy()
        earth_positions.append(sim.positions[1].copy())
        energies.append([KE, PE, TE])

    # Return Earth positions and energies calculated at each step
    return np.array(earth_positions), np.array(energies)

def EarthSun_plot(years=10, dt=0.01, savefig=False):
    """
    Figure of 4 plots describing simulation results of Sun-Earth system
    All four integration methods used
    
    Parameters
    ----------
    years   : float, optional
        Simulation duration in years (default: 10)
    dt      : float, optional
        Time step (Δt) in years (default: 10)
    savefig : bool, optional
        Save figure as PNG (default: False)
    """

    methods    = ['euler', 'rk2', 'rk4', 'leapfrog']
    colors     = {'euler':'b', 'rk2':'c', 'rk4':'m', 'leapfrog':'g'}
    linestyles = {'euler':'-', 'rk2':'--', 'rk4':'-.', 'leapfrog':':'}

    # Acquire simulation results once for all four methods
    results = {}
    for method in methods:
        pos, energies = EarthSun_test(years=years, dt=dt, method=method)
        results[method] = {'positions': pos, 'energies': energies}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    interval = results['euler']['energies'].shape[0] # years/Δt, rounded
    times = np.linspace(0, years, interval)

    # Top left: simulated orbits
    ax = axes[0, 0]
    for method in methods:
        pos = results[method]['positions']
        ax.plot(pos[:,0], pos[:,1], label=method,
                color=colors[method], linestyle=linestyles[method])

    ax.set_title(f"Earth's Orbit over {years} Year{'s' if years != 1 else ''}")
    ax.set_xlabel(r"$x\ \rm [AU]$"); ax.set_ylabel(r"$y\ \rm [AU]$")
    ax.set_aspect('equal', 'box')
    ax.legend(loc=1)

    # Top right: total energies overtime
    ax = axes[0, 1]
    for method in methods:
        Etot = results[method]['energies'][:, 2]
        ax.plot(times, Etot, label=method, color=colors[method])

    ax.set_title("Total Energies vs. Time")
    ax.set_xlabel(r"$t\ \rm [yr]$")
    ax.set_ylabel(r"$E_{\rm tot}\ \rm [M_\odot\,AU^2/yr^2]$")
    ax.legend(loc=1)

    # Bottom left: relative energy errors overtime
    ax = axes[1, 0]
    for method in methods:
        energies = results[method]['energies']
        Etot = energies[:, 2]
        Etrue = Etot[0] # energy of system at t=0
        rel_error = np.abs(Etot - Etrue) / abs(Etrue)
        ax.plot(times, rel_error, label=method, color=colors[method])

    ax.set_yscale('log')
    ax.set_title("Relative Energy Errors vs. Time")
    ax.set_xlabel(r"$t\ \rm [yr]$"); ax.set_ylabel(r"$\Delta E\,/\,|E_0|$")
    ax.legend(loc=1)

    # Bottom right: virial ratios overtime
    ax = axes[1, 1]
    for method in methods:
        energies = results[method]['energies']
        KE, PE = energies[:, 0], energies[:, 1]
        Q = np.abs(2*KE + PE) / np.abs(PE)
        ax.plot(times, Q, label=method, color=colors[method])

    ax.set_title("Virial Ratios vs. Time")
    ax.set_xlabel(r"$t\ \rm [yr]$"); ax.set_ylabel("$Q$")
    ax.legend(loc=1)

    fig.suptitle(f"Results for the Earth-Sun System over a {years}-Year Period",
                 fontsize=16, weight='bold')
    if savefig: plt.savefig('EarthSun_plot.png')
    plt.show()

def EarthJupiterSun_test(years=24, dt=0.01, method='leapfrog'):
    """
    Simulate Earth-Jupiter-Sun system with chosen integrator

    Parameters
    ----------
    years  : float, optional
        Simulation duration in years (default: 24)
    dt     : float, optional
        Time step (Δt) in years (default: 0.01)
    method : {'euler', 'rk2', 'rk4', 'leapfrog'}, optional
        ODE integration method (default: 'leapfrog')

    Returns
    -------
    all_positions : (n_steps, 3, 3) array
        Positions of Earth, Jupiter, Sun at each step
    energies      : (n_steps, 3) array
        Energy at each step in M_sun * AU^2 / yr^2
        Includes kinetic, potential, total energy

    Notes
    -----
    n_steps = int(years/dt)
    """

    # Initial parameters
    masses = np.array(
        # Sun, Earth, Jupiter masses in M_sun
        [1.0, 1/333000, 9.55e-4]
    )
    positions = np.array([
        # [x, y, z] in AU, Sun at origin
        [0.0, 0.0, 0.0], # Sun
        [1.0, 0.0, 0.0], # Earth
        [5.2, 0.0, 0.0]  # Jupiter
    ])  # All three bodies collinear
    v_E = np.sqrt(
        (4 * np.pi**2) * masses[0] / np.linalg.norm(positions[1])
    )   # √ GM/r
    v_J = np.sqrt(
        (4 * np.pi**2) * masses[0] / np.linalg.norm(positions[2])
    )   # √ GM/r
    velocities = np.array([
        # [vx, vy, vz] in AU/yr, Sun at rest
        [0.0, 0.0, 0.0],  # Sun
        [0.0, v_E, 0.0],  # Earth, one revolution per year
        [0.0, v_J, 0.0]   # Jupiter, one revolution per 12 years
    ])  # counterclockwise, parallel to y-axis

    # Create and run simulation
    sim = NBodyIntegrator(masses, positions, velocities,
                          epsilon=1e-3, method=method)

    n_steps = int(years/dt)
    all_positions, energies = [], []
    for _ in range(n_steps):
        sim.step(dt)
        KE, PE, TE = sim.compute_energy()
        all_positions.append(sim.positions.copy())
        energies.append([KE, PE, TE])

    # Return positions and energies of all three bodies calculated at each step
    return np.array(all_positions), np.array(energies)

def EarthJupiterSun_plot(years=24, dt=0.01, savefig=False):
    """
    Figure of 4 plots describing simulation results of Earth-Jupiter-Sun system
    Only RK4 and Leapfrog integration methods used

    
    Parameters
    ----------
    years   : float, optional
        Simulation duration in years (default: 24)
    dt      : float, optional
        Time step (Δt) in years (default: 0.01)
    savefig : bool, optional
        Save figure as PNG (default: False)
    """

    methods    = ['rk4', 'leapfrog']
    colors     = {'rk4':'m', 'leapfrog':'g'}
    linestyles = {'rk4':'-.', 'leapfrog':':'}

    # Acquire simulation results once for both methods
    results = {}
    for method in methods:
        pos, energies = EarthJupiterSun_test(years=years, dt=dt, method=method)
        results[method] = {'positions': pos, 'energies': energies}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    interval = results['rk4']['energies'].shape[0] # years/Δt, rounded
    times = np.linspace(0, years, interval)

    # Top left: simulated orbits
    ax = axes[0, 0]
    for method in methods:
        pos = results[method]['positions']
        ax.plot(pos[:, 0, 0], pos[:, 0, 1], label=f"Sun ({method})",
                color=colors[method], linestyle=linestyles[method])
        ax.plot(pos[:, 1, 0], pos[:, 1, 1], label=f"Earth ({method})",
                color=colors[method], linestyle=linestyles[method], alpha=0.7)
        ax.plot(pos[:, 2, 0], pos[:, 2, 1], label=f"Jupiter ({method})",
                color=colors[method], linestyle=linestyles[method], alpha=0.7)

    ax.set_title(f"3-Body Orbits over {years} Year{'s' if years != 1 else ''}")
    ax.set_xlabel(r"$x\ \rm [AU]$"); ax.set_ylabel(r"$y\ \rm [AU]$")
    ax.set_aspect('equal', 'box')
    ax.legend(loc=1)

    # Top right: energy components overtime
    ax = axes[0, 1]
    for method in methods:
        energies = results[method]['energies']
        KE, PE, TE = energies[:, 0], energies[:, 1], energies[:, 2]
        ax.plot(times, KE, label=rf"$E_{{\rm K}}$ ({method})",
                color=colors[method], linestyle="--", alpha=0.6)
        ax.plot(times, PE, label=f"$W$ ({method})",
                color=colors[method], linestyle=":", alpha=0.6)
        ax.plot(times, TE, label=rf"$E_{{\rm tot}}$ ({method})",
                color=colors[method], linestyle="-", alpha=1.0, linewidth=1.8)

    ax.set_title("Energy Components vs. Time")
    ax.set_xlabel(r"$t\ \rm [yr]$")
    ax.set_ylabel(r"$E\ \rm [M_\odot AU^2/yr^2]$")
    ax.legend(loc=1)

    # Bottom left: relative energy errors overtime
    ax = axes[1, 0]
    for method in methods:
        energies = results[method]['energies']
        Etot = energies[:, 2]
        Etrue = Etot[0] # energy of system at t=0
        rel_error = np.abs(Etot - Etrue) / abs(Etrue)
        ax.plot(times, rel_error, label=method,
                color=colors[method], linestyle=linestyles[method])

    ax.set_yscale('log')
    ax.set_title("Relative Energy Errors vs. Time")
    ax.set_xlabel(r"$t\ \rm [yr]$"); ax.set_ylabel(r"$\Delta E\,/\,|E_0|$")
    ax.legend(loc=1)

    # Bottom right: virial ratios overtime
    ax = axes[1, 1]
    for method in methods:
        energies = results[method]['energies']
        KE, PE = energies[:, 0], energies[:, 1]
        Q = np.abs(2*KE + PE) / np.abs(PE)
        ax.plot(times, Q, label=method,
                color=colors[method], linestyle=linestyles[method])

    ax.set_title("Virial Ratios vs. Time")
    ax.set_xlabel(r"$t\ \rm [yr]$"); ax.set_ylabel("$Q$")
    ax.legend(loc=1)

    fig.suptitle(
        f"Results for the Earth-Jupiter-Sun System over a {years}-Year Period",
        fontsize=16, weight='bold'
    )
    if savefig: plt.savefig('EarthJupiterSun_plot.png')
    plt.show()

def ClusterUnifMass_test(N=10, R=100, years=100, dt=0.01, method='leapfrog'):
    """
    Simulate simple N-body stellar cluster with chosen integrator
    N stars of 1 M_sun, random positions in sphere of radius R, at rest

    Parameters
    ----------
    N      : int, optional
        Number of stars (default: 10)
    R      : float, optional
        Cluster radius in AU (default: 100)
    years  : float, optional
        Simulation duration in years (default: 100)
    dt     : float, optional
        Time step (Δt) in years (default: 0.01)
    method : {'euler', 'rk2', 'rk4', 'leapfrog'}, optional
        ODE integration method (default: 'leapfrog')

    Returns
    -------
    all_positions : (n_steps, N, 3) array
        Position of every star at each step
    energies      : (n_steps, 3) array
        Energy at each step in M_sun * AU^2 / yr^2
        Includes kinetic, potential, total energy

    Notes
    -----
    n_steps = int(years/dt)
    """

    pos = []
    for _ in range(N):
        # For each star, randomly generate positions until condition met
        while True:
            x, y, z = np.random.uniform(-R, R, 3)
            # Condition: star must be within sphere
            if x**2 + y**2 + z**2 <= R**2:
                pos.append([x, y, z])
                break

    # Initial parameters
    masses = np.ones(N)           # every star 1 M_sun
    positions = np.array(pos)     # every star within cluster sphere
    velocities = np.zeros((N, 3)) # every star at rest
    epsilon = 0.01 * R / N**(1/3) # softening parameter (≈ 0.464 AU default)

    # Create and run simulation
    sim = NBodyIntegrator(masses, positions, velocities,
                          epsilon=epsilon, method=method)
    n_steps = int(years/dt)
    all_positions, energies = [], []

    for _ in range(n_steps):
        sim.step(dt)
        KE, PE, TE = sim.compute_energy()
        all_positions.append(sim.positions.copy())
        energies.append([KE, PE, TE])

    # Return position and energies of every star calculated at each step
    return np.array(all_positions), np.array(energies)

def ClusterUnifMass_plot(N=10, R=100.0, years=100, dt=0.01, savefig=False):
    """
    Figure of 3 plots describing simulation results of simple N-body cluster
    For more information, see parent function `ClusterUnifMass_test()`
    Only RK4 and Leapfrog integration methods used

    Parameters
    ----------
    N       : int, optional
        Number of stars (default: 10)
    R       : float, optional
        Cluster radius in AU (default: 100)
    years   : float, optional
        Simulation duration in years (default: 100)
    dt      : float, optional
        Time step (Δt) in years (default: 0.01)
    savefig : bool, optional
        Save figure as PNG (default: False)
    """

    methods    = ['rk4', 'leapfrog']
    colors     = {'rk4':'m', 'leapfrog':'g'}
    linestyles = {'rk4':'-.', 'leapfrog':':'}

    # Acquire simulation results once for both methods
    results = {}
    for method in methods:
        _, energies = ClusterUnifMass_test(N=N, R=R,
                                           years=years, dt=dt, method=method)
        results[method] = {'energies': energies}

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)
    interval = results['rk4']['energies'].shape[0] # years/Δt, rounded
    times = np.linspace(0, years, interval)

    # Top: energy components overtime
    ax = axes[0]
    for method in methods:
        energies = results[method]['energies']
        KE, PE, TE = energies[:, 0], energies[:, 1], energies[:, 2]
        ax.plot(times, KE, label=rf"$E_{{\rm K}}$ ({method})",
                color=colors[method], linestyle="--", alpha=0.6)
        ax.plot(times, PE, label=f"$W$ ({method})",
                color=colors[method], linestyle=":", alpha=0.6)
        ax.plot(times, TE, label=rf"$E_{{\rm tot}}$ ({method})",
                color=colors[method], linestyle="-", alpha=1.0, linewidth=1.8)

    ax.set_title("Energy Components vs. Time")
    ax.set_xlabel(r"$t\ \rm [yr]$")
    ax.set_ylabel(r"$E\ \rm [M_\odot\,AU^2/yr^2]$")
    ax.legend(loc=1)

    # Center: relative energy errors overtime
    ax = axes[1]
    for method in methods:
        energies = results[method]['energies']
        Etot = energies[:, 2]
        Etrue = Etot[0] # energy of system at t=0
        rel_error = np.abs(Etot - Etrue) / abs(Etrue)
        ax.plot(times, rel_error, label=method,
                color=colors[method], linestyle=linestyles[method])
    ax.set_yscale('log')
    ax.set_title("Relative Energy Errors vs. Time")
    ax.set_xlabel(r"$t\ \rm [yr]$"); ax.set_ylabel(r"$\Delta E\,/\,|E_0|$")
    ax.legend(loc=1)

    # Bottom: virial ratios overtime
    ax = axes[2]
    for method in methods:
        energies = results[method]['energies']
        KE, PE = energies[:, 0], energies[:, 1]
        Q = np.abs(2*KE + PE) / np.abs(PE)
        ax.plot(times, Q, label=method,
                color=colors[method], linestyle=linestyles[method])
    ax.set_title("Virial Ratios vs. Time")
    ax.set_xlabel(r"$t\ \rm [yr]$"); ax.set_ylabel("$Q$")
    ax.legend(loc=1)

    fig.suptitle(
        f"Energy Results for the Simple Stellar Cluster ({N}-Body)\n"
        f"Over a {years}-Year Period",
        fontsize=15, weight='bold'
    )
    if savefig: plt.savefig("Cluster_loop_plot.png")
    plt.show()

def Cluster_loop_test(N=100, a=1000, years=1000, dt=0.01, method='leapfrog'):
    """
    Simulate N-body stellar cluster with chosen integrator and improvements:
    1. Masses from Kroupa Initial Mass Function (IMF)
    2. Positions from Plummer sphere
    3. Velocities in virial equilibrium

    Parameters
    ----------
    N      : int, optional
        Number of stars (default: 100)
    a      : float, optional
        Plummer radius in AU (default: 1000)
    years  : float, optional
        Simulation duration in years (default: 100)
    dt     : float, optional
        Time step (Δt) in years (default: 0.01)
    method : {'euler', 'rk2', 'rk4', 'leapfrog'}, optional
        ODE integration method (default: 'leapfrog', highly recommended)

    Returns
    -------
    all_positions : (n_steps, N, 3) array
        Position of every star at each step
    energies      : (n_steps, 3) array
        Energy at each step in M_sun * AU^2 / yr^2
        Includes kinetic, potential, total energy

    Notes
    -----
    n_steps = int(years/dt)
    """

    ### 1. Sample masses from Kroupa IMF ###
    def sample_powerlaw(alpha, a, b, n_stars):
        """
        Sample from m^-α between [a, b] using inverse CDF
        m = [a^(1-α) + U(0, 1) * (b^(1-α) - a^(1-α))]^(1 / 1-α)
        """

        u = np.random.uniform(1e-12, 1 - 1e-12, size=n_stars)
        return \
            (a**(1-alpha) + u * (b**(1-alpha) - a**(1-alpha)))**(1/(1-alpha))

    # Compute normalization constants
    m_min, m_break, m_max = 0.08, 0.5, 150.0 
    alpha1, alpha2 = 1.3, 2.3

    I1 = (m_break**(1-alpha1) - m_min**(1-alpha1)) / (1-alpha1)
    I2 = (m_max**(1-alpha2) - m_break**(1-alpha2)) / (1-alpha2)
    A1 = N / (I1 + m_break**(alpha2-alpha1) * I2)
    A2 = A1 * m_break**(alpha2 - alpha1) # not needed, as P2 = 1 - P1
    P1 = (A1 * I1) / N  # probability of being in low-mass segment

    # Draw masses
    masses = []
    for _ in range(N):
        if np.random.rand() < P1:
            masses.append(sample_powerlaw(alpha1, m_min, m_break, 1)[0])
        else:
            masses.append(sample_powerlaw(alpha2, m_break, m_max, 1)[0])
    masses = np.array(masses)

    ### 2. Sample positions from Plummer sphere ###
    u = np.random.uniform(1e-12, 1 - 1e-12, N) # bounds to prevent bad values
    r = a * (u**(-2/3) - 1)**(-1/2)            # radial distances

    # Convert to spherical coordinates
    phi = 2 * np.pi * np.random.uniform(0, 1, N)
    cos_theta = 1 - 2 * np.random.uniform(0, 1, N)
    sin_theta = np.sqrt(1 - cos_theta**2) # following sin^2(θ) + cos^2(θ) = 1

    x = r * sin_theta * np.cos(phi)
    y = r * sin_theta * np.sin(phi)
    z = r * cos_theta
    positions = np.vstack((x, y, z)).T

    # Adjust center of mass (CoM) positions with weighted average
    r_CoM = np.average(positions, axis=0, weights=masses)
    positions -= r_CoM

    ### 3. Sample velocities from virial equilibrium ###
    sigmas = np.sqrt(
        ((4 * np.pi**2) * np.sum(masses) / (6*a)) * (1 + (r/a)**2)**(-1/2)
    ) # σ^2(r) = (GM/6a) / √(1 + r^2 / a^2), with σ in AU/yr and 1D

    vx = np.random.normal(0, sigmas)
    vy = np.random.normal(0, sigmas)
    vz = np.random.normal(0, sigmas)
    velocities = np.vstack((vx, vy, vz)).T

    # Adjust CoM velocities with weighted average
    v_cm = np.average(velocities, axis=0, weights=masses)
    velocities -= v_cm

    ### Create and run simulation ###
    epsilon = 0.01 * a / N**(1/3) # softening parameter (≈ 2.154 AU default)
    sim = NBodyIntegrator(masses, positions, velocities,
                          epsilon=epsilon, method=method)
    n_steps = int(years / dt)
    all_positions, energies = [], []

    for _ in range(n_steps):
        sim.step(dt)
        KE, PE, TE = sim.compute_energy()
        all_positions.append(sim.positions.copy())
        energies.append([KE, PE, TE])

    # Return position and energies of every star calculated at each step
    return np.array(all_positions), np.array(energies)

def Cluster_loop_energies_plot(N=100, a=1000, years=1000, dt=0.01, savefig=False):
    """
    Figure of 3 plots describing simulation results of improved N-body cluster
    For more information, see parent function `Cluster_loop_test()`
    Only Leapfrog integration method used due to strong energy conservation

    Parameters
    ----------
    N       : int, optional
        Number of stars (default: 100)
    a       : float, optional
        Plummer radius in AU (default: 1000)
    years   : float, optional
        Simulation duration in years (default: 100)
    dt      : float, optional
        Time step (Δt) in years (default: 0.01)
    savefig : bool, optional
        Save figure as PNG (default: False)
    """

    # Acquire simulation results once
    _, energies = Cluster_loop_test(N=N, a=a,
                                    years=years, dt=dt, method="leapfrog")
    KE, PE, TE = energies[:, 0], energies[:, 1], energies[:, 2]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)
    interval = energies.shape[0] # years/Δt, rounded
    times = np.linspace(0, years, interval)

    # Top: energy components overtime
    ax = axes[0]
    ax.plot(times, KE, label=r"$E_{\rm K}$", linestyle="--", alpha=0.6)
    ax.plot(times, PE, label="$W$", linestyle=":", alpha=0.6)
    ax.plot(times, TE, label=r"$E_{\rm tot}$", linewidth=1.8, alpha=1.0)
    ax.set_title("Energy Components vs. Time")
    ax.set_xlabel(r"$t\ \rm [yr]$")
    ax.set_ylabel(r"$E\ \rm [M_\odot\,AU^2/yr^2]$")
    ax.legend(loc=1)

    # Center: relative energy errors overtime
    ax = axes[1]
    Etrue = TE[0] # energy of system at t=0
    rel_err = np.abs(TE - Etrue) / abs(Etrue)
    ax.plot(times, rel_err)
    ax.set_yscale('log')
    ax.set_title("Relative Energy Errors vs. Time")
    ax.set_xlabel(r"$t\ \rm [yr]$"); ax.set_ylabel(r"$\Delta E\,/\,|E_0|$")

    # Bottom: virial ratios overtime
    ax = axes[2]
    Q = np.abs(2*KE + PE) / np.abs(PE)
    ax.plot(times, Q)
    ax.set_title("Virial Ratios vs. Time")
    ax.set_xlabel(r"$t\ \rm [yr]$"); ax.set_ylabel("$Q$")

    fig.suptitle(
        f"Energy Results for the Improved Stellar Cluster ({N}-Body)\n"
        f"Over a {years}-Year Period",
        fontsize=15, weight='bold'
    )
    if savefig:plt.savefig("Cluster_loop_energies.png")
    plt.show()

def Cluster_loop_KroupaPlummer_plot(N=100, a=1000, savefig=False):
    """
    Figure of 2 plots validating initial conditions of improved N-body cluster:
    - Sampled mass distribution vs. known Kroupa IMF
    - Sampled radial density profile vs. known Plummer model
    For more information, see parent function `Cluster_loop_test()`

    Parameters
    ----------
    N       : int, optional
        Number of stars (default: 100)
    a       : float, optional
        Plummer radius in AU (default: 1000)
    savefig : bool, optional
        Save figure as PNG (default: False)
    """

    ### Re-sample initial cluster once: copied code from `Cluster_loop_test()` ###
    def sample_powerlaw(alpha, a, b, n_stars):
        u = np.random.uniform(1e-12, 1 - 1e-12, size=n_stars)
        return (a**(1-alpha) + u * (b**(1-alpha) - a**(1-alpha)))**(1/(1-alpha))
    m_min, m_break, m_max = 0.08, 0.5, 150.0
    alpha1, alpha2 = 1.3, 2.3
    I1 = (m_break**(1-alpha1) - m_min**(1-alpha1)) / (1-alpha1)
    I2 = (m_max**(1-alpha2) - m_break**(1-alpha2)) / (1-alpha2)
    A1 = N / (I1 + m_break**(alpha2-alpha1) * I2)
    A2 = A1 * m_break**(alpha2 - alpha1)
    P1 = (A1 * I1) / N
    masses = []
    for _ in range(N):
        if np.random.rand() < P1:
            masses.append(sample_powerlaw(alpha1, m_min, m_break, 1)[0])
        else:
            masses.append(sample_powerlaw(alpha2, m_break, m_max, 1)[0])
    masses = np.array(masses)
    u = np.random.uniform(1e-12, 1 - 1e-12, N)
    r = a * (u**(-2/3) - 1)**(-0.5)
    phi = 2 * np.pi * np.random.uniform(0, 1, N)
    cos_theta = 1 - 2 * np.random.uniform(0, 1, N)
    sin_theta = np.sqrt(1 - cos_theta**2)
    x = r * sin_theta * np.cos(phi)
    y = r * sin_theta * np.sin(phi)
    z = r * cos_theta
    positions = np.vstack((x, y, z)).T
    r_cm = np.average(positions, axis=0, weights=masses)
    positions -= r_cm
    radii = np.linalg.norm(positions, axis=1)

    ## Create figure ###
    fig, axes = plt.subplots(2, 1, figsize=(8, 10), constrained_layout=True)

    # Top: mass histogram and comparison
    ax = axes[0]
    counts, bins, _ = ax.hist(
        masses, bins=np.logspace(np.log10(m_min), np.log10(m_max), 30),
        histtype='bar', density=True, label="Stellar masses", color='c', alpha=0.6
    )

    m_line = np.logspace(np.log10(m_min), np.log10(m_max), 200) # Kroupa slopes
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    ref_density = np.interp(m_break, bin_centers, counts)
    scale1 = ref_density * m_break**alpha1
    scale2 = scale1 * m_break**(alpha2 - alpha1)
    ax.plot(m_line[m_line < m_break],
            scale1 * m_line[m_line < m_break]**(-alpha1),
            'r--', lw=2, label=rf"$\alpha_1={alpha1}$")
    ax.plot(m_line[m_line >= m_break],
            scale2 * m_line[m_line >= m_break]**(-alpha2),
            'g--', lw=2, label=rf"$\alpha_2={alpha2}$")

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_title("Sampled Stellar Masses (Kroupa IMF)")
    ax.set_xlabel(r"$M\ \rm [M_\odot]$"); ax.set_ylabel("Count")
    ax.legend(loc=1)

    # Bottom: radial density profile and comparison
    ax = axes[1]
    counts, bins = np.histogram(radii, bins=30)
    shell_volumes = (4/3) * np.pi * (bins[1:]**3 - bins[:-1]**3)
    density = counts / shell_volumes
    bin_centers = (bins[1:] + bins[:-1]) / 2
    ax.loglog(bin_centers, density, 'o', label="Radial densities")

    radii_model = np.logspace(np.log10(radii.min()), np.log10(radii.max()), 200)
    Plummer_model = (1 + (radii_model/a)**2)**(-2.5) # Plummer profile
    Plummer_model *= density.max() / Plummer_model.max() # normalization
    ax.loglog(radii_model, Plummer_model, label="Plummer model")

    ax.set_xlabel(r"$r\ \rm [AU]$")
    ax.set_ylabel(r"$\rho\,\rm *\,constant\ [M_\odot/AU^3]$")
    ax.set_title("Sampled Radial Density Profile (Plummer Model)")
    ax.legend(loc=1)

    fig.suptitle(
          "Initial Conditions Validation for the Improved Stellar Cluster\n"
        rf"With $N={N}\ \rm stars $ and $a={a}\ \rm AU$",
        fontsize=15, weight='bold'
    )
    if savefig: plt.savefig("Cluster_loop_KroupaPlummer_plot.png")
    plt.show()

def Cluster_loop_snapshots(N=100, a=1000, years=1000, dt=0.01, savefig=False,
                           milestones=[0.00, 0.25, 0.50, 0.75, 1.00]):
    """
    Figure of position snapshots during simulation of improved N-body cluster
    For more information, see parent function `Cluster_loop_test()`

    Parameters
    ----------
    N          : int, optional
        Number of stars (default: 100)
    a          : float, optional
        Plummer radius in AU (default: 1000)
    years      : float, optional
        Simulation duration in years (default: 100)
    dt         : float, optional
        Time step (Δt) in years (default: 0.01)
    savefig    : bool, optional
        Save figure as PNG (default: False)
    milestones : array of floats, optional
        Fractions of total simulation duration
        (default: [0.00, 0.25, 0.50, 0.75, 1.00])
    """

    # Input validation for milestones
    milestones = np.sort(np.asarray(milestones))
    if np.any(milestones < 0) or np.any(milestones > 1):
        raise ValueError("Fractions must be between 0 and 1")

    # Acquire simulation results once
    positions, _ = Cluster_loop_test(N=N, a=a,
                                     years=years, dt=dt, method='leapfrog')
    intervals = positions.shape[0] # years/Δt, rounded
    times = np.linspace(0, years, intervals)

    # Configure figure to have 2 columns maximum
    idxs = [int(m * (intervals - 1)) for m in milestones]
    n_snapshots = len(idxs)
    ncols = 2 if n_snapshots > 1 else 1
    nrows = int(np.ceil(n_snapshots / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 6*nrows),
                             tight_layout=True)
    axes = np.atleast_1d(axes).flatten()

    for k, (ax, idx, frac) in enumerate(zip(axes, idxs, milestones)):
        pos = positions[idx]
        ax.scatter(pos[:, 0], pos[:, 1], s=10, alpha=0.7)
        ax.set_aspect('equal', 'box')
        ax.set_title(rf"$t={times[idx]:.1f}\ \rm yr\ ({frac*100:.1f}\rm \%)$")
        ax.set_xlabel(r"$x\ \rm [AU]$"); ax.set_ylabel(r"$y\ \rm [AU]$")

    for j in range(k+1, len(axes)):
        # Hide any unused subplots
        axes[j].set_visible(False)

    fig.suptitle(
        f"Position Snapshots for the Improved Stellar Cluster ({N}-Body)\n"
        f"Over a {years}-Year Period",
        fontsize=15, weight='bold'
    )
    if savefig: plt.savefig("Cluster_loop_snapshots.png")
    plt.show()