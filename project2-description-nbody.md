# Project 2: N-body Dynamics with Monte Carlo Sampling

**ASTR 596 Fall 2025**  
**Instructor:** Anna Rosen  
**Due:** Wednesday September 24, 2025 by 11:59 pm

## Pair Programming Assignments for Project 2

**Project 2 Pairings:**

- **Pair 1:** Katarzyna + Paige
- **Pair 2:** Hy + Aisling  
- **Triple:** Caleb + Kaitlyn + Nodoka

**Individual Implementation Requirements:**
While you work together during Friday sessions:

- Core algorithms should be independently written
- Shared: Overall structure, test cases, debugging strategies
- Independent: Actual implementation of integrators, force calculation
- OK to share: Plotting code, file I/O, validation tests

Review the [Pair Programming Guidelines](https://astrobytes-edu.github.io/astr596-modeling-universe/pair-programming-guidelines/) for best practices.

## N-body Dynamics with Monte Carlo Initial Conditions

Write a Python software package to simulate the dynamics of a star cluster using N-body algorithms. You will implement and compare Euler, RK2, RK4, and Leapfrog ODE integration schemes.

Your code must be modular and object-oriented, allowing users to integrate the orbits of $N$ bodies under their mutual gravitational interactions, assuming the system is isolated. Your code should allow users to select any integration method (e.g., via a class `ODEIntegrator`).

**Relevant Course Materials:**

- [ODE Integration Methods: Euler's Method](https://astrobytes-edu.github.io/astr596-modeling-universe/module3-part1-euler/)
- [Runge-Kutta Methods](https://astrobytes-edu.github.io/astr596-modeling-universe/module3-part2-runge-kutta/)
- [Symplectic Integrators (Leapfrog)](https://astrobytes-edu.github.io/astr596-modeling-universe/module3-part3-symplectic/)

## Numerical Problem Description

The acceleration of particle $i$ in an $N$-body gravitational system, including a softening parameter $\epsilon$, is given by:

$$
\vec{a}_i = \sum_{\substack{j=1 \\ j \ne i}}^{N} G\,m_j \frac{\vec{r}_j - \vec{r}_i}{\left(\lvert\vec{r}_j - \vec{r}_i\rvert^2 + \epsilon^2\right)^{3/2}}
$$

where $m_i$ and $m_j$ are the particle masses, and the distance $r_{ij}$ between particles $i$ and $j$ is defined as:

$$
r_{ij}^2 = (x_j - x_i)^2 + (y_j - y_i)^2 + (z_j - z_i)^2
$$

The softening parameter $\epsilon$ prevents numerical divergences when two particles approach closely, effectively replacing the point-mass potential at small scales with a smoothed potential core. Use

$$\epsilon \approx 0.01 \times R_{\text{cluster}}/N^{1/3},$$

where $R_{\text{cluster}}/N^{1/3}$ roughly corresponds to the mean inter-particle spacing.

**Choosing the Softening Parameter $\varepsilon$:**

- For testing (a = 100 AU, N ≤ 50): $ε ≈ 1~\text{AU}$
- For production $(a = 1000~\text{AU}, \, N = 100-200)$: $ε ≈ 5~\text{AU}$
- *General rule:* ε should be ~1% of mean inter-particle spacing
- *Too small:* numerical errors when particles approach
- *Too large:* artificially smooths dynamics

**Required units:**

$$
\begin{aligned}
\text{Mass}: &\quad M_{\odot} \quad | \quad \text{Length}: \quad \text{AU} \quad | \quad
\text{Time}: &\quad \text{yr}
\end{aligned}
$$

In these units, the gravitational constant becomes:
$$
G = 4\pi^2~\text{AU}^3 \text{ M}_{\odot}^{-1} \text{ yr}^{-2}
$$

**Energy and Virial Diagnostics**: See Section 7 of the **Kroupa IMF & Plummer Sphere Guide** for complete formulae and physical interpretation.

You must monitor and plot the evolution of:

- Total kinetic energy $E_K$
- Total gravitational potential energy $W$ *(count each pair only once!)*
- Total energy $E_{\rm tot} = E_K + W$
- Virial Ratio $Q = |2E_K + W|/|W|$ (should be < 0.01 for equilibrium)

Monitoring all components separately helps debug which type of energy causes conservation issues.

***Note:** These quantities represent the system values (i.e., summed over all particles). When computing the potential energy, count each pair only once (i.e., use $i<j$ in the sum).*

### ODE Integration Methods Reference

For N-body dynamics, you'll implement four integration schemes. Full derivations in course materials:

- [Euler's Method](https://astrobytes-edu.github.io/astr596-modeling-universe/module3-part1-euler/)
- [Runge-Kutta Methods](https://astrobytes-edu.github.io/astr596-modeling-universe/module3-part2-runge-kutta/)
- [Symplectic Integrators](https://astrobytes-edu.github.io/astr596-modeling-universe/module3-part3-symplectic/)

**Key Algorithmic Structures** for the system $\dot{\vec{y}} = \vec{f}(t, \vec{y})$ where $\vec{y} = [\vec{r}, \vec{v}]$ and $\vec{f}$ returns $[\vec{v}, \vec{a}]$:

**Euler's Method** (1st order, simplest):
$$\vec{y}_{n+1} = \vec{y}_n + \Delta t \cdot \vec{f}(t_n, \vec{y}_n)$$

**RK2** (Midpoint Method; 2nd order):
$$\begin{aligned}
\vec{k}_1 &= \vec{f}(t_n, \vec{y}_n) \\
\vec{k}_2 &= \vec{f}\left(t_n + \frac{\Delta t}{2}, \vec{y}_n + \frac{\Delta t}{2} \vec{k}_1\right) \\
\vec{y}_{n+1} &= \vec{y}_n + \Delta t \cdot \vec{k}_2
\end{aligned}$$

**RK4** (4th order, high accuracy):
$$\begin{aligned}
\vec{k}_1 &= \vec{f}(t_n, \vec{y}_n) \\
\vec{k}_2 &= \vec{f}\left(t_n + \frac{\Delta t}{2}, \vec{y}_n + \frac{\Delta t}{2} \vec{k}_1\right) \\
\vec{k}_3 &= \vec{f}\left(t_n + \frac{\Delta t}{2}, \vec{y}_n + \frac{\Delta t}{2} \vec{k}_2\right) \\
\vec{k}_4 &= \vec{f}(t_n + \Delta t, \vec{y}_n + \Delta t \cdot \vec{k}_3) \\
\vec{y}_{n+1} &= \vec{y}_n + \frac{\Delta t}{6}\left(\vec{k}_1 + 2\vec{k}_2 + 2\vec{k}_3 + \vec{k}_4\right)
\end{aligned}$$

**Leapfrog** (2nd order, symplectic - conserves energy):
$$\begin{aligned}
\vec{v}_{n+1/2} &= \vec{v}_n + \frac{\Delta t}{2} \cdot \vec{a}(\vec{r}_n) \quad \text{(half-step velocity)} \\
\vec{r}_{n+1} &= \vec{r}_n + \Delta t \cdot \vec{v}_{n+1/2} \quad \text{(full-step position)} \\
\vec{v}_{n+1} &= \vec{v}_{n+1/2} + \frac{\Delta t}{2} \cdot \vec{a}(\vec{r}_{n+1}) \quad \text{(half-step velocity)}
\end{aligned}$$

*Critical Implementation Notes*:
- For N-body: $\vec{f}$ computes accelerations from positions
- Leapfrog updates positions and velocities separately (different structure!)
- Store intermediate $\vec{k}$ values efficiently - they're expensive to compute
- Test with two-body first where analytical solution exists

## Steps to Complete the Problem

### Why This Progression?

We build complexity incrementally to isolate and debug different aspects:

- **2-body (Sun-Earth):** Tests your integrators against a known analytical solution. Any errors here are purely from the integration method, not force calculation.
- **3-body (Sun-Earth-Jupiter):** Tests multi-body force summation without overwhelming complexity. You can visually verify Jupiter's 12-year period.
- **N-body:** The full problem where both integration accuracy and computational efficiency matter.

### Why Implement All Four Methods?

You're implementing methods of increasing sophistication to understand the trade-offs:

- **Euler (1st order):** Simplest but unstable - shows why higher-order methods exist
- **RK2 (2nd order):** Better accuracy but still drifts - demonstrates improvement from higher order
- **RK4 (4th order):** Excellent accuracy but not symplectic - energy slowly drifts
- **Leapfrog (2nd order, symplectic):** Only 2nd order but conserves energy exactly - shows importance of structure preservation

**Building from simple to complex helps you:**

- Debug incrementally (if Euler works, your force calculation is correct)
- Understand each algorithm's structure before tackling the next
- Appreciate why complexity is added (each method addresses previous limitations)
- Have a working baseline to compare against when implementing advanced methods

Each step builds on the previous one - don't skip ahead until the current step works correctly!

### Step 1: Two-Body System (Sun-Earth)

**Goal:** Validate your four integration methods against known physics.

**Implementation:**

- Initial conditions: Earth at $r = 1$ AU, circular velocity $v_c = \sqrt{GM/r}$
- Simulate for 10 orbital periods (10 years)
- Test with $dt = 0.01$ years initially
- Implement all four methods: Euler, RK2, RK4, Leapfrog

**What You Should Observe:**

- **Euler:** Spiral trajectory, systematic energy drift
- **RK2:** Better than Euler but still visible drift
- **RK4:** Nearly perfect circle, very small drift
- **Leapfrog:** Perfect circle, energy conserved to machine precision

**After This Step:** Select your two best methods for the remaining work. You should find that **Leapfrog** (for energy conservation) and **RK4** (for accuracy) perform best. If your results don't show this, debug before proceeding!

### Step 2: Three-Body System (Sun-Earth-Jupiter)

**Goal:** Verify correct force summation between multiple bodies.

**Implementation:**

- Sun at origin ($m = 1 \, M_{\odot}$)
- Earth at 1 AU ($m = 3 \times 10^{-6} \, M_{\odot}$)
- Jupiter at 5.2 AU ($m = 9.55 \times 10^{-4} \, M_{\odot}$)
- Simulate for 24 years (captures 2 Jupiter orbits)
- **Use only your two best methods from Step 1** (should be RK4 and Leapfrog)

**Success Criteria:**

- No particles escape or crash into the Sun
- Energy conservation consistent with your Step 1 results
- Jupiter's gravitational perturbations on Earth visible but small
- Earth completes ~24 orbits while Jupiter completes 2

### Step 3: N-Body Star Cluster

**Goal:** Implement realistic star cluster with Monte Carlo initial conditions, first with loops then with vectorization.

**Implementation Phase A - Simple Test with Loops ($N=10$):**

- Uniform masses ($1 \, M_{\odot}$ each)
- Random positions in sphere of radius $R = 100$ AU
- Zero initial velocities
- **Implement with explicit Python loops first**
- **Test both of your best methods from Step 1** (RK4 and Leapfrog)
- **Method selection:** Based on energy conservation comparison, explain why you choose Leapfrog for Phases B and C
  - You should observe that Leapfrog conserves energy exactly while RK4 slowly drifts
  - This energy conservation is critical for long N-body simulations

**Implementation Phase B - Full Physics with Loops:**

- Kroupa IMF mass sampling with $m \in [0.08, 150] \, M_{\odot}$
- Plummer sphere positions with scale radius $a$
- Virial equilibrium velocities: $v_i \sim \mathcal{N}(0, \sigma_{1D}(r_i))$
- **Use Leapfrog integrator only** (based on Phase A results)
- **Still using loops:** Test with $N = [10, 20, 50, 100]$
- **Do NOT attempt $N > 100$ with loops**

**Implementation Phase C - Vectorized Implementation:**
After your loop version works correctly:

- Rewrite force calculation using NumPy broadcasting
- **Continue using Leapfrog** for direct loop vs. vectorized comparison
- No explicit `for i in range(N): for j in range(i+1, N):` loops
- Use array operations: `r_ij` shape `(N, N, 3)` for all pairwise vectors at once
- **First test with same $N = [10, 20, 50, 100]$ as loops for direct comparison**
- **Then explore larger systems:** $N = [200, 500, 1000, 2000, 5000]$ or even $N = 10000$ if performance allows!

**Success Criteria:**

- Loop implementation works correctly for $N \leq 100$
- Vectorized matches loop results exactly (same random seed)
- Vectorization provides significant speedup (10-100×)
- Energy conservation maintained across all implementations
- Can simulate $N \geq 1000$ particles with vectorized code

## Required Plots and Analysis

### Standard Energy Diagnostics (Apply to ALL Steps)
For every simulation phase, create a 3-panel figure showing:
1. **Energy Components:** Plot $E_K$, $W$, and $E_{tot}$ vs time on same axes
2. **Relative Energy Error:** Plot $(E_{tot}(t) - E_{tot,0})/|E_{tot,0}|$ vs time (log scale)
3. **Virial Ratio:** Plot $Q = |2E_K + W|/|W|$ vs time

Expected values:
- Virial ratio: ~0.5 for circular orbits, <0.01 for virialized clusters
- Relative error: Leapfrog ~$10^{-15}$, RK4 ~$10^{-8}$, RK2 ~$10^{-4}$, Euler ~$10^{-2}$

### Step-Specific Additional Plots

**Step 1 (Two-Body):**
- 2×2 comparison figure with all four methods showing:
  - Orbital trajectories (different line styles)
  - Total energy vs time (different colors)
  - Relative energy error (log scale)
  - Virial ratio
- Document which two methods perform best

**Step 2 (Three-Body):**
- Orbital trajectories showing Earth's ~24 orbits during Jupiter's 2
- Energy diagnostics comparing your two best methods

**Step 3A (Simple N-Body):**
- Energy diagnostics comparing RK4 vs Leapfrog to justify method selection

**Step 3B (Full Physics):**
- Initial condition validation (N=100):
  - Log-log mass histogram with Kroupa slopes (-1.3, -2.3)
  - Radial density profile showing Plummer $\rho \propto (1+r^2/a^2)^{-5/2}$
- Cluster evolution snapshots at $t/t_{\text{sim}} = [0, 0.25, 0.5, 0.75, 1.0]$

**Step 3C (Vectorized):**
- Verification: Energy components for loop vs vectorized (N=100, identical)
- Performance analysis:
  - Log-log time vs N for both implementations
  - Overlay $O(N^2)$ reference line
  - Speedup factor (time_loop/time_vectorized) vs N
- Large-N showcase (your largest stable N):
  - 3×2 or 2×3 evolution snapshots at $\tfrac{t}{t_\text{final}} = [0, 0.2, 0.4, 0.6, 0.8, 1.0]$
  - Color by mass, consistent axes, time labels
  - Energy diagnostics showing conservation at large N

### Key Implementation Steps

1. Implement Kroupa mass sampling using analytical normalization (Section 2 of guide)

2. Implement Plummer position sampling (Section 3 of guide)

3. Assign virial equilibrium velocities (Section 3.6 of guide): draw each component from $\mathcal{N}(0, \sigma_{1D}(r))$

4. Apply center-of-mass corrections: subtract $\vec{r}_{\text{cm}}$ and $\vec{v}_\text{cm}$ from all particles

### Validation Checklist

Before production runs, Validate your initial conditions by verifying:

- [ ] Mass histogram (log-log plot) shows Kroupa slopes $(-1.3, -2.3)$
- [ ] Radial density follows Plummer profile
- [ ] $|r_\text{cm}| < 0.001 \times a$ (center of mass position)
- [ ] $|v_\text{cm}| < 0.001 \times σ_\text{3D}$ (center of mass velocity)
- [ ] Initial virial ratio Q < 0.01

## Simulation Parameters

**Timestep Selection:**
- Start with $dt = 0.001$ years for $a = 100$ AU systems
- Use $dt = 0.01$ years for $a = 1000$ AU systems
- Rule of thumb: $dt \approx 0.01 \times$ shortest orbital period
- If energy conservation poor (>1% drift), reduce $dt$ by factor of 2

**Timestep Stability Criterion:**
$$\Delta t < \eta \min\left(\sqrt{\frac{\epsilon}{|\vec{a}|_{\max}}}, \frac{0.01 \times \text{min orbital period}}{2\pi}\right)$$
where $\eta \approx 0.01-0.1$ is a safety factor.

**Simulation Duration:**
- For $a = 100$ AU: simulate for at least 100 years
- For $a = 1000$ AU: simulate for at least 1000 years
- Stop early if any particle reaches $r > 10a$ (escaper)

**Output Management**:
- Save snapshots every 10-100 timesteps (balance file size vs. temporal resolution)
- Store at minimum: positions, velocities, energies (total, kinetic and potential), virial ratio
- For large $N$ and long integration times, consider saving only every 100th timestep

## Implementation Tips

### Data Structure Convention:
- **Positions**: `shape (N, 3)` - each row is `[x, y, z]` for one particle
- **Velocities**: `shape (N, 3)` - each row is `[vx, vy, vz]` for one particle  
- **Forces/Accelerations**: `shape (N, 3)` - each row is `[ax, ay, az]`
- Masses: `shape (N,)` - 1D array of masses

### Testing Strategy

- **Unit tests:** Test each component separately (force calculation, energy computation, single ODE integration step)
- **Conservation tests:** Energy should be conserved (especially for Leapfrog)
- **Known solutions:** Two-body problem has analytical solution
- **Scaling tests:** Verify $O(N^2)$ scaling for force calculations (use `timeit`)

### Common Bug Patterns

Watch out for these frequent mistakes:

1. **Self-force Bug**: Computing force of particle on itself
   - Symptom: NaN values or particles shooting to infinity
   - Fix: Skip when i==j in force loop

2. **Double Counting Bug**: Counting each pair twice in potential energy
   - Symptom: Potential energy exactly 2× what it should be
   - Fix: Use i<j not i≠j in summation

3. **Sign Error Bug**: Wrong sign in force or acceleration
   - Symptom: Particles repel instead of attract
   - Fix: Check force points from i to j

4. **Unit Vector Bug**: Not normalizing direction vectors
   - Symptom: Forces scale incorrectly with distance
   - Fix: Divide by |r_ij| not |r_ij|²

5. **Timestep Bug**: Using different dt in different parts of integrator
   - Symptom: Energy not conserved even with symplectic integrator
   - Fix: Use consistent dt throughout each method

6. **Array Shape Bug**: Mixing (N,3) and (3,N) arrays
   - Symptom: Broadcasting errors or wrong results
   - Fix: Be consistent with array shapes throughout

7. **Accumulation Bug**: Not resetting force array each timestep
   - Symptom: Forces grow without bound
   - Fix: Zero force array before computing new forces

## Suggested Timeline & Milestones

To ensure steady progress and avoid last-minute rushing, aim for these checkpoints:

### By End of Day 3 (Wednesday)
✓ Two-body system working with at least Euler and one other integrator  
✓ Energy conservation plots showing method differences  
✓ Repository set up with initial commits

### By End of Week 1 (Sunday)
✓ All four integrators implemented and tested on two-body problem  
✓ Three-body (Sun-Earth-Jupiter) system working correctly  
✓ Basic N-body working with loops (N=10, uniform masses)  
✓ At least 3-4 meaningful Git commits

### By End of Day 10 (Wednesday of Week 2)
✓ Vectorized implementation complete and tested  
✓ Performance comparison data collected  
✓ Kroupa IMF sampling with analytical normalization working

### By End of Week 2 (Sunday)
✓ Plummer spatial distribution implemented  
✓ Center-of-mass corrections applied  
✓ Chosen extension implemented  
✓ All plots generated  
✓ Both memos written

### Monday (Due Date)
✓ Final code review and cleanup  
✓ Verify all deliverables complete  
✓ Submit by 11:59 PM

**Key Advice**: If you're behind at any checkpoint, come to Student Hacking Hours (Monday/Wednesday 1-2 pm) or post on Slack immediately. Don't wait until the last weekend!

## Example Extension Ideas

**You must implement at least one extension beyond the base requirements.** Extensions are graded on depth of exploration, not just implementation. A thoughtful analysis of one extension is better than superficial implementation of three. Let your curiosity drive you! Talk to me if you want to discuss other extensions of interest.

### Performance Extensions
- **Numba JIT Compilation**: Add a third implementation using Numba's @jit decorator for additional 10-100× speedup
- **Adaptive Timesteps**: Implement timestep adjustment based on local dynamics

### Physics Extensions
- **Alternative IMFs**: Explore bottom-heavy (α > 2.3) or top-heavy (α < 2.3) IMFs and their effects on cluster dynamics
- **Initial Virial Ratio**: Start with Q = 2K/|W| > 1 (super-virial) or Q < 1 (sub-virial) and watch evolution
- **Mass Segregation**: Place massive stars preferentially in center, observe dynamical evolution

### Analysis Extensions
- **Lagrangian Radii**: Track radii containing 10%, 50%, 90% of mass over time
- **Core Collapse**: Measure core density evolution and collapse time
- **Velocity Dispersion Profiles**: Compare radial velocity dispersion to theoretical predictions
- **Relaxation Time**: Calculate and verify two-body relaxation timescale

### Visualization Extensions
- **3D Animation**: Create rotating 3D visualization of cluster evolution
- **Movies/Animations**: Create movies of your simulations using [matplotlib.animation](https://matplotlib.org/stable/api/animation_api.html) or [imageio](https://imageio.readthedocs.io/en/stable/examples.html#gif-files) for GIFs
- **Phase Space**: Plot position-velocity phase space diagrams
- **Energy Distribution**: Show distribution of specific energies of stars
- **Color by Property**: Color particles by mass, energy, or velocity

## Deliverables

Following the [Project Submission Guide](https://astrobytes-edu.github.io/astr596-modeling-universe/project-submission-guide/), you must submit:

### 1. Code Repository
- Well-organized modular code with clear separation of physics, numerics, and visualization
- Both loop-based and vectorized implementations
- Clear README with installation and usage instructions
- At least 5-10 meaningful Git commits showing incremental development

### 2. Research Memo (2-3 pages)
Focus on your N-body cluster simulation results. Your memo should include:

**Main Content:**
- **Executive Summary**: What you accomplished and key findings
- **Methodology**: Your implementation approach and algorithmic choices
- **Results**: Key findings from Step 3 (N-body) with embedded plots:
  - Initial condition validation (Kroupa IMF and Plummer density profile)
  - Energy conservation for N-body cluster ($E_K$, $W$, $E_{tot}$ vs time)
  - Virial ratio evolution
  - Performance comparison plot (loop vs. vectorized)
  - Speedup analysis
  - Large-N showcase visualization
- **Computational Performance**: Runtime analysis, bottlenecks, optimization gains
- **Extensions**: Detailed description and results of your chosen extension
- **Challenges & Solutions**: What was difficult and how you solved it

**Appendices** (not counted toward page limit):
- **Appendix A**: Two-body validation (Step 1)
  - Orbital plots comparing all four methods
  - Energy conservation comparison
  - Brief explanation of why you selected your two best methods
- **Appendix B**: Three-body verification (Step 2)
  - Sun-Earth-Jupiter orbital plot
  - Confirmation of Jupiter's 12-year period
  - Energy components comparison between your two best methods
- **Appendix C**: Additional diagnostics (optional)
  - Any debugging plots that helped you solve problems
  - Additional validation tests
  - Extended performance analysis

### 3. Growth Memo (1 page)
Reflect on your learning process:
- Key learning moments and breakthroughs
- Skill evolution from Project 1 to Project 2
- How you used AI tools appropriately (within course guidelines)
- Next learning goals

### Submission Notes
- Main memo focuses on N-body results; validation work goes in appendices
- All plots must have labeled axes with units
- Include figure captions explaining what each plot shows
- Code should be executable with clear installation instructions

## Appendix: Example Class Structure

## Appendix: Example Code Structure
Here's a skeleton of how you might begin to structure your N-body integrator class. You will need to fill in the methods with the appropriate logic for force calculation, integration steps, and energy computation.

```python
import numpy as np
import ODEIntegrators as odeint  # Your module containing integration methods

class NBodyIntegrator:
    """Base class for N-body integration."""

    def __init__(self, masses, positions, velocities, epsilon=1.0, method='leapfrog'):
        # TODO: Implement
        pass

    def compute_accelerations(self, positions):
        # TODO: Implement
        pass

    def step(self, dt):
        # TODO: Implement
        pass

    def compute_energy(self):
        # TODO: Implement
        pass
```




## Appendix: Visualization Tips for Large N

When visualizing your star clusters, choose your plotting strategy based on particle count:

### Plotting Strategy by Scale
- **N ≤ 10,000**: Use `plt.scatter()` with tuning
- **N > 20,000**: Consider density maps (hexbin/hist2d) with massive stars overlaid
- **N ∈ [10,000-20,000]**: Either approach works - experiment!

### Tips for N ≈ 10,000 Scatter Plots
1. **Sort by depth**: Plot particles sorted by z-coordinate so nearer stars appear on top
2. **Size scaling**: Try `size ∝ m^(2/3)` to make massive stars visible without overwhelming
3. **Transparency**: Use `alpha=0.5-0.7` with `lw=0` (no edge lines)
4. **Performance**: Set `rasterized=True` when saving PDFs to reduce file size
5. **Highlight interesting features**: Overlay the top 100-200 most massive stars with slightly larger, opaque markers

### When Scatter Plots Struggle
If your cluster core looks like an ink blot:
- Switch to `plt.hexbin()` or `plt.hist2d()` for the main distribution
- Overlay massive stars as individual points
- Consider an inset zoom of the dense core region

### Consistent Visualization
- Use `ax.set_aspect('equal')` for proper spatial representation
- Fix axis limits across snapshots for easy comparison
- Always include colorbars and axis labels with units

## Appendix: Essential NumPy Functions for N-body

| Function | Purpose | Example Usage |
|----------|---------|---------------|
| [`np.zeros`](https://numpy.org/doc/stable/reference/generated/numpy.zeros.html) | Initialize arrays | `positions = np.zeros((N, 3))` |
| [`np.random.uniform`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html) | Random sampling | `u = np.random.uniform(0, 1, size=N)` |
| [`np.random.random`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.random.html) | Random [0,1) values | `rand_vals = np.random.random(N)` |
| [`np.linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html) | Vector magnitudes | `r = np.linalg.norm(pos[i] - pos[j])` |
| [`np.cross`](https://numpy.org/doc/stable/reference/generated/numpy.cross.html) | Cross product for perpendicular vectors | `v_perp = np.cross(r_vec, z_hat)` |
| [`np.sum`](https://numpy.org/doc/stable/reference/generated/numpy.sum.html) | Summation with axis control | `total_KE = np.sum(0.5 * m * v**2, axis=0)` |
| [`np.sqrt`](https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html) | Vectorized square root | `distances = np.sqrt(dx**2 + dy**2 + dz**2)` |
| [`np.newaxis`](https://numpy.org/doc/stable/reference/constants.html#numpy.newaxis) | Add dimension for broadcasting | `r_ij = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]` |
| [`np.where`](https://numpy.org/doc/stable/reference/generated/numpy.where.html) | Conditional operations | `massive = masses[np.where(masses > 10)]` |
| [`np.arccos`](https://numpy.org/doc/stable/reference/generated/numpy.arccos.html) | Inverse cosine for angles | `theta = np.arccos(1 - 2*u)` |
| [`np.sin`](https://numpy.org/doc/stable/reference/generated/numpy.sin.html), [`np.cos`](https://numpy.org/doc/stable/reference/generated/numpy.cos.html) | Trig functions | `x = r * np.sin(theta) * np.cos(phi)` |
| [`np.arange`](https://numpy.org/doc/stable/reference/generated/numpy.arange.html) | Integer sequences | `indices = np.arange(N)` |
| [`np.linspace`](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html) | Evenly spaced values | `times = np.linspace(0, t_end, n_steps)` |
| [`np.logspace`](https://numpy.org/doc/stable/reference/generated/numpy.logspace.html) | Log-spaced values for mass bins | `mass_bins = np.logspace(-1, 2, 50)` |
| [`np.histogram`](https://numpy.org/doc/stable/reference/generated/numpy.histogram.html) | Binning data for analysis | `counts, bins = np.histogram(masses, bins=mass_bins)` |
| [`np.triu_indices`](https://numpy.org/doc/stable/reference/generated/numpy.triu_indices.html) | Upper triangular indices | `i, j = np.triu_indices(N, k=1)` for unique pairs |
| [`np.fill_diagonal`](https://numpy.org/doc/stable/reference/generated/numpy.fill_diagonal.html) | Zero self-forces | `np.fill_diagonal(force_matrix, 0)` |
| [`np.clip`](https://numpy.org/doc/stable/reference/generated/numpy.clip.html) | Clamp values to range | `u_safe = np.clip(u, 1e-12, 1-1e-12)` |
| [`np.save`](https://numpy.org/doc/stable/reference/generated/numpy.save.html)/[`np.load`](https://numpy.org/doc/stable/reference/generated/numpy.load.html) | Save/load arrays | `np.save('positions.npy', pos)` |
| [`timeit.timeit`](https://docs.python.org/3/library/timeit.html) | Performance testing | `time = timeit.timeit(lambda: compute_forces(pos), number=100)` |

**Vectorization Tips**:
- Use broadcasting to avoid loops: `r_ij` shape `(N, N, 3)` computes all pairwise differences at once
- Mask diagonal for self-forces: `np.fill_diagonal(force_matrix, 0)`
- Use `axis` parameter in reductions: `np.sum(forces, axis=1)` sums forces on each particle
- For advanced users: `np.einsum` provides elegant notation for tensor operations