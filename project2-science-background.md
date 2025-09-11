# Project 2: Science Background

**ASTR 596 Fall 2025**  
**Instructor:** Anna Rosen  

> **Course ethos:** Glass-box modeling. Every function is physics-motivated and transparent. This document is your end-to-end reference to sample **stellar masses** from the Kroupa IMF and **positions** from a Plummer sphere for star-cluster simulations.

---

## Why these models?

- **Kroupa Initial Mass Function (IMF)**: Empirical, piecewise power-law description of how many stars form at each mass. It matches the local, near-solar-metallicity stellar population and remains a standard baseline for cluster-scale work.
<br>

- **Plummer model**: Simple, analytic, finite-mass, non-singular density profile that approximates bound, relaxed stellar systems. It gives closed-form cumulative mass and an exact inverse-CDF for Monte Carlo sampling of radii.

**Learning goals:**

1. Translate physics → probability distributions → samplers.
2. Build a correct, testable sampler for a broken power law (IMF).
3. Build a correct, testable sampler for a spherical density profile (Plummer).

---

## 2 Canonical Kroupa IMF

We use the **stellar IMF** (counts *individual stars*, not unresolved systems) with lower and upper mass limits $m_{\min}, m_{\max}$ and piecewise power-law slopes. For Project 2 you may use the two-segment form (since $\alpha_2=\alpha_3$ for $m>0.5\,M_\odot$):

$$
\xi(m) = k_\xi\,k_i\, m^{-\alpha_i},\quad m\in[m_{\min}, m_{\max}],
$$

with
$$
\alpha_1 = 1.3\quad (0.08\le m/M_\odot < 0.5),\qquad
\alpha_2 = 2.3\quad (0.5\le m/M_\odot \le m_{\max}).
$$

**Recommended limits**:

- $m_{\min}=0.08\,M_\odot$ (hydrogen-burning limit) 
- $m_{\max}=150\,M_\odot$ (approximate upper stellar mass limit)

### 2.1 Continuity and Normalization

We choose continuity at the break $m_b=0.5\,M_\odot$:

$$
\xi_1(m) = A_1\, m^{-\alpha_1},\quad m\in[m_{\min}, m_b),\qquad
\xi_2(m) = A_2\, m^{-\alpha_2},\quad m\in[m_b, m_{\max}],
$$

with $A_2 = A_1\, m_b^{\alpha_2-\alpha_1}$ so that $\xi_1(m_b)=\xi_2(m_b)$.

Let $N$ be the expected number of stars. Then
$$
N = \int_{m_{\min}}^{m_b} A_1 m^{-\alpha_1}\,dm + \int_{m_b}^{m_{\max}} A_2 m^{-\alpha_2}\,dm.
$$
Solve for $A_1$ given $N$ (or set $k_\xi$ such that $\int \xi\,dm = 1$ and then scale to $N$). For $\alpha\neq 1$,
$$
\int_a^b m^{-\alpha}dm = \frac{b^{1-\alpha}-a^{1-\alpha}}{1-\alpha}.
$$

#### Computing Normalization Constants Analytically

To generate exactly $N$ stars, compute the normalization constants without numerical integration:

**Step 1 - Calculate segment integrals**:

For segment 1 ($m \in [m_{\min}, m_b]$ with slope $\alpha_1$):

$$
I_1 = \frac{m_b^{1-\alpha_1} - m_{\min}^{1-\alpha_1}}{1-\alpha_1}
$$

For segment 2 ($m \in [m_b, m_{\max}]$ with slope $\alpha_2$):

$$
I_2 = \frac{m_{\max}^{1-\alpha_2} - m_b^{1-\alpha_2}}{1-\alpha_2}
$$

**Step 2 - Apply continuity constraint**:

With $A_2 = A_1 \cdot m_b^{\alpha_2-\alpha_1}$ ensuring continuity at $m_b$:

$$
A_1 = \frac{N}{I_1 + m_b^{\alpha_2-\alpha_1} \cdot I_2}
$$

$$
A_2 = A_1 \cdot m_b^{\alpha_2-\alpha_1}
$$

**Step 3 - Compute segment probabilities**:

The probability of drawing from each segment:

$$
P_1 = \frac{A_1 \cdot I_1}{N}, \quad P_2 = \frac{A_2 \cdot I_2}{N}
$$

These probabilities sum to 1 and determine which segment to sample from.

### 2.2 Inverse-transform sampling within one segment

To draw $m$ from $p(m)\propto m^{-\alpha}$ on $[a,b]$ with $\alpha\neq 1$:

1. Precompute the normalization constant:
$$C = \dfrac{b^{1-\alpha}-a^{1-\alpha}}{1-\alpha}$$
2. Draw $u$ from a uniform distribution:
$$u\sim\mathcal U(0,1)$$
3. Invert the CDF:

$$
 m = \big[\,a^{1-\alpha} + u\,(b^{1-\alpha}-a^{1-\alpha})\,\big]^{\!1/(1-\alpha)}.
$$

### 2.3 Piecewise Sampling Algorithm (Two-Segment IMF)

1. Compute segment weights (probabilities) $P_1$ and $P_2$ from the **normalized** integrals of $\xi_1$ and $\xi_2$.
<br>

2. Draw $s\sim\mathcal U(0,1)$. If $s<P_1$ sample from segment 1 using $\alpha_1, a=m_{\min}, b=m_b$; else sample from segment 2 using $\alpha_2, a=m_b, b=m_{\max}$ with the inverse formula above.
<br>

> **Numerical guardrails**: work in double precision, precompute powers $a^{1-\alpha}$, $b^{1-\alpha}$, and clamp $u\in[10^{-12}, 1)$ to avoid boundary issues.

---

## 3 Plummer Sphere for Spatial Positions

### 3.1 Density and cumulative mass

The Plummer density with total mass $M$ and scale radius $a$ is
$$
\rho(r) = \frac{3M}{4\pi a^3}\left(1+\frac{r^2}{a^2}\right)^{-5/2}.
$$
The enclosed mass is
$$
 M(<r) = M\,\frac{r^3}{\big(r^2+a^2\big)^{3/2}}.
$$
Define the CDF for radius: $F(r)\equiv M(<r)/M\in[0,1)$.

### 3.2 Inverse-transform for the Radius

Set $F(r)=u$ with $u\sim\mathcal U(0,1)$. Solve for $r$:
$$
 r = a\,\Big(u^{-2/3}-1\Big)^{-1/2}.
$$
> Draw $u\in[10^{-12},1-10^{-12})$ to avoid infinities.

### 3.3 Sample angles uniformly on a sphere
Draw $\phi=2\pi v$ with $v\sim\mathcal U(0,1)$ and $\cos\theta=1-2w$ with $w\sim\mathcal U(0,1)$. Then
$$
 x=r\sin\theta\cos\phi,\quad y=r\sin\theta\sin\phi,\quad z=r\cos\theta.
$$

### 3.4 Scale by half-mass radius (user-friendly)
If you want a specific half-mass radius $r_{1/2}$:
$
 r_{1/2} = \frac{a}{\sqrt{2^{2/3}-1}}\;\;\Rightarrow\;\; a = r_{1/2}\,\sqrt{2^{2/3}-1} \approx 0.766\, r_{1/2}.
$

### 3.4.1 Typical scales for real star clusters

```{list-table} Plummer Scale Parameters for Different Cluster Types
:header-rows: 1
:widths: 30 20 20 30

* - Cluster Type
  - Scale Radius (a)
  - Half-mass Radius
  - Examples
* - Ultra-compact clusters
  - 0.1-0.3 pc
  - 0.08-0.23 pc
  - Super star clusters, nuclear clusters
* - Young massive clusters
  - 0.3-1 pc
  - 0.23-0.8 pc
  - Arches, Westerlund 1
* - Open clusters
  - 0.5-3 pc
  - 0.4-2.3 pc
  - Pleiades, Hyades
* - Globular clusters
  - 1-5 pc
  - 0.8-4 pc
  - Milky Way globulars
* - OB associations (sparse)
  - 5-20 pc
  - 4-15 pc
  - Extended stellar associations
```

**For computational simulations with N ≤ 200 particles:** Use much smaller scales to represent a cluster core:

- Testing (N ≤ 50): $a = 100$ AU ≈ 0.0005 pc
- Production runs (N = 100-200): $a = 1000$ AU ≈ 0.005 pc
- Full cluster (N > 1000): $a = 0.1-1$ pc (requires advanced algorithms or long runtimes)

### 3.5 Re-centering

After sampling $N$ stars with positions $\{\boldsymbol r_i\}$ and masses $\{m_i\}$, subtract the mass-weighted center of mass so that $$\sum_i m_i\boldsymbol r_i=\boldsymbol 0$$

### 3.6 Virial Equilibrium Velocities

For a Plummer sphere in virial equilibrium, assign velocities by drawing each component independently from a Gaussian distribution with radius-dependent (1D) velocity dispersion:

$$\sigma_{1D}^2(r) = \frac{GM}{6a} \cdot \left(1 + \frac{r^2}{a^2}\right)^{-1/2}$$
where $M$ is the total mass and $a$ is the Plummer scale radius.

**Implementation**: 

For each particle at radius $r$:

1. Calculate $\sigma_{1D}(r)$ using the formula above
2. Draw $v_x \sim \mathcal{N}(0, \sigma_{1D})$
3. Draw $v_y \sim \mathcal{N}(0, \sigma_{1D})$
4. Draw $v_z \sim \mathcal{N}(0, \sigma_{1D})$

**Validation check**: The 3D velocity dispersion should satisfy:
$\sigma_{3D} = \sqrt{\langle v^2 \rangle} = \sqrt{3} \cdot \sigma_{1D}$

After assigning velocities, subtract the mass-weighted center-of-mass velocity to ensure $\sum_i m_i \vec{v}_i = 0$.

This initialization produces a system with virial ratio $Q \approx 0$ (equilibrium). If your initial $Q > 0.01$, check your velocity assignment.

---

## 4 Putting It All Together - Initial Conditions Sampler Blueprint

**Inputs**: $N$, $r_{1/2}$ (or $a$), $m_{\min}, m_{\max}$.

**Outputs**: arrays of stellar masses $m_i$, positions $(x_i,y_i,z_i)$, and initial velocities $(v_{x,i}, v_{y,i}, v_{z,i})$.

### 4.1 If $N$ is given
1. Precompute segment probabilities $P_1, P_2$.
2. For each star: choose segment → draw $m_i$ via inverse-CDF.
3. Draw $u,v,w$ → compute $r_i,\theta_i,\phi_i$ → $(x_i,y_i,z_i)$.
4. Recentre to the mass-weighted COM.
5. Draw velocities from Gaussian with $\sigma_{1D}(r_i)$ → $(v_{x,i}, v_{y,i}, v_{z,i})$.

> **Avoid rescaling all masses post-hoc** (it distorts the IMF).

---

## 5) Validation & Diagnostics (Required)

### 5.1 IMF checks

- **Segment counts**: fraction below/above $0.5\,M_\odot$ matches analytic expectations within sampling error.
- **Slope recovery**: fit a line to log-binned $dN/d\log m$ in each segment; slopes $\approx \{-\alpha_1, -\alpha_2\}$.
- **Mean mass**: compare sample mean $\langle m\rangle$ to the analytic mean
$$\langle m\rangle = \frac{\int m\,\xi(m)\,dm}{\int \xi(m)\,dm}\,$$ computed analytically using the closed-form integrals for each power-law segment

### 5.2 Plummer checks

- **CDF check**: verify that $u_i = r_i/\sqrt{r_i^2+a^2}$ raised to $3$ matches $\mathcal U(0,1)$. Equivalently, verify that
$$\frac{r_i^3}{(r_i^2+a^2)^{3/2}}$$ is uniform.
- **Shell counts**: number density in spherical shells: $n \propto (1+r^2/a^2)^{-5/2}$.
- **Half-mass radius**: compute $r_{1/2,\rm sample}$ and compare to target.

---

## 6) Common pitfalls (and fixes)

1. **Forgetting normalization** of piecewise segments. *Fix*: compute normalized integrals for segment weights.
2. **Uniform in $\log m$** instead of a power law in $m$. *Fix*: use inverse-CDF for $m^{-\alpha}$, not uniform in $\log m$ unless $\alpha=1$.
3. **Segment weighting error**. *Fix*: compute normalized segment probabilities from integrals, not from slopes or widths.
4. **Angular sampling bias**. *Fix*: sample $\cos\theta\sim\mathcal U(-1,1)$, $\phi\sim\mathcal U(0,2\pi)$.
5. **Radius divergence** at $u\to 1$. *Fix*: clamp $u$ away from 0 and 1.
4. **Rescaling masses to hit $M_{\rm cl}$**. *Fix*: prefer stop-at-or-above or reject-last; document choice.
6. **Forgetting Center-of-Mass (COM) recentering**. *Fix*: subtract mass-weighted COM from positions.

---

## 7) Energy and Virial Diagnostics for N-body Systems

### 7.1 Energy Components
For an N-body gravitational system, monitor these quantities throughout your simulation:

**Kinetic Energy** (sum over all particles):
$$
E_K = \frac{1}{2} \sum_{i=1}^{N} m_i v_i^2
$$
where $v_i^2 = v_{x,i}^2 + v_{y,i}^2 + v_{z,i}^2$ for star $i$.

**Gravitational Potential Energy** (sum over unique pairs):
$$
W = -\sum_{i<j} \frac{Gm_im_j}{r_{ij}}
$$
**Critical:** Count each pair only once using $i<j$, not $i \neq j$ which would double-count.

**Total Energy**:
$$
E_{\rm tot} = E_K + W
$$

### 7.2 The Virial Theorem
For a self-gravitating system in equilibrium, the time-averaged kinetic and potential energies satisfy:
$$
\langle 2E_K + W \rangle = 0
$$

Define the **Virial Ratio**:
$$
Q = \frac{|2E_K + W|}{|W|}
$$

**Physical Interpretation:**
- $Q \approx 0$: System in virial equilibrium (bound and stable)
- $Q > 0.01$: System out of equilibrium (will evolve)
- $2E_K > |W|$: System is "super-virial" (will expand)
- $2E_K < |W|$: System is "sub-virial" (will contract)

### 7.3 Using Energy Conservation for Debugging
Monitor all three components ($E_K$, $W$, $E_{\rm tot}$) separately:
- If $E_{\rm tot}$ drifts but $E_K$ is stable → potential energy calculation error
- If $E_{\rm tot}$ drifts but $W$ is stable → kinetic energy or velocity update error
- If both drift proportionally → force calculation or integration scheme error
- Sudden jumps → likely self-force bug or sign error

**Expected Conservation by Method:**
- **Euler/RK2/RK4**: $E_{\rm tot}$ will drift (non-symplectic)
- **Leapfrog**: $E_{\rm tot}$ conserved to machine precision (~$10^{-15}$)

The virial ratio should remain < 0.01 for a properly initialized cluster.

---

## 8) Optional Project Extension Ideas

- **Metallicity or density dependence**: allow low-mass slopes ($\alpha_1,\alpha_2$) and/or the high-mass slope to vary with environmental parameters.

- **Mass segregation**: sample positions conditionally on mass, e.g. $r(m)\sim a(m)\,(u^{-2/3}-1)^{-1/2}$ with $a(m)=a_0\,(m/\langle m\rangle)^{-\beta}$.

- **Velocities**: draw from the Plummer distribution function to start near virial equilibrium, or rescale drawn velocities to a desired virial ratio.

---

## 9) Quick-reference formulas

**IMF inverse within [a,b], $\alpha\ne 1$**:
$$
 m(u)=\Big[a^{1-\alpha}+u\,(b^{1-\alpha}-a^{1-\alpha})\Big]^{1/(1-\alpha)}\,,\quad u\sim\mathcal U(0,1).
$$
**Plummer radius inverse**:
$$ r(u)= a\,\big(u^{-2/3}-1\big)^{-1/2},\quad u\sim\mathcal U(0,1). $$
**Half-mass scaling**: $a = r_{1/2}\,\sqrt{2^{2/3}-1}\approx 0.766\,r_{1/2}$.
**Continuity constant**: $A_2=A_1\,m_b^{\alpha_2-\alpha_1}$.

---

### Code Verification Checklist 

*Before you perform production runs.*

- [ ] Verified slopes from a log-histogram of sampled masses.
- [ ] CDF test for Plummer radii passes (uniformity of $r^3/(r^2+a^2)^{3/2}$).
- [ ] Sample half-mass radius matches target within sampling error.
- [ ] Documented choice for mass-budget handling (stop-at-or-above vs reject-last).
- [ ] Code has named constants, units on plots, and asserts for numerical guards.

*End of guide.*