# MSF-UDE — Multi-Strain Fitness via Universal Differential Equations

A method to **jointly** estimate effective reproduction numbers (`Rt`) and variant
fitness from epidemic data, by fitting a single mechanistic multi-strain
compartmental model whose time-varying quantities are **neural functions learned
inside a stiff ODE solver**.

This repository is the **early-stage testbed** for that program: minimal, readable
simulators and inference scripts that establish the mechanics and stress-test the
central identifiability question *before* any real data or probabilistic layer is
added.

---

## Why this project exists

Current practice estimates the overall time-varying reproduction number `Rt` and
per-variant growth advantages in **two separate steps**, converting a growth
advantage into a relative reproduction number by plugging in a fixed
generation-time assumption. That split:

- severs uncertainty propagation between the two estimates, and
- inherits the generation-interval bias of Wallinga & Lipsitch (2007).

MSF-UDE instead fits one mechanistic system in which the unknown time-varying
drivers — overall transmission and per-variant **log-fitness** `s(t)` — are neural
functions learned inside a real ODE integrator. The scientific core is an
**identifiability-aware decomposition** of *effective* fitness into intrinsic
transmissibility versus differential susceptible depletion / immune escape.


---

## The core idea in one model

A two-strain SIR with a shared susceptible pool. Variant A is the reference; variant
B's transmission is scaled by a (possibly time-varying) log-fitness `s`:

```
βB(t) = βA · exp(s(t))          # s > 0 ⇒ B is intrinsically fitter
```

The **observable** signal is variant B's share of new infections,

```
freqB(t) = incB / (incA + incB)
```

and the inference problem is to recover `s` (a scalar, or a function `s(t)`) from
`freqB`. The mechanism is embedded in a stiff ODE, so the physics is enforced
exactly by the solver rather than through a soft penalty.

---

## Stack

Python + JAX ecosystem:

- **[JAX](https://github.com/google/jax)** — autodiff and JIT.
- **[diffrax](https://github.com/patrick-kidger/diffrax)** — differentiable ODE
  solvers (`Tsit5`, PID step control); we differentiate through the integrator.
- **[optax](https://github.com/google-deepmind/optax)** — Adam optimizer.
- **SciPy** (`solve_ivp`) — trusted forward simulation to generate ground-truth data.
- **matplotlib** — plots.

> The project plan flags **Julia SciML** as the alternative stack for the mature
> phases (stiffer solvers, adjoint sensitivity); these Python scripts are the fast
> iteration testbed.

### Setup

```bash
pip install jax jaxlib diffrax optax scipy numpy matplotlib
```

### Running

Each script is standalone:

```bash
python two_strain_sim.py                              # forward sim + signal plots
python fitness_recovery.py                            # recover scalar s
python neural_s.py                                    # recover s(t); writes neural_s_fit.png
python identifiability-test/two-phase-simulate2.py    # immune-escape forward sim
python identifiability-test/neural_s_immune_escape.py # the confounding demonstration
```

---

## Key references

- Wallinga & Lipsitch (2007), *Proc. R. Soc. B* — generation intervals ↔ `Rt`.
- Figgins & Bedford (2022) — variant `Rt` differences (GARW).
- Obermeyer et al. (2022), *Science* — mutation-level fitness (PyR0).
- Rackauckas et al. (2020) — Universal Differential Equations.
- Qin (2026), arXiv:2605.30382 — growth-rate ↔ reproduction-number bridge.

(Full annotated list in the project document.)
