# MSF-UDE Cookbook: Build the Multi-Strain Fitness / Reproduction-Number Model From Zero

This is the step-by-step companion to the project plan. It assumes **no prior knowledge** of epidemiology, neural ODEs, or Bayesian statistics. If you can open a terminal and copy-paste, you can follow this. Every code block is runnable. Concepts are explained in plain English the first time they appear.

Work through the phases **in order** — each one builds on the code from the previous one. Budget roughly: Phase 0–1 a day, Phase 2–3 a week, Phase 4 a week, Phase 5 open-ended.

---

## How to use this cookbook

Each phase has the same layout:

- **Goal in one sentence** — what you're building.
- **Plain-English idea** — the concept, no jargon.
- **Do this** — exact commands and code.
- **What you should see** — the actual expected output, so you know it worked.
- **Why it matters** — how this feeds the science.
- **Definition of done** — the checklist to pass before moving on.
- **If it breaks** — the common errors.

A convention: `>` means type this in your terminal. Code blocks are files you save and run.

---

## Prerequisites and the 90-second mental model

Before any code, hold these five ideas in your head. Everything else is detail.

1. **Reproduction number `Rt`** = average number of people one infected person infects at time `t`. Above 1 the epidemic grows; below 1 it shrinks. It changes over time.
2. **Growth rate `r`** = how fast case numbers rise or fall per day (the slope of the case curve on a log scale). `r` and `Rt` describe the same thing in different units; converting between them needs the **generation interval** (the typical gap between one infection and the next it causes).
3. **Variant** = a genetic version of the virus. **Fitness** = how much faster one variant grows than another. You read fitness off the *share* of samples that are each variant over time (an S-shaped curve), **not** off the total case count.
4. **These are two separate signals from two separate data streams**: total cases tell you `Rt`; the variant mix tells you fitness. A new variant can be winning (rising share) while total cases fall.
5. **The whole project** = estimate both signals in one model so their uncertainties are shared, instead of estimating them separately and gluing the results together.

You need a computer with a terminal (Mac, Linux, or Windows with WSL). No GPU required for the small examples.

---

## Phase 0 — Set up the tools and run your first epidemic simulation

**Goal:** install everything and simulate a two-variant outbreak so you can *see* the two signals.

**Plain-English idea:** an epidemic model is a set of buckets — Susceptible (can catch it), Infected (has it), Recovered (immune) — and rules for how people move between buckets. We write those rules as equations, hand them to a solver, and the solver plays the movie forward in time. We'll use **two** infected buckets, one per variant, and make variant B spread faster than variant A.

### Do this — 0.1 Install Python and packages

If you don't have Python, install [Miniconda](https://docs.conda.io/en/latest/miniconda.html), then:

```
> conda create -n msfude python=3.12 -y
> conda activate msfude
> pip install numpy scipy matplotlib
```

(If `pip` complains about a "managed environment," add `--break-system-packages`.)

### Do this — 0.2 Simulate a two-strain outbreak

Save as `phase0_simulate.py`:

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- fixed numbers describing the disease and population ---
N = 1_000_000        # total people
gamma = 1/5          # recovery rate: 1/(5-day infectious period)
beta1 = 0.30         # transmission rate of variant A (older)
beta2 = 0.45         # transmission rate of variant B (newer, FITTER: 0.45 > 0.30)

# --- the model rules: how the buckets change each day ---
def two_strain_sir(t, y):
    S, I1, I2, R = y                 # unpack the four buckets
    infA = beta1 * I1 * S / N        # new A infections per day
    infB = beta2 * I2 * S / N        # new B infections per day
    dS  = -(infA + infB)             # susceptibles leave as they get infected
    dI1 =  infA - gamma * I1         # A: gain infections, lose recoveries
    dI2 =  infB - gamma * I2         # B: same
    dR  =  gamma * (I1 + I2)         # everyone recovers eventually
    return [dS, dI1, dI2, dR]

# --- starting point: almost everyone susceptible, mostly A, a seed of B ---
y0 = [N - 110, 100, 10, 0]
days = np.arange(0, 201)
sol = solve_ivp(two_strain_sir, (0, 200), y0, t_eval=days, rtol=1e-8, atol=1e-8)
S, I1, I2, R = sol.y

# --- derive the two signals ---
incA = beta1 * I1 * S / N
incB = beta2 * I2 * S / N
total_incidence = incA + incB                 # SIGNAL 1: size of epidemic
freqB = incB / (incA + incB)                  # SIGNAL 2: variant B's share

# --- sanity check: nobody should appear or vanish ---
total_people = S + I1 + I2 + R
print("population drift (should be ~0):", total_people.max() - total_people.min())
print("variant B share on days 0/50/100/150:",
      *[round(freqB[d], 3) for d in (0, 50, 100, 150)])

# --- plot the two signals ---
fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax[0].plot(days, total_incidence); ax[0].set_ylabel("new infections / day")
ax[0].set_title("Signal 1 — size of the whole epidemic")
ax[1].plot(days, freqB); ax[1].set_ylabel("variant B share")
ax[1].set_title("Signal 2 — which variant is winning"); ax[1].set_xlabel("day")
plt.tight_layout(); plt.savefig("phase0.png", dpi=120)
print("saved phase0.png")
```

Run it:

```
> python phase0_simulate.py
```

### What you should see

```
population drift (should be ~0): 8.1e-10
variant B share on days 0/50/100/150: 0.13 0.991 0.998 0.999
```

and a two-panel picture: the top panel is a single bump (the epidemic grows then shrinks); the bottom panel is an S-curve climbing from ~0.13 to ~1.0 (variant B taking over). **This is the entire conceptual foundation made concrete.** Notice B's share keeps climbing even as the top panel comes back down — the two signals are independent.

### Why it matters

`beta2 > beta1` is variant B's fitness advantage. The S-curve's steepness encodes that advantage. Later phases will *reverse-engineer* that advantage from the S-curve, which is exactly what real fitness estimation does.

### Definition of done

- [ ] Script runs without error.
- [ ] Population drift is tiny (< 1e-6) — this is your conservation check.
- [ ] The `phase0.png` shows a bump on top and an S-curve on the bottom.

### If it breaks

- `ModuleNotFoundError` → you didn't `conda activate msfude` first, or the `pip install` failed.
- Population drift is large → you edited an equation so the `dS/dI/dR` terms no longer sum to zero. They must: every arrow out of one bucket is an arrow into another.

---

## Phase 1 — Recover a hidden fitness value from data (the first real model)

**Goal:** pretend you *don't* know variant B's fitness, and have the computer figure it out from the variant-share curve. Then run the experiment that reveals when this is impossible.

**Plain-English idea:** we made data with a known fitness. Now we hide the fitness, start from a wrong guess, and let the computer nudge the guess until its simulated S-curve matches the data. The nudging uses **gradients** — the computer can compute "if I increase fitness a little, does my error go up or down?" and step in the improving direction. This is the same machinery as training a neural network. The tool that lets us take gradients *through an ODE solver* is called **automatic differentiation**, provided by **JAX**, and the differential-equation solver that works with it is **diffrax**.

### Do this — 1.1 Install the fitting stack

```
> pip install jax diffrax optax
```

- **JAX** = NumPy that can take gradients automatically.
- **diffrax** = ODE solver written in JAX (so gradients flow through it).
- **optax** = the optimizer that does the nudging (Adam).

### Do this — 1.2 Recover a single hidden fitness number

Save as `phase1_recover.py`:

```python
import jax, jax.numpy as jnp, diffrax, optax
import numpy as np
from scipy.integrate import solve_ivp

N, gamma, beta1 = 1_000_000, 1/5, 0.30
true_s = 0.4                              # THE SECRET we will try to recover
beta2 = beta1 * np.exp(true_s)            # fitness s means beta2 = beta1 * e^s

# 1) make the "observed" data with scipy (easy forward simulation)
def truth(t, y):
    S, I1, I2, R = y
    a, b = beta1*I1*S/N, beta2*I2*S/N
    return [-(a+b), a-gamma*I1, b-gamma*I2, gamma*(I1+I2)]
sol = solve_ivp(truth, (0,120), [N-110,100,10,0],
                t_eval=np.arange(0,121), rtol=1e-9, atol=1e-9)
S, I1, I2, R = sol.y
freqB_obs = jnp.array((beta2*I2*S/N) / (beta1*I1*S/N + beta2*I2*S/N))

# 2) the SAME model, but now `s` is unknown (passed in as `args`)
def ode(t, y, s):
    b2 = beta1 * jnp.exp(s)
    S, I1, I2, R = y
    a, b = beta1*I1*S/N, b2*I2*S/N
    return jnp.array([-(a+b), a-gamma*I1, b-gamma*I2, gamma*(I1+I2)])

term, ts = diffrax.ODETerm(ode), jnp.arange(0., 121.)
def simulate(s):
    sol = diffrax.diffeqsolve(
        term, diffrax.Tsit5(), t0=0., t1=120., dt0=0.1,
        y0=jnp.array([N-110., 100., 10., 0.]), args=s,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=1e-7, atol=1e-7),
        max_steps=100000)
    S, I1, I2, R = sol.ys.T
    b2 = beta1*jnp.exp(s)
    return (b2*I2*S/N) / (beta1*I1*S/N + b2*I2*S/N)

# 3) "loss" = how wrong we are. Minimize it.
def loss(s):
    return jnp.mean((simulate(s) - freqB_obs)**2)

s = jnp.array(0.0)                         # deliberately WRONG starting guess
opt = optax.adam(0.05); state = opt.init(s)
step = jax.jit(jax.value_and_grad(loss))  # value_and_grad = error + its gradient
for i in range(200):
    l, g = step(s)
    updates, state = opt.update(g, state)
    s = optax.apply_updates(s, updates)

print(f"true s = {true_s} | recovered s = {float(s):.4f} | loss = {float(l):.1e}")
```

### What you should see

```
true s = 0.4 | recovered s = 0.4000 | loss = 4.1e-10
```

The computer recovered the hidden fitness to four decimals starting from a wrong guess. **That is the core inverse-problem engine of the whole project.** Everything later is this same loop with more moving parts.

### Do this — 1.3 Let fitness vary over time (a tiny neural network)

Real fitness isn't a single number; it drifts (immune escape builds up). We replace the number `s` with a **small neural network** `s(t)` — a flexible function whose shape is learned. This is what "UDE" (Universal Differential Equation) means: a neural network living *inside* the differential equation, standing in for the part we don't know.

Save as `phase1_timevarying.py`:

```python
import jax, jax.numpy as jnp, diffrax, optax, numpy as np
from scipy.integrate import solve_ivp

N, gamma, beta1 = 1_000_000, 1/5, 0.30
def true_s(t):                            # secret: an S-shaped rise 0.1 -> 0.5
    return 0.1 + 0.4/(1+np.exp(-(t-40)/8))

def truth(t, y):
    S,I1,I2,R = y; b2 = beta1*np.exp(true_s(t))
    a,b = beta1*I1*S/N, b2*I2*S/N
    return [-(a+b), a-gamma*I1, b-gamma*I2, gamma*(I1+I2)]
sol = solve_ivp(truth,(0,120),[N-110,100,10,0],
                t_eval=np.arange(0,121),rtol=1e-9,atol=1e-9)
S,I1,I2,R = sol.y
st = np.array([true_s(t) for t in range(121)])
freqB_obs = jnp.array((beta1*np.exp(st)*I2*S/N)/(beta1*I1*S/N+beta1*np.exp(st)*I2*S/N))

# a small multilayer perceptron (MLP): input = time, output = fitness s(t)
def init_mlp(key, w=16):
    k1,k2,k3 = jax.random.split(key,3)
    return dict(W1=jax.random.normal(k1,(1,w))*0.3, b1=jnp.zeros(w),
                W2=jax.random.normal(k2,(w,w))*0.3, b2=jnp.zeros(w),
                W3=jax.random.normal(k3,(w,1))*0.3, b3=jnp.zeros(1))
def mlp(p, t):
    x = jnp.array([t/120.])               # scale time into [0,1]
    x = jnp.tanh(x@p['W1']+p['b1']); x = jnp.tanh(x@p['W2']+p['b2'])
    return (x@p['W3']+p['b3'])[0]

def ode(t, y, p):
    s = mlp(p, t); b2 = beta1*jnp.exp(s)
    S,I1,I2,R = y; a,b = beta1*I1*S/N, b2*I2*S/N
    return jnp.array([-(a+b), a-gamma*I1, b-gamma*I2, gamma*(I1+I2)])
term, ts = diffrax.ODETerm(ode), jnp.arange(0.,121.)
def simulate(p):
    sol = diffrax.diffeqsolve(term, diffrax.Tsit5(), 0.,120.,0.1,
        jnp.array([N-110.,100.,10.,0.]), args=p,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=1e-6,atol=1e-6),
        max_steps=200000)
    S,I1,I2,R = sol.ys.T
    b2 = beta1*jnp.exp(jax.vmap(lambda t: mlp(p,t))(ts))
    return (b2*I2*S/N)/(beta1*I1*S/N+b2*I2*S/N)
def loss(p): return jnp.mean((simulate(p)-freqB_obs)**2)

p = init_mlp(jax.random.PRNGKey(0)); opt = optax.adam(3e-3); state = opt.init(p)
step = jax.jit(jax.value_and_grad(loss))
for i in range(1500):
    l,g = step(p); u,state = opt.update(g,state); p = optax.apply_updates(p,u)

srec = jax.vmap(lambda t: mlp(p,t))(ts)
strue = np.array([true_s(t) for t in range(121)])
for d in (20,60,100):
    print(f"day {d}: recovered s={float(srec[d]):.3f}  true s={strue[d]:.3f}")
```

### What you should see

Something like:

```
day 20: recovered s=0.159  true s=0.130
day 60: recovered s=0.435  true s=0.470
day 100: recovered s=0.664  true s=0.500
```

The middle of the trajectory is recovered well; the **tail (day 100) drifts off**. This is not a bug — it is the project's central problem showing up on day one. By day 100 variant B's share has saturated near 1.0, so the data barely changes and no longer pins down the fitness. The network fills the vacuum with whatever it likes. **Remember this:** wherever the data stops constraining a quantity, a flexible model will invent values with false confidence. That is why Phase 4 (uncertainty) and the identifiability controls exist.

### Do this — 1.4 The identifiability experiment (the make-or-break test)

**This is the single most important experiment in the project.** The question: can the model tell apart two *different* biological stories that produce the *same* variant-share curve?

- **Story A (intrinsic):** variant B genuinely transmits faster (`beta2 > beta1`), susceptibles shared.
- **Story B (immune escape):** both variants transmit equally, but variant B can infect a pool of people who are already immune to A. B's share rises for a completely different reason.

Recipe:

1. Copy `phase0_simulate.py` twice. In copy A, keep `beta2 > beta1`. In copy B, set `beta2 = beta1` but give variant B its **own** partially-susceptible pool (add a compartment `S_B` of people immune to A but not B, and route B's infections through it). Tune the numbers so both copies produce a **visually identical** `freqB` curve.
2. Fit the Phase 1.3 model (which only knows about *intrinsic* fitness) to **both** datasets.
3. Inspect what the model infers.

**What the result tells you:**

- If the intrinsic-only model fits Story B's data just as well as Story A's by bending `s(t)`, then intrinsic transmissibility and immune escape are **confounded** — you cannot separate them from share data alone. (This is the expected outcome, and it matches the field's understanding: the quantity you actually measure is *effective* fitness, everything lumped together.)
- The fix, tested in later phases: add an independent data stream (serology / immunity surveys) or restrict the fitness network's flexibility so it can't silently absorb the depletion.

### Why it matters

Phase 1.4 defines whether your eventual model is measuring what it claims. A paper that honestly maps *when* the decomposition is and isn't identifiable is a real contribution on its own.

### Definition of done

- [ ] Single hidden fitness recovered to ≈ 0.40 (1.2).
- [ ] Time-varying fitness recovered in the middle, drifting at the saturated tail (1.3).
- [ ] Two datasets built with identical share curves but different mechanisms, and you've observed whether the intrinsic-only model can tell them apart (1.4).

### If it breaks

- `max_steps` error from diffrax → increase `max_steps`, or loosen `rtol/atol` to `1e-5`.
- Loss stuck / NaN → lower the learning rate (`optax.adam(1e-3)`); check you scaled time (`t/120`) before the MLP.
- Fit is poor everywhere (not just the tail) → train longer (more iterations) or widen the MLP (`w=32`).

---

## Phase 2 — Use realistic data and prove the joint model beats the two-step pipeline

**Goal:** replace the perfect S-curve with the messy data you'd actually have (noisy case counts + a handful of sequenced samples per week), fit `Rt` and fitness **together**, and show that doing it together gives better-calibrated answers than doing it in two separate steps.

**Plain-English idea:** real data is counted, not continuous. Cases arrive as noisy integers; sequences are a small random sample of cases labeled by variant. We describe each with a **likelihood** — a statement of how probable the observed counts are given the model. Two likelihoods here:

- **Negative binomial** for case counts: a standard choice for "count data that's more variable than a plain average would predict."
- **Multinomial** for sequences: given you sequenced 40 samples this week and the model says variant B is 30% of infections, how likely were you to see, say, 11 B's? The multinomial answers that.

### Do this — 2.1 Generate realistic noisy data

Take the Phase 1.3 truth simulation and, at weekly time points, (a) draw a noisy case count around the true incidence, and (b) draw a small multinomial sample of variant labels around the true frequency:

```python
import numpy as np
rng = np.random.default_rng(0)
weeks = np.arange(0, 121, 7)
true_incidence = (incA + incB)[weeks]                 # from your truth run
cases_obs = rng.negative_binomial(10, 10/(10+true_incidence*0.3))  # noisy, under-reported
n_sequenced = 40                                       # samples sequenced per week
B_counts = rng.binomial(n_sequenced, freqB_true[weeks])# variant B labels seen
```

(Adjust the reporting fraction `0.3` and dispersion `10` to taste. `freqB_true` is the true B share from your truth run.)

### Do this — 2.2 Write the joint likelihood and fit

Replace the mean-squared-error loss from Phase 1 with the sum of two **negative log-likelihoods** (lower = data more probable). In JAX:

```python
import jax.scipy.stats as jstats
def neg_log_like(params):
    inc_pred, freqB_pred = simulate_full(params)       # model incidence + B share at `weeks`
    # cases: negative binomial around predicted (reported) incidence
    rep = params['report_frac'] * inc_pred
    ll_cases = jstats.nbinom.logpmf(cases_obs, n=10, p=10/(10+rep)).sum()
    # sequences: binomial (2-variant special case of multinomial)
    ll_seq = jstats.binom.logpmf(B_counts, n=n_sequenced, p=freqB_pred).sum()
    return -(ll_cases + ll_seq)
```

Optimize with optax exactly as in Phase 1 (`value_and_grad(neg_log_like)`), now over a parameter dictionary that includes the fitness MLP **and** an overall transmission function `u(t)` (a second small MLP multiplying `beta1`) **and** a reporting fraction.

### Do this — 2.3 Build the two-step baseline and compare

- **Two-step (current standard):** Step 1, estimate `Rt` from cases alone. Step 2, *separately* fit fitness from sequences, treating a fixed generation time as known. Glue.
- **Joint (yours):** the single fit from 2.2.

Compare them not on the point estimate but on **calibration**: generate ~100 fresh noisy datasets from the same truth, fit each with both methods, and count how often each method's 95% interval actually contains the true fitness. A well-calibrated method hits ≈ 95%. The claim you're testing (and expect to confirm) is that the two-step method is **over-confident** (coverage below 95%) because it throws away the generation-time and `Rt` uncertainty at the seam, while the joint method is closer to nominal.

> Note: "95% interval" needs some notion of uncertainty. For Phase 2 a quick-and-dirty version is a **bootstrap** (refit on many resampled datasets and take the spread). The statistically clean version comes in Phase 4; you can revisit this comparison then.

### Why it matters

This is the experiment that justifies the entire project. If the joint model isn't better calibrated than two-step, the project's premise is wrong — so run this early.

### Definition of done

- [ ] Noisy weekly cases + sequence counts generated.
- [ ] Joint negative-log-likelihood fit runs and recovers fitness and `Rt`.
- [ ] Two-step baseline implemented.
- [ ] Coverage compared across ~100 replicate datasets; you can state which method is better calibrated and by how much.

### If it breaks

- Likelihood is `-inf` → your predicted probability hit exactly 0 or 1, or predicted incidence went negative. Clamp predictions into a safe range (e.g. `jnp.clip(p, 1e-6, 1-1e-6)`).
- The two MLPs "fight" (overall transmission and fitness trade off) → this is the identifiability issue again; give the fitness MLP fewer parameters than the transmission MLP, and add a smoothness penalty (see Phase 4).

---

## Phase 3 — Add wastewater and make the generation interval mechanistic

**Goal:** fold in a third data stream (virus in sewage) and upgrade the model so the `growth-rate → Rt` conversion is built from real biology instead of an accidental assumption.

**Plain-English idea, part 1 (wastewater):** infected people shed virus into sewage for several days after infection. So the RNA concentration measured at the treatment plant is a *blurred, delayed* copy of the infection curve. We model that blur as a **convolution**: today's signal = a weighted sum of recent days' infections, weights given by a shedding curve.

```python
shed = np.array([0.05,0.15,0.25,0.25,0.15,0.10,0.05])   # shedding weights, days since infection
def wastewater_pred(incidence):
    return np.convolve(incidence, shed, mode='full')[:len(incidence)]
```

Add a third likelihood term (e.g. log-normal) comparing this to measured concentrations. **Watch the trap:** if variant B sheds differently from A, wastewater is no longer a variant-neutral thermometer and its bias leaks into your fitness estimate — so include a per-variant shedding scale and see whether the data can identify it.

**Plain-English idea, part 2 (generation interval):** in the simple SIR model, the "waiting time" between infections is secretly locked to an unrealistic (exponential) shape. That shape controls how a measured growth rate converts into `Rt` (the Wallinga–Lipsitch relationship), so getting it wrong biases `Rt`. The fix is to split the single infected bucket into a **chain of `k` sub-buckets** (Erlang/gamma stages). Passing through `k` stages in series gives a realistic bell-shaped waiting time.

### Do this — 3.1 Upgrade SIR to staged SEIR

Replace each `I` with an Exposed chain `E_1..E_k` then an Infectious chain `I_1..I_k`, per variant. Movement is `E_1 → E_2 → ... → I_1 → ... → I_k → R`, each step at rate `k * rate` so the *mean* stays fixed while the *shape* becomes realistic. Concretely for one variant with `k=3` infectious stages:

```python
# infectious sub-compartments I_1,I_2,I_3; total infectious = I_1+I_2+I_3
# each transitions at rate 3*gamma (so mean infectious period is still 1/gamma)
dI1 =  new_infections - 3*gamma*I1
dI2 =  3*gamma*I1     - 3*gamma*I2
dI3 =  3*gamma*I2     - 3*gamma*I3
dR  =  3*gamma*I3
```

### Do this — 3.2 Quantify the bridge effect (answers RQ3)

Fit the model twice on the same data — once with `k=1` (the accidental-exponential case) and once with `k=5` — and compare the recovered variant-`Rt` contrasts. The gap between them is the bias the field usually ignores. Report it.

### Why it matters

This is where your model stops "borrowing" the growth-rate-to-`Rt` bridge as a fixed scalar and starts *generating* it mechanistically — the technical heart of turning descriptive fitness into a coherent `Rt` contrast.

### Definition of done

- [ ] Wastewater convolution + likelihood added; per-variant shedding scale included and its identifiability checked.
- [ ] Staged SEIR implemented; conservation still holds (re-run the Phase 0 drift check).
- [ ] `k=1` vs `k=5` comparison quantified.

### If it breaks

- Adding stages makes the solver slow or unstable → the system is now **stiff**; switch the diffrax solver from `Tsit5()` to `Kvaerno5()` (a stiff solver) and keep the `PIDController`.
- Wastewater term dominates the loss → rescale it, or move to the adaptive loss-weighting described in Phase 4.

---

## Phase 4 — Turn point estimates into calibrated uncertainty

**Goal:** stop reporting single best-guess curves and start reporting **credible intervals** — "fitness is probably 0.4, plausibly 0.3 to 0.5" — that are honest about what the data does and doesn't pin down (remember the Phase 1.3 tail).

**Plain-English idea:** so far we found the *one* parameter setting that fits best. Bayesian inference instead finds the *whole cloud* of settings consistent with the data — the **posterior**. Wide cloud = uncertain; narrow = confident. We get this with **NumPyro**, which takes your model plus priors and explores the posterior for you.

### Do this — 4.1 Install and write the probabilistic model

```
> pip install numpyro
```

NumPyro wants your model expressed as "priors + likelihood." Sketch:

```python
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, autoguide

def model(cases_obs, B_counts, n_seq):
    # priors: our beliefs before seeing data
    s_level = numpyro.sample("s_level", dist.Normal(0.0, 0.5))     # baseline fitness
    s_drift = numpyro.sample("s_drift", dist.Normal(0.0, 0.2))     # how much it moves
    report  = numpyro.sample("report", dist.Beta(2, 5))            # reporting fraction
    # run the ODE with these sampled parameters -> predicted incidence + B share
    inc_pred, freqB_pred = simulate_bayes(s_level, s_drift)
    # likelihoods: connect predictions to data
    numpyro.sample("cases", dist.NegativeBinomial2(report*inc_pred, 10), obs=cases_obs)
    numpyro.sample("seqs",  dist.Binomial(n_seq, freqB_pred),           obs=B_counts)
```

### Do this — 4.2 Fit, cheap first then expensive

Uncertainty methods, in increasing cost and fidelity — do them in this order:

1. **Deep ensemble (cheapest):** just run your Phase 2 optimizer ~20 times from different random starts; the spread of answers is a rough uncertainty. No new code.
2. **SVI (medium):** NumPyro's variational inference — fast, approximate. Good for iterating.
   ```python
   guide = autoguide.AutoNormal(model)
   svi = SVI(model, guide, numpyro.optim.Adam(1e-3), Trace_ELBO())
   result = svi.run(jax.random.PRNGKey(0), 5000, cases_obs, B_counts, 40)
   ```
3. **NUTS/HMC (gold standard, slow):** the most accurate posterior; the known best choice for these models but expensive because it runs the ODE many times.
   ```python
   mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=500)
   mcmc.run(jax.random.PRNGKey(0), cases_obs, B_counts, 40)
   mcmc.print_summary()
   ```

Start with SVI to debug (minutes), switch to NUTS for final results (hours). Reduce data length while debugging NUTS.

### Do this — 4.3 Check the uncertainty is honest

Do not assume the intervals are trustworthy — **test calibration** with the coverage experiment from Phase 2.3 (generate many datasets from known truth, fit each, count how often the 95% interval contains the truth; you want ≈ 95%). A method whose intervals are too narrow is worse than useless because it looks confident while being wrong. Report calibration explicitly.

### Why it matters

This restores the exact thing the plain-neural approach threw away, and it's the difference between "the estimate crossed R=1" and "the estimate crossed R=1 but the interval still includes R<1." Public-health decisions hinge on that distinction.

### Definition of done

- [ ] Model expressed in NumPyro with priors + likelihoods.
- [ ] SVI runs; NUTS runs on at least a short series.
- [ ] Posterior credible intervals produced for `Rt(t)`, fitness `s(t)`, and their contrasts.
- [ ] Calibration checked and reported; you know whether your intervals are trustworthy.

### If it breaks

- NUTS is impossibly slow → shorten the series, reduce ODE stages, use SVI for iteration, and only run full NUTS once at the end. Ensure the ODE solve is JIT-compiled.
- "divergences" reported → tighten priors, or reparameterize the random walk (non-centered form: sample standard-normal steps, then scale).
- Posterior is absurdly wide on some quantity → that quantity genuinely isn't identified by your data (revisit Phase 1.4); this is information, not failure.

---

## Phase 5 — Apply it to a real epidemic

**Goal:** run the finished model on a documented real variant takeover and check it behaves.

**Plain-English idea:** swap synthetic data for real case counts, real sequence counts, and real wastewater, then repeat Phase 4. The hard part is not the model — it's cleaning and aligning three messy real datasets.

### Do this — 5.1 Get the three data streams

- **Cases:** national/regional public health dashboards, or curated repositories (e.g. Our World in Data). You want daily or weekly case counts for one region.
- **Sequences:** variant counts over time come from **GISAID** or **Nextstrain** (GISAID requires a free registered account and has data-use terms — read them). You want, per week, how many sequenced samples were each variant.
- **Wastewater:** open portals such as national wastewater surveillance programs publish RNA concentrations by site. Optional but valuable.

Pick a well-studied transition (for example, an Omicron sub-lineage displacing its predecessor) so you have ground truth to sanity-check against.

### Do this — 5.2 Format to the model's inputs

Get everything onto one weekly timeline for one region: a table with columns `week, cases, n_sequenced, variantB_count, ww_concentration`. Align dates carefully (off-by-one-week errors are the most common real-data bug). Handle missing weeks explicitly (NumPyro can skip missing observations).

### Do this — 5.3 Fit and validate

Run the Phase 4 model on the real table. Then three checks:

1. **Early-warning:** does the model flag variant B's fitness advantage (posterior for `s` clearly above 0) *before* B became the majority? The literature reports genomic signals appearing weeks ahead of dominance — you're looking for the same lead time.
2. **Cross-region consistency:** fit two or three regions. The *direction* of the fitness signal should agree everywhere; the *magnitude* may differ with local immunity (expected, and itself a finding).
3. **Posterior predictive check:** simulate data from the fitted posterior and overlay it on the real data — they should look like siblings.

### Why it matters

Everything before this is scaffolding. This is the result.

### Definition of done

- [ ] Three real streams downloaded, cleaned, aligned to one weekly table.
- [ ] Model fit with uncertainty on real data.
- [ ] Early-warning lead time measured; cross-region direction consistent; posterior predictive looks like the real data.

### If it breaks

- Fit is terrible on real data but fine on synthetic → real biases you didn't simulate (reporting changes, sequencing bias, holidays). Add/relax the relevant observation-model parameters.
- GISAID access blocked → use Nextstrain's curated open metadata counts as a substitute for a first pass.

---

## Phase 6 (stretch) — Attempt the virulence coupling, honestly

**Goal:** try to estimate whether transmissibility and severity are linked — and be willing to conclude "the data can't tell."

**Plain-English idea:** severity (hospitalizations, deaths) is a *different* measurement, delayed and rescaled from infections. Only this stream carries information about how *harmful* a variant is. Without it, any "virulence" parameter is invented.

### Do this — 6.1 Add a severity stream

Add variant-stratified hospitalizations or deaths. Model them as a **lagged, scaled** copy of each variant's infections: `severe_v(t) = IHR_v * (infections_v convolved with an onset-to-outcome delay)`, with a variant-specific severity ratio `IHR_v`. Add a count likelihood (negative binomial) for the observed severe counts.

### Do this — 6.2 Test the trade-off, don't assume it

Introduce a parameter linking transmissibility and severity (e.g. severity as a function of fitness). Then **check whether the data identifies it** with the Phase 1.4 / Phase 4 tools. Expect one of two honest outcomes:

- The severity data does pin down `IHR_v` per variant, and you can report whether fitter variants were more or less severe **in this dataset** (note: for SARS-CoV-2, Omicron was *more* transmissible and *less* intrinsically severe — the opposite of the classic trade-off, so don't hard-code the classic shape).
- The coupling parameter's posterior just returns its prior → the data can't identify it. Report that. A clean null is a real result.

### Definition of done

- [ ] Severity stream added with its own delay + likelihood.
- [ ] Coupling parameter included and its identifiability tested.
- [ ] You can state, with uncertainty, either the estimated relationship or that it's unidentified.

---

## Appendix A — Plain-English glossary

- **Compartment / bucket:** a group of people in the same disease state (Susceptible, Exposed, Infected, Recovered).
- **ODE (ordinary differential equation):** a rule for how something changes moment to moment; the solver turns the rule into a trajectory over time.
- **Rt (effective reproduction number):** average onward infections per case at time `t`; >1 grows, <1 shrinks.
- **Growth rate r:** daily rate of increase/decrease of cases; related to `Rt` through the generation interval.
- **Generation interval:** time from one infection to the next it causes; its distribution controls the r↔Rt conversion.
- **Variant fitness / growth advantage / selective coefficient:** how much faster one variant grows than another; read from the variant-share S-curve.
- **UDE (universal differential equation):** an ODE with a neural network standing in for an unknown term (here, time-varying fitness or transmission).
- **PINN:** an alternative where the physics is enforced as a training penalty rather than by a solver; the project uses UDE as primary and PINN as a benchmark.
- **Likelihood:** how probable the observed data is under the model; fitting = making the data probable.
- **Negative binomial / multinomial:** count distributions for cases / for category labels (variants).
- **Posterior / Bayesian:** the full range of parameter values consistent with the data, not just the single best one.
- **Calibration / coverage:** whether your 95% intervals actually contain the truth 95% of the time.
- **Identifiability:** whether the data can, even in principle, separate two parameters; the project's central risk is separating intrinsic transmissibility from immune escape.
- **Stiff ODE:** an equation with fast and slow parts together; needs a special solver (`Kvaerno5`) or it becomes slow/unstable.

## Appendix B — Universal troubleshooting

- **NaNs in the loss:** predictions left a valid range. Clip probabilities to `[1e-6, 1-1e-6]`; keep incidence positive with a `softplus`; lower the learning rate.
- **Solver `max_steps` exceeded:** raise `max_steps`, loosen `rtol/atol`, or switch to a stiff solver.
- **Everything is slow:** make sure the expensive function is wrapped in `jax.jit`; shorten the time series while developing; only run full NUTS at the very end.
- **Two learned functions trade off against each other:** the identifiability problem. Give the fitness function less flexibility than the transmission function, add a smoothness penalty, or bring in an extra data stream (serology). This recurs in every phase — it is the science, not a nuisance.

## Appendix C — Suggested order of attack

1. Phases 0–1 until the identifiability experiment genuinely makes sense to you. Do not skip 1.4.
2. Phase 2 coverage comparison — this validates the whole premise; if it fails, stop and rethink.
3. Phases 3–4 to make it mechanistic and honest about uncertainty.
4. Phase 5 for the real result.
5. Phase 6 only if Phase 5 is clean; treat a null result as publishable.

The minimum publishable unit is Phases 0–2 (the identifiability map plus the calibration win). Everything after strengthens it.
