import jax, jax.numpy as jnp, diffrax, optax, numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Identifiability test: the TRUTH is an immune-escape model (two-phase-simulate2)
# in which variants A and B are equally transmissible (beta1 == beta2, so the
# intrinsic fitness advantage is exactly s(t) = 0).  Variant B's rising share is
# driven purely by reinfection of an A-recovered "partial-susceptible" pool S_b.
#
# The FITTING model is the neural UDE, which can only explain a rising B share
# through an intrinsic time-varying fitness s(t)  (b2 = beta1*exp(s(t))).
# The question: does the network invent a positive, rising s(t) even though the
# true intrinsic advantage is zero?  If so, fitness and immune escape are
# confounded through the freqB signal.
# ---------------------------------------------------------------------------

N, gamma, beta1 = 1_000_000, 1/5, 0.30
T = 200                      # days
beta2 = 0.30                 # variant B: SAME transmissibility as A (no fitness edge)
par_sus = 0.7                # fraction of A-recovered who become B-susceptible again

# --- TRUTH: 5-bucket immune-escape model (from two-phase-simulate2) ----------
def truth(t, y):
    S, S_b, I1, I2, R = y
    infA     = beta1 * I1 * S   / N          # A infects fully susceptibles
    inf_susB = beta2 * I2 * S_b / N          # B reinfects A-recovered pool (escape)
    infB     = beta2 * I2 * S   / N          # B infects fully susceptibles
    dS   = -(infA + infB)
    dS_b = par_sus * gamma * I1 - inf_susB   # A-recovered flow into B's escape pool
    dI1  =  infA - gamma * I1
    dI2  = (infB + inf_susB) - gamma * I2
    dR   = (1 - par_sus) * gamma * I1 + gamma * I2
    return [dS, dS_b, dI1, dI2, dR]

y0 = [N - 130, 20, 100, 10, 0]
sol = solve_ivp(truth, (0, T), y0, t_eval=np.arange(0, T + 1),
                rtol=1e-9, atol=1e-9)
S, S_b, I1, I2, R = sol.y

incA = beta1 * I1 * S / N
incB = beta2 * I2 * (S_b + S) / N            # B incidence from both susceptible pools
freqB_obs = jnp.array(incB / (incA + incB))  # the only signal the network sees

# --- FITTING model: neural fitness s(t) inside a 4-bucket SIR ----------------
# input = time, output = fitness s(t); the S_b escape pool is UNMODELLED (latent).
def init_mlp(key, w=16):
    k1, k2, k3 = jax.random.split(key, 3)
    return dict(W1=jax.random.normal(k1, (1, w)) * 0.3, b1=jnp.zeros(w),
                W2=jax.random.normal(k2, (w, w)) * 0.3, b2=jnp.zeros(w),
                W3=jax.random.normal(k3, (w, 1)) * 0.3, b3=jnp.zeros(1))
def mlp(p, t):
    x = jnp.array([t / float(T)])            # scale time into [0,1]
    x = jnp.tanh(x @ p['W1'] + p['b1']); x = jnp.tanh(x @ p['W2'] + p['b2'])
    return (x @ p['W3'] + p['b3'])[0]

def ode(t, y, p):
    s = mlp(p, t); b2 = beta1 * jnp.exp(s)
    S, I1, I2, R = y; a, b = beta1 * I1 * S / N, b2 * I2 * S / N
    return jnp.array([-(a + b), a - gamma * I1, b - gamma * I2, gamma * (I1 + I2)])
term, ts = diffrax.ODETerm(ode), jnp.arange(0., T + 1.)
def simulate(p):
    sol = diffrax.diffeqsolve(term, diffrax.Tsit5(), 0., float(T), 0.1,
        jnp.array([N - 110., 100., 10., 0.]), args=p,   # no S_b: escape is hidden
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6),
        max_steps=400000)
    S, I1, I2, R = sol.ys.T
    b2 = beta1 * jnp.exp(jax.vmap(lambda t: mlp(p, t))(ts))
    return (b2 * I2 * S / N) / (beta1 * I1 * S / N + b2 * I2 * S / N)
def loss(p): return jnp.mean((simulate(p) - freqB_obs) ** 2)

p = init_mlp(jax.random.PRNGKey(0)); opt = optax.adam(3e-3); state = opt.init(p)
step = jax.jit(jax.value_and_grad(loss))
for i in range(1500):
    l, g = step(p); u, state = opt.update(g, state); p = optax.apply_updates(p, u)

# --- report: recovered s(t) vs the TRUE intrinsic advantage (which is 0) -----
srec = jax.vmap(lambda t: mlp(p, t))(ts)
print(f"final loss on freqB: {float(loss(p)):.2e}")
for d in (20, 60, 100, 150):
    print(f"day {d}: network s={float(srec[d]):+.3f}   true intrinsic s=0.000")

fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax[0].plot(ts, freqB_obs, 'k', label='truth (immune escape)')
ax[0].plot(ts, simulate(p), '--', label='neural fit')
ax[0].set_ylabel('variant B share'); ax[0].legend()
ax[0].set_title('Same freqB signal, two mechanisms')
ax[1].plot(ts, srec, label='network-recovered s(t)')
ax[1].axhline(0, color='r', ls=':', label='true intrinsic s = 0')
ax[1].set_ylabel('fitness s(t)'); ax[1].set_xlabel('day'); ax[1].legend()
ax[1].set_title('Network invents fitness to mimic immune escape')
plt.tight_layout(); plt.show()
