import jax, jax.numpy as jnp, diffrax, optax
import numpy as np
from scipy.integrate import solve_ivp

N, gamma, beta1 = 1_000_000, 1/5, 0.30
true_s = 0.4                        
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

# 2) the SAME model, but now `s` is unknown
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

# 3) loss minimizing.
def loss(s):
    return jnp.mean((simulate(s) - freqB_obs)**2)

s = jnp.array(0.0)                         #WRONG starting guess
opt = optax.adam(0.05); state = opt.init(s)
step = jax.jit(jax.value_and_grad(loss))  
for i in range(200):
    l, g = step(s)
    updates, state = opt.update(g, state)
    s = optax.apply_updates(s, updates)

print(f"true s = {true_s} | recovered s = {float(s):.4f} | loss = {float(l):.1e}")
