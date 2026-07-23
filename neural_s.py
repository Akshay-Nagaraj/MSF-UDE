import jax, jax.numpy as jnp, diffrax, optax, numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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

# --- plots: true vs neural fit ------------------------------------------------
freqB_fit = np.asarray(simulate(p))
srec_np   = np.asarray(srec)
fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax[0].plot(ts, np.asarray(freqB_obs), 'k',  label='true (data)')
ax[0].plot(ts, freqB_fit, '--', color='tab:blue', label='neural fit')
ax[0].set_ylabel('variant B share'); ax[0].legend()
ax[0].set_title('Signal fit — freqB')
ax[1].plot(ts, strue, 'k',  label='true s(t)')
ax[1].plot(ts, srec_np, '--', color='tab:red', label='recovered s(t)')
ax[1].set_ylabel('fitness s(t)'); ax[1].set_xlabel('day'); ax[1].legend()
ax[1].set_title('Latent fitness — true vs recovered')
plt.tight_layout()
plt.savefig('neural_s_fit.png', dpi=130)
plt.show()