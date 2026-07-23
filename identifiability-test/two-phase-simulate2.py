import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# --- fixed numbers describing the disease and population ---
N = 1_000_000        # total people
gamma = 1/5          # recovery rate: 1/(5-day infectious period)
beta1 = 0.30         # transmission rate of variant A (older)
beta2 = 0.30       # transmission rate of variant B (same as A, additional suscpetible pool)

# --- the model rules: how the buckets change each day ---
def two_strain_sir(t, y):
    S, S_b, I1, I2, R = y                 # unpack the 5 buckets (S_b is the additional susceptible pool for variant B)
    par_sus = 0.7 #partial suscpeitbility for variant B of A recovered
    infA = beta1 * I1 * S / N        # new A infections per day
    inf_susB = beta2 * I2 * S_b/ N  
    infB = beta2 * I2 * S / N      # new B infections per day
    dS  = -(infA + infB)
    dS_b = par_sus * gamma * I1 - inf_susB               # susceptibles leave as they get infected
    dI1 =  infA - gamma * I1         # A: gain infections, lose recoveries
    dI2 =  (infB + inf_susB)- gamma * I2         # B: same
    dR  = (1-par_sus)*gamma*I1 +  gamma * (I2)         # everyone recovers eventually
    return [dS, dS_b, dI1, dI2, dR]
# --- starting point: almost everyone susceptible, mostly A, a seed of B ---
y0 = [N - 130, 20, 100, 10, 0]
days = np.arange(0, 201)
sol = solve_ivp(two_strain_sir, (0, 200), y0, t_eval=days, rtol=1e-8, atol=1e-8)
S, S_b, I1, I2, R = sol.y

# --- derive the two signals ---
incA = beta1 * I1 * S / N
incB = beta2 * I2 * (S_b + S) / N
total_incidence = incA + incB                 # SIGNAL 1: size of epidemic
freqB = incB / (incA + incB)                  # SIGNAL 2: variant B's share

# --- sanity check: nobody should appear or vanish ---
total_people = S + I1 + I2 + R + S_b
print("population drift (should be ~0):", total_people.max() - total_people.min())
print("variant B share on days 0/50/100/150:",
      *[round(freqB[d], 3) for d in (0, 50, 100, 150)])

# --- plot the two signals ---
fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
ax[0].plot(days, total_incidence); ax[0].set_ylabel("new infections / day")
ax[0].set_title("Signal 1 — size of the whole epidemic")
ax[1].plot(days, freqB); ax[1].set_ylabel("variant B share")
ax[1].set_title("Signal 2 — which variant is winning"); ax[1].set_xlabel("day")
plt.tight_layout(); plt.show()
