# CSTR3 (closed loop with step)
# 28 June 2024
# Bijoy Bera

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# Define parameters
Ea = 60000 # Activation energy [J/mol]
R = 8.314 # Gas constant [J/(molK)]
k0 = 1e7 # Arrhenius rate constant [1/s]

V = 0.5 # Reactor volume [m3]
rho = 900.0 # Density [kg/m3]
Cp = 2500 # Heat capacity [J/kg/K]
dHr = 10000 # Reaction enthalpy [J/mol]
hA = 520 # Heat transfer [J/(Ks]
F = 0.0005 # Flowrate [m3/s]
CA0 = 18000 # Inlet feed concentration [mol/m3]
T0 = 303.0 # Inlet feed temperature [K]
KC = 10.0 # Gain PI controller
Ti = 1000 # Integral time PI controller
Ng = 1801 # Number of points for time grid

# Define Arrhenius rate expression
def k(T):
    return k0*np.exp(-Ea/(R*T))

# Define step on SP temperature controller
def SP(t):
    if t <= 3600:
        a = 345
    else:
        a = 340
    return a

# Define differential equations and PI controller
def deriv(t, X):
    CA,T,IE = X
    dCAdt = (F/V)*CA0 - (F/V)*CA - k(T)*CA
    Error = SP(t) - T # Error calculation
    TC = KC*(Error + 1/Ti*IE) # PI controller
    dIEdt = Error # Integrate error
    dTdt = (F/V)*T0 - (F/V)*T - hA/(V*rho*Cp)*(T - TC) + dHr/(rho*Cp)*k(T)*CA
    return [dCAdt, dTdt, dIEdt]

# Solve
IC = [2565, 340, 29300]
t_initial = 0.0
t_final = 7200
t = np.linspace(t_initial, t_final, Ng)
soln = solve_ivp(deriv, [t_initial, t_final], IC, t_eval=t)

# Construct reference (SP) and output (TC)
Ref = np.zeros(Ng)
for i in range(Ng):
    Ref[i]= SP(t[i])
Output = np.zeros(Ng)
Output = KC*((Ref - soln.y[1]) + 1/Ti*soln.y[2])

# Plot results
fig, axs = plt.subplots(2)
fig.suptitle('CSTR3 results')
axs[0].plot(soln.t, soln.y[0], label = 'CA', color = 'blue')
axs[0].set_ylabel('CA [mol/m3]')
axs[0].set_ylim(1000, 3000)
axs[0].legend(loc = 4)
axs[1].plot(soln.t, soln.y[1], label = 'T', color = 'red')
axs[1].plot(soln.t, Ref, '--', label = 'Ref', color = 'red')
axs[1].set_ylabel('T [K]')
axs[1].set_ylim(335, 355)
axs[1].set_xlabel('Time [s]')
axs[1].legend(loc = 1)
plt.show()