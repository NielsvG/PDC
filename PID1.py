import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

# parameters
eta = 0.65 
Cs1 = 7000 #kJ/C
Cp_elec = 3.24 #kJ/(kgC)
rho_elec = 1320 #kg/m^3
delta_Hr = 286 #kJ/mol H2O
V1_gas = 0.1 #m^3
V2_gas = 0.1 #m^3
p1 = 0.034/440 #kg/kJ
p2 = 0.002/440 #kg/kJ
p3 = 0.016/440 #kg/kJ
p4 = 0.35

# Physical quantities
mwH2 = 2.016/1000 #kg/mol
mwO2 = 32./1000 #kg/mol
R = 8.314

def Ept(t):
    if t<=(60*60):
        return 100
    elif t>(60*60) and t<(70*60):
        return 100 + 400*(t-(60*60))/(10*60)
    elif t >= (70*60) and t <= (130*60):
        return 500
    elif t > (130*60) and t<(140*60):
        return  500 - 400*(t-(130*60))/(10*60)
    elif t >= (140*60):
        return 100
    
def F3_H2t(EP):
    # Equation 6
    return p2*EP
    
def F4_O2t(EP):
    # Equation 7
    return p3*EP

def p_set(t):
    convH2 = 100000*(mwH2/(R*353)) # bar to hydrogen density
    convO2 = 100000*(mwO2/(R*353)) # bar to oxygen density
    if t<(5*60):
        p_setH2 = 2*convH2 # 2 bar
        p_setO2 = 2*convO2 # 2 bar
    else:
        p_setH2 = 2.50*convH2
        p_setO2 = 2.50*convO2
    return [p_setH2, p_setO2]

Nt = 500
tf = 100
TCA_SP = 273.15+70
F1_elec = 2
F2_elec = 2
F5SS = 0.000454545454
F6SS = 0.003636363636

t_arr = np.linspace(0,tf*60,Nt)
pH2_Kp = -10 #m^-3
pO2_Kp = -10 #m^-3
lmb = 100 #just chose, higher for slower systems. Probably in seconds.
pH2_tau_i = 2*lmb
pO2_tau_i= 2*lmb
Kc_pH2 = 2/(pH2_Kp*lmb) # m^3/s
# Kp_pH2 = 2/(-20*lmb)
# Ki_pH2 = 2/(-10*lmb)
Kc_pO2 = 2/(pO2_Kp*lmb) # m^3/s

def eq(u,t): #u = [PH2, PO2, I1, I2, T1, T3]
    pH2, pO2, I1, I2, T1, T3 = u
    error1 = p_set(0)[0]-pH2 # error H2
    error2 = p_set(t)[1]-pO2 # error O2
    C1 = Kc_pH2*(error1+1/pH2_tau_i*I1) # m^3/s * error in density
    # C1 = Kp_pH2*error1 + Ki_pH2*(1/pH2_tau_i*I1)
    C2 = Kc_pO2*(error2+1/pO2_tau_i*I2) # m^3/s 
    F5 = F5SS + C1
    F6 = F6SS + C2
    dpH2dt = (F3_H2t(Ept(t))-F5)/V1_gas
    # if t < 50*60:
    #     F3 = F3_H2t(Ept(0))
    #     dpH2dt = (F3-F5)/V1_gas # gas setpoint increase good, density increase bad
    # else: 
    #     F3 = F3_H2t(150)
    #     dpH2dt = (F3-F5)/V1_gas
    dpO2dt = ((F4_O2t(Ept(t)))-F6)/V2_gas 
    dI1dt = error1
    dI2dt = error2
    dT1dt = (TCA_SP-T1)/60 # T1
    dT3dt = (F1_elec+F2_elec)*Cp_elec*(T1-T3)/Cs1 + p4*Ept(t)/Cs1 # T3
    if t>= 59*60 and t < 75*60:
        print(f'At t: {t:.2f} s, SP = {p_set(0)[0]:.5f}, pH2 = {pH2:.5f}, error = {error1:.10f}, Total error = {I1:.10f}, C = {C1*1000:.10f}, F3 = {F3_H2t(Ept(t)):.10f}, F5 = {F5:.10f}, dpH2dt = {dpH2dt:.10f}')
        print(-(dpH2dt)*V1_gas+p2*Ept(t))

    return np.array([dpH2dt,dpO2dt,dI1dt,dI2dt,dT1dt,dT3dt])

#res0 = [p_set(0)[0],p_set(0)[1],SS.F5*pH2_tau_i/Kc_pH2,SS.F6*pO2_tau_i/Kc_pH2,SS.T1,SS.T3]
res0 = [p_set(0)[0],p_set(0)[1],0,0,70,72.7]

res = scipy.integrate.odeint(eq, res0, t_arr)

# fig1, ax1 = plt.subplots()
# fig1.set_size_inches(16,8)
# fig2, ax2 = plt.subplots()
# fig2.set_size_inches(16,8)
# fig3, ax3 = plt.subplots()
# fig3.set_size_inches(16,8)
# fig4, ax4 = plt.subplots()
# fig4.set_size_inches(16,8)
# fig5, ax5 = plt.subplots()
# fig5.set_size_inches(16,8)
# fig6, ax6 = plt.subplots()
# fig6.set_size_inches(16,8)
# fig7, ax7 = plt.subplots()
# fig7.set_size_inches(16,8)
fig8, ax8 = plt.subplots()
fig8.set_size_inches(16,8)

EP_array = np.zeros(t_arr.size)
for i in range(0,t_arr.size):
    EP_array[i] = Ept(t_arr[i])
# ax1.plot(t_arr/60, EP_array, label = 'EP')
# ax1.set_xlabel("Time (min)")
# ax1.set_ylabel("EP")
# ax1.set_title("EP")
# ax1.legend()
# plt.show(block=False)

# ax2.plot(t_arr/60, res[:,4]-273.15, label = 'T1')
# ax2.plot(t_arr/60, res[:,5]-273.15, label = 'T3')
# ax2.set_title("Temperature of TC1 and TC3 over time")
# ax2.set_xlabel("Time (min)")
# ax2.set_ylabel("Temperature (K)")
# ax2.legend()
# plt.show(block=False)

pH2_array = np.zeros(t_arr.size)
pO2_array = np.zeros(t_arr.size)
print(t_arr.size)
for i in range(0,t_arr.size):
    pH2_array[i], pO2_array[i] = p_set(t_arr[i])
# ax3.plot(t_arr/60, pH2_array*(R*353/mwH2)/100000, label = 'PC1.SP')
# ax3.plot(t_arr/60, res[:,0]*(R*353/mwH2)/100000, label = 'PC1.PV')
# ax3.set_xlabel("Time (min)")
# ax3.set_ylabel("Hydrogen pressure (bar)")
# ax3.set_title("Hydrogen pressure over Time")
# ax3.legend()
# plt.show(block=False)
# print(str(np.min(res[:,0]))+ ' ' + str(np.max(res[:,0])))

# ax4.plot(t_arr/60, pO2_array*(R*353/mwO2), label = 'PC2.SP')
# ax4.plot(t_arr/60, res[:,1]*(R*353/mwO2), label = 'PC2.PV')
# ax4.set_xlabel("Time (min)")
# ax4.set_ylabel("Oxygen pressure (Pa)")
# ax4.set_title("Oxygen pressure over time")
# ax4.legend()
# plt.show(block=False)

F5_array = np.zeros(t_arr.size-1)
F6_array = np.zeros(t_arr.size-1)
FI_array = np.zeros(t_arr.size)
for i in range(0,t_arr.size-1):
    #F5_array[i] = SS.F5 + KcH2*(res[i+1,2]+res[i,2]/taui)
    # if t_arr[i] < 50*60:
    #     F5_array[i] = -(res[:,0][i+1]-res[:,0][i])*V1_gas+p2*100 # Remember to make this EPt
    # else: 
    #     F5_array[i] = -(res[:,0][i+1]-res[:,0][i])*V1_gas+p2*90
    F5_array[i] = -(res[:,0][i+1]-res[:,0][i])*V1_gas+p2*Ept(t_arr[i])
    F6_array[i] = -(res[:,1][i+1]-res[:,1][i])*V2_gas+p3*Ept(t_arr[i])
    FI_array[i] = res[:,2][i+1]-res[:,2][i]

# ax5.plot(t_arr[0:-1]/60, F5_array, label = 'F5')
# ax5.plot(t_arr/60, F3_H2t(EP_array), label = 'F3_H2')
# #ax5.plot(t_arr/60, FI_array*0.001, label = 'dI1dt')
# ax5.set_xlabel("Time (min)")
# ax5.set_ylabel("Flow (kg/s)")
# ax5.set_title("Flow in F5 and F3_H2 over time")
# ax5.legend()
# plt.show(block=False)

# ax6.plot(t_arr[0:-1]/60, F6_array, label = 'F6')
# ax6.plot(t_arr/60, F4_O2t(EP_array), label = 'F4_O2')
# ax6.set_xlabel("Time (min)")
# ax6.set_ylabel("Flow (kg/s)")
# ax6.set_title("Flow in F6 and F4_O2 over time")
# ax6.legend()
# plt.show(block=False)

dIdt = np.ones(t_arr.size-1)
for i in range(0, t_arr.size-1):
    dIdt[i] = res[i+1,2] - res[i,2]
# ax7.plot(t_arr/60, res[:,2], label = 'Total error in rhoH2')
# ax7.plot(t_arr[:-1]/60, dIdt, label = 'Error in rhoH2')
# ax7.set_xlabel("Time (min)")
# ax7.set_ylabel("Flow (kg/s)")
# ax7.set_title("Error")
# plt.show(block=False)

ax8.plot(t_arr/60, res[:,2]/abs(np.min(res[:,2])), label = 'Total error scaled')
ax8.plot(t_arr[:-1]/60, dIdt*10/abs(np.min(res[:,2])), label = 'Error scaled')
ax8.plot(t_arr[:-1]/60, F5_array/(np.max(F5_array)), label='F5 scaled')
ax8.plot(t_arr/60, F3_H2t(EP_array)/np.max(F5_array), label = 'F3_H2 scaled')
ax8.plot(t_arr/60, pH2_array/np.max(res[:,0]), label = 'PC1.SP scaled')
ax8.plot(t_arr/60, res[:,0]/np.max(res[:,0]), label = 'PC1.PV scaled')
ax8.legend()
plt.show()

print(f"\nNt: {t_arr.size:.2f}, tf: {t_arr[-1]/60:.2f} min, lmb: {lmb:.2f}")
print(f"max F5: {np.max(F5_array):.5f}, min F5: {np.min(F5_array):.5f}, max change: ")