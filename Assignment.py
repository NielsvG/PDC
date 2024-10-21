import numpy as np
import numpy.typing as npt
import scipy.integrate
import matplotlib.pyplot as plt
import sympy as sp

# variables     Min     Max     Units
EP = 100       #100     500     kW
F1_elec = 2    #2       2       kg/s
F2_elec = 2    #2       2       kg/s
TCA_SP = 70    #70      70      C

# unknown variables
#F3_elec 
#F3_H2
#F_OH
#F4_elec
#F4_O2
#F5
#F6
#F7_elec
#F8_elec
#F9
#rho_H2
#rho_O2
#T1
#T2
#T3
#T4

class SSt:
    def __init__(self,EP=100,F1_elec=2,F2_elec=2,TCA_SP=70):
        self.EP = EP
        self.F1_elec = F1_elec
        self.F2_elec = F2_elec
        self.TCA_SP = 273.15 + TCA_SP

        # parameters
        self.eta = 0.65 
        self.Cs1 = 7000 #kJ/C
        self.Cp_elec = 3.24 #kJ/(kgC)
        self.rho_elec = 1320 #kg/m^3
        self.delta_Hr = 286 #kJ/mol H2O
        self.V1_gas = 0.1 #m^3
        self.V2_gas = 0.1 #m^3
        self.p1 = 0.034/440 #kg/kJ
        self.p2 = 0.002/440 #kg/kJ
        self.p3 = 0.016/440 #kg/kJ
        self.p4 = 0.35

        self.calc = 0

    def calculate(self):
        # Assumption 6
        self.F3_elec = self.F1_elec
        self.F4_elec = self.F2_elec

        # Equation 2
        self.T1 = self.TCA_SP

        # Equation 15
        self.T2 = self.T1

        # Equation 9, Steady State.
        self.T3 = self.p4*self.EP/((self.F1_elec+self.F2_elec)*self.Cp_elec)+self.T1

        # Assumption 5
        self.T4 = self.T3
        
        # Equation 5
        self.F_OH = self.p1*self.EP

        # Equation 6
        self.F3_H2 = self.p2*self.EP

        # Equation 7
        self.F4_O2 = self.p3*self.EP

        # Equation 3
        self.F3_elec = self.F1_elec - self.F3_H2 - self.F_OH

        # Equation 4
        self.F4_elec = self.F2_elec - self.F4_O2 + self.F_OH

        # Steady state so V1_gas*d(rho_h2)/dt = F3_H2 - F5 (Equation 10) becomes 0 = F3_H2 - F5
        self.F5 = self.F3_H2

        # Steady state so V2_gas*d(rho_O2)/dt = F4_O2 - F6 (Equation 12) becomes 0 = F4_O2 - F6
        self.F6 = self.F4_O2

        # 0 = F4_elec - F8_elec (Equation 13)
        self.F8_elec = self.F4_elec

        # 0 = F3_elec + F9 - F7_elec (Equation 11)
        #self.F7_elec = self.F3_elec + self.F9
        # F1_elec + F2_elec = F7_elec + F8_elec
        self.F7_elec = self.F1_elec + self.F2_elec - self.F8_elec

        # 0 = F9 - F5 - F6 (Compound balance)
        #self.F9 = self.F5 + self.F6
        self.F9 = self.F7_elec - self.F3_elec

        #sanity check
        if (self.F9 - (self.F5 + self.F6)) > 0.0001:
            print(f"Error: Total mass balance incorrect\nF9 ({self.F9}) = F5 ({self.F5}) + F6 ({self.F6}) = {self.F5+self.F6}")
        if (self.F7_elec + self.F8_elec - (self.F1_elec + self.F2_elec)) > 0.0001:
            print(f"Error: B0 mass balance incorrect\nF7_elec ({self.F7_elec}) + F8 ({self.F8_elec}) = F1_elec ({self.F1_elec}) + F2_elec ({self.F2_elec})")
        

        self.calc = 1

    def __str__(self):
        if self.calc==1:
            return f'''
Steady State Alkaline Electrolyser initialized with the following variables and parameters:\n\n
Variables   Min     Max     Units
EP={self.EP}      100     500     kW
F1_elec={self.F1_elec}   2       2       kg/s
F2_elec={self.F2_elec}   2       2       kg/s
TCA_SP={self.TCA_SP-273.15} 70      70      C

Results  
F3_elec = {self.F3_elec:.3g}
F3_H2 = {self.F3_H2:.3g}
F_OH = {self.F_OH:.3g}
F4_elec = {self.F4_elec:.3g}
F4_O2 = {self.F4_O2:.3g}
F5 = {self.F5:.3g}
F6 = {self.F6:.3g}
F7_elec = {self.F7_elec:.3g}
F8_elec = {self.F8_elec:.3g}
F9 = {self.F9:.5g}
rho_H2 = unknown
rho_O2 = unknown
T1 = {self.T1-273.15:.3g}
T2 = {self.T2-273.15:.3g}
T3 = {self.T3-273.15:.3g}
T4 = {self.T4-273.15:.3g}
'''
        elif self.calc==0:
            return f'''
Steady State Alkaline Electrolyser initialized with the following variables and parameters:\n\n
Variables   Min     Max     Units
EP={self.EP}      100     500     kW
F1_elec={self.F1_elec}   2       2       kg/s
F2_elec={self.F2_elec}   2       2       kg/s
TCA_SP={self.TCA_SP}   70      70      C

Results not yet calculated.
'''
        
class dynmodel: 
    def __init__(self,EP: float = 100, F1_elec: float = 2,F2_elec: float = 2,TCA_SP: float = 70):
        self.EP = EP
        self.F1_elec = F1_elec
        self.F2_elec = F2_elec
        self.TCA_SP = 273.15 + TCA_SP

        # parameters
        self.eta = 0.65 
        self.Cs1 = 7000 #kJ/C
        self.Cp_elec = 3.24 #kJ/(kgC)
        self.rho_elec = 1320 #kg/m^3
        self.delta_Hr = 286 #kJ/mol H2O
        self.V1_gas = 0.1 #m^3
        self.V2_gas = 0.1 #m^3
        self.p1 = 0.034/440 #kg/kJ
        self.p2 = 0.002/440 #kg/kJ
        self.p3 = 0.016/440 #kg/kJ
        self.p4 = 0.35

        # Physical quantities
        self.mwH2 = 2.016/1000 #kg/mol
        self.mwO2 = 32./1000 #kg/mol
        self.R = 8.314

        # SS values
        self.SS = SSt(EP, F1_elec, F2_elec, TCA_SP)
        self.SS.calculate()

        self.calc = 0
        self.pidr = 0
        self.pidt = 0

    def Ept(self, t):
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
        
    def p_set(self, t):
        convH2 = 100000*(self.mwH2/(self.R*353)) # bar to hydrogen density
        convO2 = 100000*(self.mwO2/(self.R*353)) # bar to oxygen density
        if t<(5*60):
            p_setH2 = 2*convH2 # 2 bar
            p_setO2 = 2*convO2 # 2 bar
        else:
            p_setH2 = 2.50*convH2
            p_setO2 = 2.50*convO2
        return [p_setH2, p_setO2]

    def F3_H2t(self, EP):
        # Equation 6
        return self.p2*EP
        
    def F4_O2t(self, EP):
        # Equation 7
        return self.p3*EP
    
    def F5t(self, t):
        # For step test
        if t < 100*60:
            return self.SS.F5
        else:
            return self.SS.F5 + 1
        
    def F6t(self, t):
        # For step test
        if t < 120*60:
            return self.SS.F6
        else:
            return self.SS.F6 + 1
        
    def TCA_SPt(self, t):
        # For step test
        if t < 100*60:
            return self.TCA_SP
        else:
            return self.TCA_SP + 1

    def calculate(self, Nt: int = 100, tf: float = 200):
        self.t_arr = np.linspace(0,tf*60,Nt)

        # For dynamic calculation
        def eq(u,t): # u = [T1, T3, rho_H2, rho_O2]
            dT1dt = (self.TCA_SP-u[0])/60 # T1, adapted from equation 2 (DT1/dt=(TCA_SP-?)/60)
            dT3dt = ((self.F1_elec+self.F2_elec)*self.Cp_elec*(u[0]-u[1]))/self.Cs1 + (self.p4*self.Ept(t))/self.Cs1 # T3, adapted from equation 8
            drhoH2dt = (self.F3_H2t(self.Ept(t))-self.SS.F5)/self.V1_gas # rho_H2, adapted from equation 10
            drhoO2dt = (self.F4_O2t(self.Ept(t))-self.SS.F6)/self.V2_gas # rho_O2, adapted from equation 12
            return np.array([dT1dt,dT3dt,drhoH2dt,drhoO2dt])
        
        # For step test F3H2 and F4O2 minimum EP=100
        # def eq(u,t): # u = [T1, T3, rho_H2, rho_O2]
        #     dT1dt = (self.TCA_SP-u[0])/60 # T1, adapted from equation 2 (DT1/dt=(TCA_SP-?)/60)
        #     dT3dt = ((self.F1_elec+self.F2_elec)*self.Cp_elec*(u[0]-u[1]))/self.Cs1 + (self.p4*100)/self.Cs1 # T3, adapted from equation 8
        #     drhoH2dt = (self.F3_H2t(100)-self.F5t(t))/self.V1_gas # rho_H2, adapted from equation 10
        #     drhoO2dt = (self.F4_O2t(100)-self.F6t(t))/self.V2_gas # rho_O2, adapted from equation 12
        #     return np.array([dT1dt,dT3dt,drhoH2dt,drhoO2dt])    

        # For step test TCA.SP minimum EP=100
        # def eq(u,t): # u = [T1, T3, rho_H2, rho_O2]
        #     dT1dt = (self.TCA_SPt(t)-u[0])/60 # T1, adapted from equation 2 (DT1/dt=(TCA_SP-?)/60)
        #     dT3dt = ((self.F1_elec+self.F2_elec)*self.Cp_elec*(u[0]-u[1]))/self.Cs1 + (self.p4*self.Ept(0))/self.Cs1 # T3, adapted from equation 8
        #     drhoH2dt = (self.F3_H2t(self.Ept(t))-self.SS.F5)/self.V1_gas # rho_H2, adapted from equation 10
        #     drhoO2dt = (self.F4_O2t(self.Ept(t))-self.SS.F6)/self.V2_gas # rho_O2, adapted from equation 12
        #     return np.array([dT1dt,dT3dt,drhoH2dt,drhoO2dt])        

        res0 = np.array([self.SS.T1,self.SS.T3,0.071,1.1369])
        self.res = scipy.integrate.odeint(eq, res0, self.t_arr)

        self.calc = 1
        self.pidr = 0
        self.pidt = 0
        return self.res
    
    def PID1(self, Nt: int = 10000, tf: float = 200):
        self.t_arr = np.linspace(0,tf*60,Nt)
        self.pH2_Kp = -10 #m^-3
        self.pO2_Kp = -10 #m^-3
        self.lmb = 100 #just chose, higher for slower systems. Probably in seconds.
        self.pH2_tau_i = 2*self.lmb
        self.pO2_tau_i= 2*self.lmb
        self.Kc_pH2 = 2/(self.pH2_Kp*self.lmb) # m^3/s
        self.Kc_pO2 = 2/(self.pO2_Kp*self.lmb) # m^3/s

        def eq(u,t): #u = [PH2, PO2, I1, I2, T1, T3]
            pH2, pO2, I1, I2, T1, T3 = u
            error1 = self.p_set(t)[0]-pH2 # error H2
            error2 = self.p_set(t)[1]-pO2 # error O2
            C1 = self.Kc_pH2*(error1+1/self.pH2_tau_i*I1) # m^3/s * error in density
            C2 = self.Kc_pO2*(error2+1/self.pO2_tau_i*I2) # m^3/s 
            F5 = self.SS.F5 + C1
            F6 = C2
            dpH2dt = (self.F3_H2t(self.Ept(t))-F5)/self.V1_gas
            # if t < 25*60:
            #     F3 = self.F3_H2t(self.Ept(0))
            #     dpH2dt = (F3-F5)/self.V1_gas # gas setpoint increase good, density increase bad
            # else: 
            #     F3 = self.F3_H2t(90)
            #     dpH2dt = (F3-F5)/self.V1_gas
            dpO2dt = ((self.F4_O2t(self.Ept(t)))-F6)/self.V2_gas 
            dI1dt = error1
            dI2dt = error2
            dT1dt = (self.TCA_SP-T1)/60 # T1
            dT3dt = (self.SS.F1_elec+self.SS.F2_elec)*self.Cp_elec*(T1-T3)/self.Cs1 + self.p4*self.Ept(t)/self.Cs1 # T3
            if t>= 60*60-5 and t < 70*60+5:
                print(f'At t: {t:.2f} s, SP = {self.p_set(t)[0]:.5f}, pH2 = {pH2:.10f}, error = {error1:.10f}, Total error = {I1:.10f}, C = {C1:.10f}, F3 = {self.F3_H2t(self.Ept(t)):.10f}, F5 = {F5:.10f} and dpH2dt = {dpH2dt:.10f}')

            return np.array([dpH2dt,dpO2dt,dI1dt,dI2dt,dT1dt,dT3dt])

        #res0 = [self.p_set(0)[0],self.p_set(0)[1],self.SS.F5*self.pH2_tau_i/self.Kc_pH2,self.SS.F6*self.pO2_tau_i/self.Kc_pH2,self.SS.T1,self.SS.T3]
        res0 = [self.p_set(0)[0],self.p_set(0)[1],0,0,self.SS.T1,self.SS.T3]

        self.res = scipy.integrate.odeint(eq, res0, self.t_arr)

        self.pidr = 1
        self.calc = 0
        self.pidt = 0
        return self.res
    
    def T_set(self, t):
        if t<100*60:
            T_setT1 = 80 + 273.15
        else:
            T_setT1 = 78 + 273.15
        return T_setT1
    
    def PID2(self, Nt: int = 1000, tf: float = 200):
        self.t_arr = np.linspace(0,tf*60,Nt)
        
        lmb = 200
        KcT3 = 1
        tau_i = 600
        self.KcT1 = tau_i/(KcT3*lmb)

        def eq3(u,t): #u = [T1, T3, I1]
            T1, T3, I = u
            error = self.T_set(t) - T3
            C = self.SS.T1 + self.KcT1*(error+1/tau_i*I) # TCA_SP
            dT1dt = (C-T1)/60
            dT3dt = (self.SS.F1_elec+self.SS.F2_elec)*self.Cp_elec*(T1-T3)/self.Cs1 + self.p4*self.Ept(t)/self.Cs1
            dIdt = error
            # if t>=0 and t < 245:
            #    print(f'At t: {t:.2f} s, SP = {self.T_set(t):.2f}, T3 = {T3:.2f}, T1 = {T1:.2f}, error = {error:.2f}, Total error = {I:.2f}, C = {C:.2f}, dT1dt = {dT1dt:.2f} and dT3dt = {dT3dt:.2f}')
            return [dT1dt, dT3dt, dIdt]
        
        def eq3_ivp(t,u):
            return eq3(u,t)

        #T0 = np.array([self.T_set(0)-(self.p4*self.Ept(0))/((self.F1_elec+self.F2_elec)*self.Cp_elec), self.T_set(0), 0])
        T0 = np.array([self.SS.T1, self.SS.T3, self.T_set(0)-self.SS.T3])
        self.res = scipy.integrate.odeint(eq3, T0, self.t_arr)
        #self.resivp = scipy.integrate.solve_ivp(eq3_ivp, [0,tf*60], T0, t_eval=self.t_arr)

        self.pidt = 1
        self.calc = 0
        self.pidr = 0
        return self.res
    
    def plot_calcgraphs(self, Nt: int = 100, tf: float = 200):
        if self.calc == 0:
            print("Results not yet calculated. Please take a break for a moment")
            self.calculate(Nt, tf)

        fig1, ax1 = plt.subplots()
        fig1.set_size_inches(16,8)
        fig2, ax2 = plt.subplots()
        fig2.set_size_inches(16,8)
        fig3, ax3 = plt.subplots()
        fig3.set_size_inches(16,8)
        fig4, ax4 = plt.subplots()
        fig4.set_size_inches(16,8)

        EP_array = np.zeros(self.t_arr.size)
        for i in range(0,self.t_arr.size):
            EP_array[i] = self.Ept(self.t_arr[i])
        ax1.plot(self.t_arr/60, EP_array, label = 'EP')
        ax1.set_xlabel("Time (min)")
        ax1.set_ylabel("Electrical Power (EP)")
        #ax1.set_title("Electrical Power over Time")
        plt.show(block=False)

        ax2.plot(self.t_arr/60, self.res[:,0]-273.15, label = 'TC1')
        ax2.plot(self.t_arr/60, self.res[:,1]-273.15, label = 'TC3')
        #ax2.plot(self.t_arr[35:]/60, self.res[:,1][35:]-self.SS.T3, label = 'TC3 above SS value') # step test
        ax2.set_xlabel("Time (min)")
        ax2.set_ylabel("Temperature (C)")
        #ax2.set_title("Temperature of TC1 and TC3 over time")
        ax2.legend()
        plt.show(block=False)

        # for i in range(0,self.t_arr.size):
        #     if self.res[:,1][i] > (0.62*(self.res[:,1][-1]-self.SS.T3)+self.SS.T3) and self.res[:,1][i] < (0.64*(self.res[:,1][-1]-self.SS.T3)+self.SS.T3):
        #         print("\nTime")
        #         print(self.t_arr[i])
        #         print("Temp")
        #         print(self.res[:,1][i])
        
        # print("\nSS")
        # print(0.63*(self.res[:,1][-1]-self.SS.T3)+self.SS.T3)

        ax3.plot(self.t_arr/60, self.res[:,2], label = 'rho H2')
        ax3.plot(self.t_arr/60, self.res[:,3], label = 'rho O2')
        ax3.set_xlabel("Time (min)")
        ax3.set_ylabel("rho (kg/m^3)")
        #ax3.set_title("Density of hydrogen and oxygen over time")
        ax3.legend()
        plt.show(block=False)
        #print((self.res[:,2][90]-self.res[:,2][70])/(self.t_arr[90]-self.t_arr[70]))

        ax4.plot(self.t_arr/60, self.SS.F5*np.ones(self.t_arr.size), label = 'F5')
        ax4.plot(self.t_arr/60, self.F3_H2t(EP_array), label = 'F3_H2')
        ax4.set_xlabel("Time (min)")
        ax4.set_ylabel("Flow (kg/s)")
        #ax4.set_title("Flow in F5 and F3_H2 over time")
        ax4.legend()
        plt.show(block=False)

    def plot_PIDgraphs_p(self, Nt: int = 10000, tf: float = 200):
        
        if self.pidr == 0:
            print("PID not yet calculated. Please take a break for a moment")
            self.PID1(Nt, tf)

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

        EP_array = np.zeros(self.t_arr.size)
        for i in range(0,self.t_arr.size):
            EP_array[i] = self.Ept(self.t_arr[i])
        # ax1.plot(self.t_arr/60, EP_array, label = 'EP')
        # ax1.set_xlabel("Time (min)")
        # ax1.set_ylabel("EP")
        # ax1.set_title("EP")
        # ax1.legend()
        # plt.show(block=False)

        # ax2.plot(self.t_arr/60, self.res[:,4]-273.15, label = 'T1')
        # ax2.plot(self.t_arr/60, self.res[:,5]-273.15, label = 'T3')
        # ax2.set_title("Temperature of TC1 and TC3 over time")
        # ax2.set_xlabel("Time (min)")
        # ax2.set_ylabel("Temperature (K)")
        # ax2.legend()
        # plt.show(block=False)

        pH2_array = np.zeros(self.t_arr.size)
        pO2_array = np.zeros(self.t_arr.size)
        print(self.t_arr.size)
        for i in range(0,self.t_arr.size):
            pH2_array[i], pO2_array[i] = self.p_set(self.t_arr[i])
        # ax3.plot(self.t_arr/60, pH2_array*(self.R*353/self.mwH2)/100000, label = 'PC1.SP')
        # ax3.plot(self.t_arr/60, self.res[:,0]*(self.R*353/self.mwH2)/100000, label = 'PC1.PV')
        # ax3.set_xlabel("Time (min)")
        # ax3.set_ylabel("Hydrogen pressure (bar)")
        # ax3.set_title("Hydrogen pressure over Time")
        # ax3.legend()
        # plt.show(block=False)
        # print(str(np.min(self.res[:,0]))+ ' ' + str(np.max(self.res[:,0])))

        # ax4.plot(self.t_arr/60, pO2_array*(self.R*353/self.mwO2), label = 'PC2.SP')
        # ax4.plot(self.t_arr/60, self.res[:,1]*(self.R*353/self.mwO2), label = 'PC2.PV')
        # ax4.set_xlabel("Time (min)")
        # ax4.set_ylabel("Oxygen pressure (Pa)")
        # ax4.set_title("Oxygen pressure over time")
        # ax4.legend()
        # plt.show(block=False)

        F5_array = np.zeros(self.t_arr.size-1)
        F6_array = np.zeros(self.t_arr.size-1)
        FI_array = np.zeros(self.t_arr.size)
        for i in range(0,self.t_arr.size-1):
            #F5_array[i] = self.SS.F5 + self.KcH2*(self.res[i+1,2]+self.res[i,2]/self.taui)
            # if self.t_arr[i] < 100*60:
            #     F5_array[i] = -(self.res[:,0][i+1]-self.res[:,0][i])*self.V1_gas+self.p2*100 # Remember to make this EPt
            # else: 
            #     F5_array[i] = -(self.res[:,0][i+1]-self.res[:,0][i])*self.V1_gas+self.p2*90
            F5_array[i] = -(self.res[:,0][i+1]-self.res[:,0][i])*self.V1_gas+self.p2*self.Ept(self.t_arr[i])
            F6_array[i] = -(self.res[:,1][i+1]-self.res[:,1][i])*self.V2_gas+self.p3*self.EP
            FI_array[i] = self.res[:,2][i+1]-self.res[:,2][i]

        # ax5.plot(self.t_arr[0:-1]/60, F5_array, label = 'F5')
        # ax5.plot(self.t_arr/60, self.F3_H2t(EP_array), label = 'F3_H2')
        # #ax5.plot(self.t_arr/60, FI_array*0.001, label = 'dI1dt')
        # ax5.set_xlabel("Time (min)")
        # ax5.set_ylabel("Flow (kg/s)")
        # ax5.set_title("Flow in F5 and F3_H2 over time")
        # ax5.legend()
        # plt.show(block=False)

        # ax6.plot(self.t_arr[0:-1]/60, F6_array, label = 'F6')
        # ax6.plot(self.t_arr/60, self.F4_O2t(EP_array), label = 'F4_O2')
        # ax6.set_xlabel("Time (min)")
        # ax6.set_ylabel("Flow (kg/s)")
        # ax6.set_title("Flow in F6 and F4_O2 over time")
        # ax6.legend()
        # plt.show(block=False)

        dIdt = np.ones(self.t_arr.size-1)
        for i in range(0, self.t_arr.size-1):
            dIdt[i] = self.res[i+1,2] - self.res[i,2]
        # ax7.plot(self.t_arr/60, self.res[:,2], label = 'Total error in rhoH2')
        # ax7.plot(self.t_arr[:-1]/60, dIdt, label = 'Error in rhoH2')
        # ax7.set_xlabel("Time (min)")
        # ax7.set_ylabel("Flow (kg/s)")
        # ax7.set_title("Error")
        # plt.show(block=False)

        ax8.plot(self.t_arr/60, self.res[:,2]/abs(np.min(self.res[:,2])), label = 'Total error scaled')
        ax8.plot(self.t_arr[:-1]/60, dIdt*10/abs(np.min(self.res[:,2])), label = 'Error scaled')
        ax8.plot(self.t_arr[:-1]/60, F5_array/(np.max(F5_array)), label='F5 scaled')
        ax8.plot(self.t_arr/60, self.F3_H2t(EP_array)/np.max(F5_array), label = 'F3_H2 scaled')
        ax8.plot(self.t_arr/60, pH2_array/np.max(self.res[:,0]), label = 'PC1.SP scaled')
        ax8.plot(self.t_arr/60, self.res[:,0]/np.max(self.res[:,0]), label = 'PC1.PV scaled')
        ax8.legend()
        plt.show(block=False)

    def plot_PIDgraphs_t(self):

        if self.pidt == 0:
            print("PID not yet calculated. Please take a break for a moment")
            self.PID2()

        fig1, ax1 = plt.subplots()
        fig1.set_size_inches(16,8)
        fig2, ax2 = plt.subplots()
        fig2.set_size_inches(16,8)
        fig3, ax3 = plt.subplots()
        fig3.set_size_inches(16,8)
        fig4, ax4 = plt.subplots()
        fig4.set_size_inches(16,8)
        # fig5, ax5 = plt.subplots()
        # fig5.set_size_inches(16,8)
        # fig6, ax6 = plt.subplots()
        # fig6.set_size_inches(16,8)

        EP_array = np.zeros(self.t_arr.size)
        for i in range(0,self.t_arr.size):
            EP_array[i] = self.Ept(self.t_arr[i])
        ax1.plot(self.t_arr/60, EP_array, label = 'EP')
        ax1.set_xlabel("Time (min)")
        ax1.set_ylabel("EP")
        ax1.set_title("EP")
        ax1.legend()
        plt.show(block=False)

        ax2.plot(self.t_arr/60, self.res[:,0]-273.15, label='T1')
        ax2.plot(self.t_arr/60, self.res[:,1]-273.15, label='T3')
        ax2.set_xlabel("Time (min)")
        ax2.set_ylabel("Temperature (C)")
        ax2.set_title("Temperature of T1 and T3 over time")
        ax2.legend()
        plt.show(block=False)

        TC1_SP_array = np.zeros(self.t_arr.size)
        for i in range(0, self.t_arr.size):
            TC1_SP_array[i] = self.T_set(self.t_arr[i])
        ax3.plot(self.t_arr/60, TC1_SP_array-273.15, label='TC1.SP')
        ax3.plot(self.t_arr/60, self.res[:,1]-273.15, label='TC1.PV')
        ax3.set_xlabel("Time (min)")
        ax3.set_ylabel("Temperature (C)")
        ax3.set_title("TC1 Setpoint and TC1.PV over time")
        ax3.legend()
        plt.show(block=False)

        TSet = np.ones(self.t_arr.size-1)
        for i in range(0, self.t_arr.size-1):
            TSet[i] = (self.res[i+1,0]-self.res[i,0])*60 + self.res[i,0]
        ax4.plot(self.t_arr[:-1]/60, TSet-273.15, label='TCA.SP')
        ax4.set_xlabel("Time (min)")
        ax4.set_ylabel("Temperature (C)")
        ax4.set_title("TCA.SP over time")
        ax4.legend()
        plt.show(block=False)

        # dIdt = np.ones(self.t_arr.size-1)
        # for i in range(0, self.t_arr.size-1):
        #     dIdt[i] = self.res[i+1,2] - self.res[i,2]
        # ax5.plot(self.t_arr/60, self.res[:,2]/10, label='Total error/10')
        # ax5.plot(self.t_arr[:-1]/60, dIdt, label='Error')
        # ax5.set_xlabel("Time (min)")
        # ax5.set_ylabel("Temperature (C)")
        # ax5.set_title("Error over time")
        # ax5.legend()
        # plt.show(block=False)

        # ax6.plot(self.t_arr/60, self.resivp.y[0]-273.15, label='T1')
        # ax6.plot(self.t_arr/60, self.resivp.y[1]-273.15, label='T3')
        # ax6.set_xlabel("Time (min)")
        # ax6.set_ylabel("Temperature (C)")
        # ax6.set_title("Temperature of T1 and T3 over time (Solve_IVP)")
        # ax6.legend()
        # plt.show(block=False)

    def __str__(self):
        print(self.calc)
        if self.calc==1:
            return f'''
Dynamic model Alkaline Electrolyser initialized with the following variables and parameters:\n\n
Variables   Min     Max     Units
EP={self.EP}      100     500     kW
F1_elec={self.F1_elec}   2       2       kg/s
F2_elec={self.F2_elec}   2       2       kg/s
TCA_SP={self.TCA_SP}   70      70      C

Results have been calculated.
'''
        elif self.calc==0:
            return f'''
Dynamic model Alkaline Electrolyser initialized with the following variables and parameters:\n\n
Variables   Min     Max     Units
EP={self.EP}      100     500     kW
F1_elec={self.F1_elec}   2       2       kg/s
F2_elec={self.F2_elec}   2       2       kg/s
TCA_SP={self.TCA_SP}   70      70      C

Results not yet calculated.
'''

# Uncomment to calculate and print steady state values (Question 6)    
#EP = 500
#S1 = SSt(EP, F1_elec, F2_elec, TCA_SP)
#S1.calculate()
#print(S1)

# Uncomment to calculate and plot dynamic simulation (Question 7)
# S2 = dynmodel()
# S2.plot_calcgraphs(5000, 200)
# plt.show()

# Uncomment to calculate and plot dynamic simulation with PI controllers for PC1 and PC2 (Question 8)
S2 = dynmodel()
S2.plot_PIDgraphs_p(500,200)
plt.show()

# Uncomment to calculate and plot dynamic simulation with PI controller for TC1 (Question 9)
# S2 = dynmodel()
# S2.plot_PIDgraphs_t()
# plt.show()


