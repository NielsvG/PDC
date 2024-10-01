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
        self.mwH2 = 2.016
        self.mwO2 = 32.
        self.R = 8.314

        # SS values
        self.SS = SSt(EP, F1_elec, F2_elec, TCA_SP)
        self.SS.calculate()

        self.calc = 0
        self.pidr = 0
        self.pidt = 0

    def Ept(self, t):
        if t<=60:
            return 100
        elif t>60 and t<70:
            return 100 + 400*(t-60)/10
        elif t >= 70 and t <= 130:
            return 500
        elif t > 130 and t<140:
            return  500 - 400*(t-130)/10
        elif t >= 140:
            return 100
        
    def P_set(self, t):
        if t*60<10000:
            P_setH2 = 2
            P_setO2 = 2
        else:
            P_setH2 = 2.3
            P_setO2 = 2.3
        return [P_setH2, P_setO2]

    def F3_H2t(self, EP):
        # Equation 6
        return self.p2*EP
        
    def F4_O2t(self, EP):
        # Equation 7
        return self.p3*EP

    def calculate(self, Nt: int = 100, tf: float = 200):
        self.t_arr = np.linspace(0,tf,Nt)

        def eq(u,t): # u = [T1, T3, rho_H2, rho_O2]
            u_new = np.zeros(4)
            u_new[0] = (self.TCA_SP-u[0])/60 # T1, adapted from equation 2
            u_new[1] = ((self.F1_elec+self.F2_elec)*self.Cp_elec*(u[0]-u[1]))/self.Cs1 + (self.p4*self.Ept(t))/self.Cs1 # T3, adapted from equation 8
            u_new[2] = self.F3_H2t(self.Ept(t))/self.V1_gas-u[2] # rho_H2, adapted from equation 10
            u_new[3] = self.F4_O2t(self.Ept(t))/self.V2_gas-u[3] # rho_O2, adapted form equation 12
            return u_new

        res0 = np.array([self.SS.T1,self.SS.T3,self.SS.F5,self.SS.F6])
        self.res = scipy.integrate.odeint(eq, res0, self.t_arr)

        self.calc = 1
        self.pidr = 0
        self.pidt = 0
        return self.res
    
    def PID1(self, Nt: int = 10000, tf: float = 200, lmb: float = 0.1):
        self.t_arr = np.linspace(0,tf,Nt)
        KH2 = -(self.R*self.SS.T1)/(self.mwH2*self.V1_gas)
        self.KcH2 = 2/(KH2*lmb)
        KO2 = -(self.R*self.SS.T1)/(self.mwO2*self.V2_gas)
        self.KcO2 = 2/(KO2*lmb)
        self.taui = 2*lmb

        def eq(u,t): #u = [PH2, PO2, I1, I2, T1, T3]
            u_new = np.zeros(6)
            u_new[2] = self.P_set(t)[0]-u[0] # error
            u_new[3] = self.P_set(t)[1]-u[1] # error
            C1 = self.KcH2*(u_new[2]+u[2]/self.taui)
            C2 = self.KcO2*(u_new[3]+u[3]/self.taui)
            F5 = self.SS.F5 + C1 # bias
            F6 = self.SS.F6 + C2 # bias
            u_new[0] = ((self.p2*self.Ept(t)-F5)/self.V1_gas)*self.R*u[4]/self.mwH2 #dP_H2/dt
            u_new[1] = ((self.p3*self.Ept(t)-F6)/self.V2_gas)*self.R*u[4]/self.mwO2 #dP_O2/dt
            u_new[4] = (self.TCA_SP-u[4])/60 # T1
            u_new[5] = (self.SS.F1_elec+self.SS.F2_elec)*self.Cp_elec*(u[4]-u[5])/self.Cs1 + self.p4*self.Ept(t)/self.Cs1 # T3
            return u_new

        res0 = np.array([self.SS.F1_elec,self.SS.F2_elec,0,0,self.SS.T1,self.SS.T3])
        self.res = scipy.integrate.odeint(eq, res0, self.t_arr)

        self.pidr = 1
        self.calc = 0
        self.pidt = 0
        return self.res
    
    def T_set(self, t):
        if t*60<10000:
            T_setT1 = 80 + 273.15
        else:
            T_setT1 = 70 + 273.15
        return T_setT1
    
    def PID2(self, Nt: int = 10000, tf: float = 200, lmb: float = 20):
        self.t_arr = np.linspace(0,tf,Nt)
        
        #KT1 = 0.001848
        KT1 = 1.5
        tau = 800
        self.KcT1 = tau/(KT1*lmb)
        taui = tau
        # KT3 = 0
        # self.KcT3 = 0

        def eq3(u,t): #u = [T1, T3, I1]
            u_new = np.zeros(3)
            u_new[2] = self.T_set(t)-u[1] # error
            C1 = self.KcT1*(u_new[2]+u[2]/taui)
            TC1 = C1#self.SS.T1 + C1 # bias
            u_new[0] = (TC1-u[0])/60
            u_new[1] = (self.SS.F1_elec+self.SS.F2_elec)*self.Cp_elec*(u[0]-u[1])/self.Cs1 + self.p4*self.Ept(t)/self.Cs1
            return u_new

        T0 = np.array([self.SS.T1, self.SS.T3, (80-2.7)*taui/self.KcT1])
        self.res = scipy.integrate.odeint(eq3, T0, self.t_arr)

        self.pidt = 1
        self.calc = 0
        self.pidr = 0
        return self.res
    
    def plot_calcgraphs(self):
        if self.calc == 0:
            print("Results not yet calculated. Please take a break for a moment")
            self.calculate()

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
        ax1.plot(self.t_arr, EP_array, label = 'EP')
        ax1.set_xlabel("Time (min)")
        ax1.set_ylabel("Electrical Power (EP)")
        ax1.set_title("Electrical Power over Time")
        plt.show(block=False)

        ax2.plot(self.t_arr, self.res[:,0]-273.15, label = 'TC1')
        ax2.plot(self.t_arr, self.res[:,1]-273.15, label = 'TC3')
        ax2.set_xlabel("Time (min)")
        ax2.set_ylabel("Temperature (C)")
        ax2.set_title("Temperature of TC1 and TC3 over time")
        ax2.legend()
        plt.show(block=False)

        ax3.plot(self.t_arr, self.res[:,2], label = 'rho H2')
        ax3.plot(self.t_arr, self.res[:,3], label = 'rho O2')
        ax3.set_xlabel("Time (min)")
        ax3.set_ylabel("rho (kg/m^3)")
        ax3.set_title("Density of hydrogen and oxygen over time")
        ax3.legend()
        plt.show(block=False)

        ax4.plot(self.t_arr, self.res[:,2]*self.V1_gas, label = 'F5')
        ax4.plot(self.t_arr, self.F3_H2t(EP_array), label = 'F3_H2')
        ax4.set_xlabel("Time (min)")
        ax4.set_ylabel("Flow (kg/s)")
        ax4.set_title("Flow in F5 and F3_H2 over time")
        ax4.legend()
        plt.show(block=False)

    def plot_PIDgraphs_p(self):
        
        if self.pidr == 0:
            print("PID not yet calculated. Please take a break for a moment")
            self.PID1()

        fig1, ax1 = plt.subplots()
        fig1.set_size_inches(16,8)
        fig2, ax2 = plt.subplots()
        fig2.set_size_inches(16,8)
        fig3, ax3 = plt.subplots()
        fig3.set_size_inches(16,8)
        fig4, ax4 = plt.subplots()
        fig4.set_size_inches(16,8)
        fig5, ax5 = plt.subplots()
        fig5.set_size_inches(16,8)
        fig6, ax6 = plt.subplots()
        fig6.set_size_inches(16,8)

        EP_array = np.zeros(self.t_arr.size)
        for i in range(0,self.t_arr.size):
            EP_array[i] = self.Ept(self.t_arr[i])
        ax1.plot(self.t_arr, EP_array, label = 'EP')
        ax1.set_xlabel("Time (min)")
        ax1.set_ylabel("EP")
        ax1.set_title("EP")
        ax1.legend()
        plt.show(block=False)

        ax2.plot(self.t_arr, self.res[:,4]-273.15, label = 'T1')
        ax2.plot(self.t_arr, self.res[:,5]-273.15, label = 'T3')
        ax2.set_title("Temperature of TC1 and TC3 over time")
        ax2.set_xlabel("Time (min)")
        ax2.set_ylabel("Temperature (K)")
        ax2.legend()
        plt.show(block=False)

        PH2_array = np.zeros(self.t_arr.size)
        PO2_array = np.zeros(self.t_arr.size)
        print(self.t_arr.size)
        for i in range(0,self.t_arr.size):
            PH2_array[i], PO2_array[i] = self.P_set(self.t_arr[i])
        ax3.plot(self.t_arr, PH2_array, label = 'PC1.SP')
        ax3.plot(self.t_arr, self.res[:,0], label = 'PC1.PV')
        ax3.set_xlabel("Time (min)")
        ax3.set_ylabel("Hydrogen pressure (Pa)")
        ax3.set_title("Hydrogen pressure over Time")
        ax3.legend()
        plt.show(block=False)
        print(str(np.min(self.res[:,0]))+ ' ' + str(np.max(self.res[:,0])))

        ax4.plot(self.t_arr, PO2_array, label = 'PC2.SP')
        ax4.plot(self.t_arr, self.res[:,1], label = 'PC2.PV')
        ax4.set_xlabel("Time (min)")
        ax4.set_ylabel("Oxygen pressure (Pa)")
        ax4.set_title("Oxygen pressure over time")
        ax4.legend()
        plt.show(block=False)

        F5_array = np.zeros(self.t_arr.size-1)
        F6_array = np.zeros(self.t_arr.size-1)
        for i in range(0,self.t_arr.size-1):
            F5_array[i] = self.SS.F5 + self.KcH2*(self.res[i+1,2]+self.res[i,2]/self.taui)
            F6_array[i] = self.SS.F6 + self.KcO2*(self.res[i+1,3]+self.res[i,3]/self.taui)

        ax5.plot(self.t_arr[0:-1], F5_array, label = 'F5')
        ax5.plot(self.t_arr, self.F3_H2t(EP_array), label = 'F3_H2')
        ax5.set_xlabel("Time (min)")
        ax5.set_ylabel("Flow (kg/s)")
        ax5.set_title("Flow in F5 and F3_H2 over time")
        ax5.legend()
        plt.show(block=False)

        ax6.plot(self.t_arr[0:-1], F6_array, label = 'F6')
        ax6.plot(self.t_arr, self.F4_O2t(EP_array), label = 'F4_O2')
        ax6.set_xlabel("Time (min)")
        ax6.set_ylabel("Flow (kg/s)")
        ax6.set_title("Flow in F6 and F4_O2 over time")
        ax6.legend()
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

        EP_array = np.zeros(self.t_arr.size)
        for i in range(0,self.t_arr.size):
            EP_array[i] = self.Ept(self.t_arr[i])
        ax1.plot(self.t_arr, EP_array, label = 'EP')
        ax1.set_xlabel("Time (min)")
        ax1.set_ylabel("EP")
        ax1.set_title("EP")
        ax1.legend()
        plt.show(block=False)

        ax2.plot(self.t_arr, self.res[:,0]-273.15, label='T1')
        ax2.plot(self.t_arr, self.res[:,1]-273.15, label='T3')
        ax2.set_xlabel("Time (min)")
        ax2.set_ylabel("Temperature (C)")
        ax2.set_title("Temperature of TC1 and TC3 over time")
        ax2.legend()
        plt.show(block=False)

        TC1_SP_array = np.zeros(self.t_arr.size)
        for i in range(0, self.t_arr.size):
            TC1_SP_array[i] = self.T_set(self.t_arr[i])
        ax3.plot(self.t_arr, TC1_SP_array-273.15, label='TC1.SP')
        ax3.plot(self.t_arr, self.res[:,0]-273.15, label='TC1.PV')
        ax3.set_xlabel("Time (min)")
        ax3.set_ylabel("Temperature (C)")
        ax3.set_title("TC1 Setpoint and TC1.PV over time")
        ax3.legend()
        plt.show(block=False)

        ax4.plot(self.t_arr, np.ones(self.t_arr.size)*self.TCA_SP-273.15, label='TCA.SP')
        ax4.set_xlabel("Time (min)")
        ax4.set_ylabel("Temperature (C)")
        ax4.set_title("TCA.SP over time")
        ax4.legend()
        plt.show(block=False)

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
S2 = dynmodel()
S2.plot_calcgraphs()
plt.show()

# Uncomment to calculate and plot dynamic simulation with PI controllers for PC1 and PC2 (Question 8)
# S2 = dynmodel()
# S2.plot_PIDgraphs_p()
# plt.show()

# Uncomment to calculate and plot dynamic simulation with PI controller for TC1 (Question 9)
# S2 = dynmodel()
# S2.plot_PIDgraphs_t()
# plt.show()


