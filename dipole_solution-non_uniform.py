
# coding: utf-8

# In[6]:
print("Compiling Please wait....")

import numpy as np
import pylab as py
import matplotlib.patches as patches
import matplotlib.animation as animation
print("Liberaries imported...")

class dipole_sys:
    
    # defining the equation d^2 phi/dt^2 - M X B/I = 0 as
    # dw/dt = M X B /I
    
    # and dphi/dt = w 
    # w = sqrt(MB/If)
    
    # for damped osccilations d2phi/d2t - M X B/I +f (dphi/dt) /I = 0 

    #                     NON--UNIFORM MAGNETIC FIELD
        
    def __init__(self):
        
        self.c = 10**(-7)         # constant uo/4.pi
        
        (self.wthetao,self.wphio) = (0,0)
        
        # moment of inertia of the dipole
        (self.It,self.Ip)         = (5,5)    # rotation [theta(It)] and revolution [phi(Ip)]
        
        # frictional coeff 
        (self.ft,self.fp)         = (0,0)    # in opposing rotation (wtheta (ft)), revolution (wphi (fp))
        
        self.Mi = 0
        self.Ms = 0
        self.Ri = 0
        self.Rs = 0
        
    
    # Torque acting on the rotor dipole and calculating the change in orientation 
    
    def wp_fun(self):
        
        # Ms  ---  stator dipole moment
        # Rs  ---  displacement vector of the Stator
        # Mi  ---  Rotor dipole moment
        # Ri  ---  diplacement vector of the Rotor
        # dwphi/dtheta  = Mi X B - fp x wphi / Ip
        
        Ro = self.Ri-self.Rs
        Ro_mag = np.linalg.norm(Ro)
        B = self.c*(3*(np.dot(self.Ms,Ro)*Ro/Ro_mag**2)-self.Ms)/Ro_mag**3
        tor = np.cross(self.Mi,B)
        
        return tor[2]/self.Ip-self.fp*self.wphio/self.Ip
    
    
    # Dipole moment function dphi/dtheta = wphi
    
    def p_fun(self):
        return self.wphio
    
    # for damped osccilations d2theta/d2t - r X F/It +ft (d2theta/d2t) /It = 0 
    
    def wt_fun(self):
        
        Ro_bar = self.Ri-self.Rs         # displacement vector of stator and rotor dipoles
        Ro_mag = np.linalg.norm(Ro_bar)
        C1 = np.dot(self.Ms,Ro_bar)*self.Mi+np.dot(self.Mi,Ro_bar)*self.Ms+np.dot(self.Ms,self.Mi)*Ro_bar
        C2 = 5*np.dot(self.Ms,Ro_bar)*np.dot(self.Mi,Ro_bar)*Ro_bar
        F = 3*self.c*(C1-C2/Ro_mag**2)/Ro_mag**5
        T = np.cross(self.Ri,F)
        
        return T[2]/self.It-self.ft*self.wthetao/self.It    
    
    
    def t_fun(self):
        return self.wthetao
    
    
    def potential(self,theta,phi,thetas,phis):
        
        Ms = self.ms*np.array([np.cos(phis),np.sin(phis),0])
        Rs = self.rs*np.array([np.cos(thetas),np.sin(thetas),0])
        Mi = self.mi*np.array([np.cos(phi),np.sin(phi),0])
        Ri = self.ri*np.array([np.cos(theta),np.sin(theta),0])
        
        Ro = Ri-Rs         # displacement vector of stator and rotor dipoles
        Ro_mag = np.linalg.norm(Ro)
        
        C = (Ms[0]*Ro[0]+Ms[1]*Ro[1]+Ms[2]*Ro[2])/Ro_mag**2
        
        Bx = self.c*(3*(C*Ro[0])-Ms[0])/Ro_mag**3
        By = self.c*(3*(C*Ro[1])-Ms[1])/Ro_mag**3
        Bz = self.c*(3*(C*Ro[2])-Ms[2])/Ro_mag**3
        
        U = -(Bx*Mi[0]+By*Mi[1]+Bz*Mi[2])
        return U
    


class calculations(dipole_sys):
    
    """
    calculations
    ============
    
    The derived class of the dipole_sys where the class calculates the parameters of the overall system,
    containing the number of M number of stator and N number of rotor dipoles. The parameters takes the 
    list of the dipole orientation angles and the rotational angular velocities of the N rotor dipoles,
    angular displacement of the system w.r.t to the reference, and also the angular velocity of the system.
    
    
        Parameter List:
        -------------------------------------------------------------------------------------------------
        
        
        The parameters takes the list of the instantaneous list of values of the individual dipoles. The 
        parameters and the default values are:
        
        Symmbol                     Description                                          Default Value
        
         ri        ----     radial distance of the rotor dipoles                       ----  0.01
         rs        ----     radial distance of the stator dipoles                      ----  0.02
         mi        ----     pole strength of the rotor dipoles                         ----  1.00
         ms        ----     pole strength of the stator dipoles                        ----  10.0
         theta     ----     angular displacement of the rotor disc w.r.t the reference ----  0.00
         thetas    ----     initial angular displacement of the reference stator dipole----  0.00
         phi       ----     list of orientations of the rotor dipoles with the x-axis  ----  0.00
         phis      ----     orientation of the reference stator dipole with the x-axis ----  0.00
         wtheta    ----     rate of change of angular position of the rotor disc       ----  0.00
         wphi      ----     list of rate of change of orientations of the rotor dipoles----  0.00
         It        ----     Inertia of the change of the disc angular position         ----  5.00
         Ip        ----     Inertia of the change of the each dipoles orentation       ----  5.00
         ft        ----     coefficient of friction along the revolution of the disc   ----  0.00
         fp        ----     coefficient of friction along the rotation of the dipole   ----  0.00
         N         ----     the number of rotor dipoles in the system                  ----  1
         M         ----     the number of stator dipoles in the system                 ----  1
         stat_step ----     the angular displacement between the stator dipoles        ----  5
         
         
        Methods
        -------
        
        fun_W_phi       -----     calculates the rate of change in rotational of the 'N' rotor dipoles 
                                  due to the torque acting in the net magnetic field of the 'M' stator
                                  at the position of the rotor dipoles and returns a list of angular
                                  orientations of the rotor dipoles
                                  |
                                  |                                        
                                  -->  [dwphi/dt]  =  [\sigma{i=1}{N}[\sigma{j=1}{M} (wp_fun)]]
                                       |
                                       --> wp_fun calculates the torque on the rotor dipole
                                                                            
        
        fun_W_theta     -----    calculates the rate of change in angular velocity of the rotor disc 
                                 w.r.t to the reference
                                 |
                                 |
                                 -->  dwtheta/dt  =  \sigma{i=1}{N}[\sigma{j=1}{M} (wt_fun)]
                                       |
                                       --> wt_fun calculates the torque on the rotor dipole
                                  
        fun_theta       -----    calculates the rate of change of angular displacement of the rotor 
                                 disc
                                 |
                                 |
                                 -->  dtheta/dt  =  wtheta
        
        fun_phi         -----    calculates the rate of rotation of the individual dipoles and returns
                                 a list of rotational velocities of the dipoles
                                 |
                                 |
                                 -->  dphi/dt  =  wphi
        
        
        rotor_update    -----    updates the variables of the system for evaluating the parameters of
                                 the system theta, phi, wtheta and wphi. Updates the values to find the 
                                 force acting on the rotor dipole 'j' due to the other rotor dipole 'i'
        
        stator_update   -----    updates the variables of the system to find the parameters of the 
                                 rotor dipoles. Updates the values to find the force acting on the
                                 rotor dipole 'j' due to the stator dipole 'k'
        
    
    --------------------------------------------------------------------------------------------------------
    
    dipole_sys
    ==========
    
    Calculates the parameters like the dipole angular position and the orientation and also the
    rotational and revolution angular velocities of the single rotor dipole due to the force of
    the single stator dipole at a particular instant of time.
     
         
         Methods:
         --------
         
         wp-fun   ------  Calculates the rate of change in rotational velocity of the rotor dipole
                          due to the torque acting by the stator dipole
                          |
                          |
                          -->  dwphi/dt  =  Mi X B - fp * wphi / Ip
                               |
                               -->  Mi is the rotor dipole moment
                               -->  B is the magnetic induction at the rotor due to the stator dipole
                               
                               
         p_fun    ------  Calculates the rate of change of angular displacement phi
                          |
                          |
                          -->  dphi/dt  =  wphi
                          
                          
         wt_fun   ------  Evaluates the rate of change of angular velocity of the rotor dipole due
                          to the torque acting by the stator dipole 
                          |
                          |
                          -->  dwtheta/dt  =  r X F - ft * wtheta / It
                               |
                               -->  r is the displacement vector of the rotor dipole
                               -->  F is the force acting on the rotor dipole due to the stator dipole
                               
                          
         t_fun    ------  Calculates the rate of change of angular displacement of the dipole
                          |
                          |
                          -->  dtheta/dt  =  wtheta
                          
                          
        potential ------  Calculates the potential of the two dipole
                          |
                          |
                          ==>  Parameters -- theta, phi, thetas, phis
                          |
                          |
                          -->  U  = - Bs . Mi
                               |
                               --> B is the induction due to the stator dipole at the position of the rotor dipole
                               --> Mi is the dipole moment of the dipole
        
    
    """
    
    def __init__(self):
        
        dipole_sys.__init__(self)
        
        # Parameter list
        
        (self.theta,self.phi)    = (0,0)
        (self.thetas,self.phis)  = (0,0)
        (self.wphi,self.wtheta)  = (0,0)
        (self.ri, self.rs)       = (0.01,0.02)
        (self.mi, self.ms)       = (1,10)
        (self.N,self.M)          = (1,1)
        
        self.stat_step = 5*np.pi/180
        
    
    def rotor_update(self,i,j):
        
        # Updating the parameters 
        Theta = self.theta+2*i*np.pi/self.N    # 'i' th rotor dipole instantaneous angular position, theta as a function of time
        Phi   = self.phi[i%self.N]             # 'i' th rotor dipole instantaneous orientation, phi as a function of time
        Thetas= self.theta+2*j*np.pi/self.N
        Phis  = self.phi[j%self.N]
        
        (R_s,R_i) = (self.ri,self.ri)          # the rotor dipole radial distance
        (M_i,M_s) = (self.mi,self.mi)          # pole strength of the rotor dipoles
        
        self.Ms = M_s*np.array([np.cos(Phis),np.sin(Phis),0])
        self.Rs = R_s*np.array([np.cos(Thetas),np.sin(Thetas),0])
        self.Mi = M_i*np.array([np.cos(Phi),np.sin(Phi),0])
        self.Ri = R_i*np.array([np.cos(Theta),np.sin(Theta),0])
        
        self.wphio  = self.wphi[i]
        self.wthetao= self.wtheta
        
        
    def stator_update(self,i,k):
        theta = self.theta+2*i*np.pi/self.N   # 'i' th rotor dipole instantaneous angular position, theta as a function of time
        phi   = self.phi[i%self.N]            # 'i' th rotor dipole instantaneous orientation, phi as a function of time
        thetas= self.thetas+self.stat_step*k  # the stator dipole angular position with step value of 'stat_step'
        phis  = self.phis+self.stat_step*k    # the stator dipole orientation
        
        (rs,ri) = (self.rs,self.ri)          # the stator and rotor dipole radial distances
        (mi,ms) = (self.mi,self.ms)          # pole strengths of the stator and rotor dipole
        
        self.Ms = ms*np.array([np.cos(phis),np.sin(phis),0])
        self.Rs = rs*np.array([np.cos(thetas),np.sin(thetas),0])
        self.Mi = mi*np.array([np.cos(phi),np.sin(phi),0])
        self.Ri = ri*np.array([np.cos(theta),np.sin(theta),0])
        
        self.wphio  = self.wphi[i]
        self.wthetao= self.wtheta
        
        
    def fun_W_phi(self):
        moment = []
        
        # rotational torque of the 'i' th rotor dipole due to the 'j' th rotor dipole
        # T_ij = wp_fun()_ij
        # rotational torque of the 'i' th rotor dipole due to the stator 's'th dipole
        # T_is = wp_fun()_is
        # sum_J + sum_S = wp_fun()_I + sp_fun()_I
        
        for i in range(self.N):
            mom_rot = 0
            # T_ij = wp_fun()_ij
            for j in range(i+1,self.N+i,1):
                self.rotor_update(i,j)
                mom_rot = self.wp_fun()+mom_rot
            
            # T_is = wp_fun()_is
            for k in range(self.M):
                self.stator_update(i,k)
                mom_rot = self.wp_fun()+mom_rot
                
            # returns the list of rotor dipole orientations
            moment.append(mom_rot)
            
        return np.array(moment)
    
    def fun_W_theta(self):
        dipole_rot = 0
        
        # torque on the 'i' th rotor dipole due to the 'j' th rotor dipole
        # T_ij = wt_fun()_ij
        # torque of the 'i' th rotor dipole due to the stator 's'th dipole
        # T_is = wt_fun()_is
        # sum_J + sum_S = wp_fun()_I + sp_fun()_I
        
        for i in range(self.N):
            mom_rot = 0
            # T_ij = wt_fun()_ij
            for j in range(i+1,self.N+i,1):
                self.rotor_update(i,j)
                mom_rot = self.wt_fun()+mom_rot
            
            # T_is = wt_fun()_is
            for k in range(self.M):
                self.stator_update(i,k)
                mom_rot = self.wt_fun()+mom_rot
                
            # returns the rotation of the rotor disc
            dipole_rot = mom_rot+dipole_rot
        
        return dipole_rot
    
    def fun_theta(self):
        # the disc angular velocity
        self.wthetao = self.wtheta
        return self.t_fun()
    
    def fun_phi(self):
        # returns the list rotational angular velocities of the rotor dipoles 
        self.wphio = self.wphi
        return np.array(self.p_fun())
    

print("Class compiled...")
# RK4_loop for calculating the difference equation
dipole = calculations()

print(dipole.__doc__)

def k_loop(t,theta,phi,wphi,wtheta):
    for fun in [dipole.fun_W_theta, dipole.fun_W_phi, dipole.fun_theta, dipole.fun_phi]:
        # returns k1w and k1t for ti,wi,thetai
        dipole.theta  = theta
        dipole.phi    = phi 
        dipole.wtheta = wtheta
        dipole.wphi   = wphi
        yield fun()


def RK4_loop(t,h,theta,phi,wtheta,wphi):
    l=1
    while(l<=len(t)):
        
        k1 = [val for val in k_loop(t,      theta,            phi,            wphi,            wtheta            )]
        k2 = [val for val in k_loop(t+0.5*h,theta+0.5*h*k1[2],phi+0.5*h*k1[3],wphi+0.5*h*k1[1],wtheta+0.5*h*k1[0])]
        k3 = [val for val in k_loop(t+0.5*h,theta+0.5*h*k2[2],phi+0.5*h*k2[3],wphi+0.5*h*k2[1],wtheta+0.5*h*k2[0])]
        k4 = [val for val in k_loop(t+h,    theta+h*k3[2],    phi+h*k3[3],    wphi+h*k3[1],    wtheta+h*k3[0]    )]
        theta = theta+(k1[2]+2*k2[2]+2*k3[2]+k4[2])*h/6
        phi   = phi+(k1[3]+2*k2[3]+2*k3[3]+k4[3])*h/6
        wtheta= wtheta+(k1[0]+2*k2[0]+2*k3[0]+k4[0])*h/6
        wphi  = wphi+(k1[1]+2*k2[1]+2*k3[1]+k4[1])*h/6        
        l=l+1
        yield [theta],list(phi),list(wphi),[wtheta]


# start time and end time
t_start,t_end = 0,500

# step value
h = 0.01 

# time range
t = np.arange(t_start,t_end,h)

print("compiling time is 0-500s with step value of 0.01s")
# For damped oscillations
# RK4_loop(t,theta,phi,wtheta,wphi,h,f)
# theta,phi,wtheta,wphi
print(
"""
initial values
ft,fp          =   0,0
mi,ms          =   1,10
phis,thetas    =   pi,40*np.pi/180
N,M            =   1,1
phio,whio      =   [np.pi/2],[0]  list of rotor diople orientation, diople angular angles

thetao,wthetao =   the initial value of the reference rotor position and velocity
rs,ri          =   0.02m, 0.01m
""")
dipole.ft ,dipole.fp = 0.00,0.00 # coefficient of friction
dipole.mi,dipole.ms = 1,10       # pole strenghts of the rotor and stator dipoles
dipole.phis = np.pi              # the stator dipole orientation w.r.t x-axis
dipole.thetas = 40*np.pi/180     # the stator dipole angular position
dipole.N = 1                     # number of rotor dipoles
dipole.M = 1                     # number of stator dipoles

phio = [np.pi/2]                 # list of rotor diople orientation angles
wphio = np.zeros(dipole.N)       # list of rotor diople angular velocities
                                 # the length of the list should be equal to the value of N

thetao = 0                       # the initial value of the reference rotor dipole angular position
wthetao = 0                      # the initial angular velocity of the rotor disc

dipole.rs = 0.02                 # the stator dipole radial distance
dipole.ri = 0.01                 # the rotor dipole radial distance
res = 0

RK = RK4_loop(t,h,thetao,phio,wthetao,wphio)
res = [ele for ele in RK]        # Returns the parameters [[Theta], [phi], [whi], [wtheta]]


Res = np.array(res).T[0]   # for N>1 remove the value after T (i.e just make it as np.array(res).T

Theta  = np.array(list(Res[0])).reshape(-1)
Phi    = np.array(list(Res[1])).T
Wphi   = np.array(list(Res[2])).T
Wtheta = np.array(list(Res[3])).reshape(-1)

Kinetic = np.array(Wtheta)**2*dipole.It*0.5
Potential = 0

if(dipole.N==1):
    Kinetic = Kinetic+(np.array(Wphi)**2*dipole.Ip*0.5)
    for k in range(dipole.M):
        ThetaS = dipole.thetas+dipole.stat_step*k
        PhiS = dipole.phis+dipole.stat_step*k
        Potential = dipole.potential(Theta,Phi,ThetaS,PhiS)+Potential
        
else:
    for i in range(dipole.N):
        Pot_stat_rot = 0
        Pot_rot_rot = 0
        Kinetic = Kinetic+(np.array(Wphi[i])**2*dipole.Ip*0.5)
        for k in range(dipole.M):
            ThetaS = dipole.thetas+dipole.stat_step*k
            PhiS = dipole.phis+dipole.stat_step*k
            Pot_stat_rot = dipole.potential(Theta+2*i*np.pi/dipole.N,Phi[i],ThetaS,PhiS)+Pot_stat_rot
            
        
        for j in range(i+1,dipole.N,1):
            ThetaS = Theta+2*j*np.pi/dipole.N
            PhiS = Phi[j]
            Pot_rot_rot = dipole.potential(Theta+2*i*np.pi/dipole.N,Phi[i],ThetaS,PhiS)+Pot_rot_rot
        
        Potential = Pot_stat_rot+Pot_rot_rot+Potential
        #Pot_S_R = Pot_S_R+Pot_stat_rot  # Total potential of the rotor dipoles in the field of the stator dipoles
        #Pot_R_R = Pot_R_R+Pot_rot_rot   # Total potential of the rotor system dipoles
        


# animation of the dipoles

#  ________NOTE__________

# if the N>1 then put index Phi as 'Phi[p][i]' in the animate(i) func.
# and also add arrow1[] in the return statement of the animate(i) with indexing
# the number of arrow1 should be equal to the N

fig, ax = py.subplots(figsize=(15,8))
ax.set_xlim(-0.03,0.03)
ax.set_ylim(-0.03,0.03)
center = (0,0)

num = len(t)
print("animating the dipoles...")

for k in range(dipole.M):
    R = dipole.ri
    Rs = dipole.rs
    Plen = R/2
    Pw   = 10**(-2)
    
    ThetaS = dipole.thetas+dipole.stat_step*k
    PhiS = dipole.phis+dipole.stat_step*k
        
    Rs = Rs*np.array([np.cos(ThetaS),np.sin(ThetaS)])
    vec_rs = np.array([Plen*np.cos(PhiS),Plen*np.sin(PhiS)])
    x_s,y_s = Rs-vec_rs/2
    dx_s,dy_s = vec_rs
    
    circle = patches.Circle(center,radius=R,color = 'green',fill=False)
    arrow = patches.Arrow(x_s,y_s,dx_s,dy_s,width=Pw,ec='red',fc='green')
    fig = py.gcf()
    ax.add_artist(circle)
    ax.add_artist(arrow)
    ax.grid()

time_template = 'time = %1.2fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
theta_template = 'theta = %1.2f deg'
theta_text = ax.text(0.05, 0.8, '', transform=ax.transAxes)

def animate(i):
    arrow1 = []
    for p in range(dipole.N):
        cen_pat = np.array([R*np.cos(Theta[i]+p*2*np.pi/dipole.N),R*np.sin(Theta[i]+p*2*np.pi/dipole.N)])
        vec = np.array([Plen*np.cos(Phi[i]),Plen*np.sin(Phi[i])])
        x,y = cen_pat-vec/2
        dx,dy = vec
        arrow = patches.Arrow(x,y,dx,dy,width=Pw,ec='red',fc='green')
        ax.add_artist(arrow)
        arrow1.append(arrow)
    
    time_text.set_text(time_template % (i*h))
    theta_text.set_text(theta_template % ((Theta[i])*180/np.pi))
    return arrow1[0],time_text,theta_text,

def init():
    ax.add_patch(arrow)
    time_text.set_text('')
    theta_text.set_text('')
    return arrow,time_text,theta_text

anim = animation.FuncAnimation(fig, animate, 
                               init_func=init, 
                               frames=num, 
                               interval=10,
                               blit=True)

#anim.save('dipole_simulation.mp4',fps=15)
# Draw circle

py.show()

print("plotting the parameters...")
py.subplots(figsize=(15,8))

py.plot(t,180*Theta/np.pi,t,180*Phi/np.pi)
py.legend(["Theta","Phi"])
py.xlabel('time')
py.ylabel('$Angle d^{o}$')

py.subplots(figsize=(15,8))
py.plot(t,Wtheta,t,Wphi)
py.legend(["WTheta","WPhi"])
py.xlabel('time')
py.ylabel('Angle velocity rad/sec')

py.subplots(figsize=(15,8))
py.plot(t,Kinetic,t,Potential,t,Potential+Kinetic)
py.legend(('Kin','Pot','Mech'),loc = 10)
py.xlabel('time')
py.ylabel('Energy')
py.show()
py.pause(100)

