#!/usr/bin/env python
# coding: utf-8

# In[6]:

import math
import numpy as np
import sympy as sym
from sympy.abc import t
from sympy import exp, Abs
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp



####################
# Simulation helpers
def integrate(f,x0,dt):
    """
    This function takes in an initial condition x0 and a timestep dt,
    as well as a dynamical system f(x) that outputs a vector of the
    same dimension as x0. It outputs a vector x at the future time step.
    """
    k1=dt*f(x0)
    k2=dt*f(x0+k1/2.)
    k3=dt*f(x0+k2/2.)
    k4=dt*f(x0+k3)
    xnew=x0+(1/6.)*(k1+2.*k2+2.*k3+k4)
    return xnew

def simulate(f,x0,tspan,dt):
    """
    This function takes in an initial condition x0, a timestep dt,
    a time span tspan consisting of a list [min_time, max_time],
    as well as a dynamical system f(x) that outputs a vector of the
    same dimension as x0. Additionally, this includes a flag (default false)
    that allows one to supply an Euler intergation scheme instead of 
    the given scheme. It outputs a full trajectory simulated
    over the time span of dimensions (xvec_size, time_vec_size).
    """
    N = int((max(tspan)-min(tspan))/dt)
    x = np.copy(x0)
    tvec = np.linspace(min(tspan),max(tspan),N)
    xtraj = np.zeros((len(x0),N))
    for i in range(N):
        xtraj[:,i]=integrate(f,x,dt)
        x = np.copy(xtraj[:,i])
    return xtraj   

def simulate_impact(f,x0,tspan,dt):
    """
    This function takes in an initial condition x0, a timestep dt,
    a time span tspan consisting of a list [min_time, max_time],
    as well as a dynamical system f(x) that outputs a vector of the
    same dimension as x0. Additionally, this includes a flag (default false)
    that allows one to supply an Euler intergation scheme instead of 
    the given scheme. It outputs a full trajectory simulated
    over the time span of dimensions (xvec_size, time_vec_size).
    """
    N = int((max(tspan)-min(tspan))/dt)
    x = np.copy(x0)
    tvec = np.linspace(min(tspan),max(tspan),N)
    xtraj = np.zeros((len(x0),N))
    count = 0;
    for i in range(N):
        if impact_condition(x):
            xtraj[:,i]=integrate(f,impact_update(x),dt)
            count = count+1
        else:
            xtraj[:,i]=integrate(f,x,dt)
        
        x = np.copy(xtraj[:,i])
    return xtraj   


def derivative(f,x):
    ans = sym.Matrix([f]).jacobian(x)
    return ans;


# In[7]:


def animate(theta_array,T=1):
    """
    YOU MUST CHANGE THIS FUNCTION TO PLOT TRIPLE PENDULUM.
    Currently the innards of this function would plot a double pendulum.
    """
    ################################
    # Imports required for animation.
    from plotly.offline import init_notebook_mode, iplot
    from IPython.display import display, HTML
    
    #######################
    # Browser configuration.
    def configure_plotly_browser_state():
      import IPython
      display(IPython.core.display.HTML('''
            <script src="/static/components/requirejs/require.js"></script>
            <script>
              requirejs.config({
                paths: {
                  base: '/static/base',
                  plotly: 'https://cdn.plot.ly/plotly-1.5.1.min.js?noext',
                },
              });
            </script>
            '''))
    configure_plotly_browser_state()
    init_notebook_mode(connected=False)
    
    ###############################################
    # Getting data from pendulum angle trajectories.
    xx1=theta_array[0]
    yy1=theta_array[1]
    xx2=theta_array[2]
    yy2=theta_array[3]
    xx3=theta_array[4]
    yy3=theta_array[5]
    xx4=theta_array[6]
    yy4=theta_array[7]

    lxx1=theta_array[8]
    lyy1=theta_array[9]
    lxx2=theta_array[10]
    lyy2=theta_array[11]
    lxx3=theta_array[12]
    lyy3=theta_array[13]

    cxx1=theta_array[14]
    cyy1=theta_array[15]
    cxx2=theta_array[16]
    cyy2=theta_array[17]

    N = len(x1) # Need this for specifying length of simulation
    
    ####################################
    # Using these to specify axis limits.
    xm=-1
    xM=1
    ym=-1
    yM=5

    ###########################
    # Defining data dictionary.
    # Trajectories are here.
    data=[dict(x=xx1, y=yy1, 
               mode='lines', name='Pendulum', 
               line=dict(width=2, color='blue')
              ),
          dict(x=xx1, y=yy1, 
               mode='lines', name='Mass 1 Traj.',
               line=dict(width=2, color='purple')
              ),
          dict(x=xx2, y=yy2, 
               mode='lines', name='Mass 2 Traj.',
               line=dict(width=2, color='green')
              ),
          dict(x=xx3, y=yy3, 
               mode='lines', name='Mass 2 Traj.',
               line=dict(width=2, color='red')
              ),
        ]
    
    ################################
    # Preparing simulation layout.
    # Title and axis ranges are here.
    layout=dict(xaxis=dict(range=[xm, xM], autorange=False, zeroline=False,dtick=1),
                yaxis=dict(range=[ym, yM], autorange=False, zeroline=False,scaleanchor = "x",dtick=1),
                title="Swinging Sticks Simulation", 
                hovermode='closest',
                updatemenus= [{'type': 'buttons',
                               'buttons': [{'label': 'Play','method': 'animate',
                                            'args': [None, {'frame': {'duration': T, 'redraw': False}}]},
                                           {'args': [[None], {'frame': {'duration': T, 'redraw': False}, 'mode': 'immediate',
                                            'transition': {'duration': 0}}],'label': 'Pause','method': 'animate'}
                                          ]
                              }]
               )
    
   ########################################
    # Defining the frames of the simulation.
    # This is what draws the lines from
    # joint to joint of the pendulum.
    frames=[dict(data=[dict(x=[cxx1[k],cxx2[k]], 
                            y=[cyy1[k],cyy2[k]],
                            
                            mode='lines',
                            line=dict(color='black', width=1),
                            uid = 't1'
                            ),
                       dict(x=[lxx1[k],lxx2[k],lxx3[k]], 
                            y=[lyy1[k],lyy2[k],lyy3[k]],
                            
                            mode='lines',
                            line=dict(color='black', width=1),
                            uid = 't1'
                          
                            ),
                       dict(x=[xx1[k],xx2[k]], 
                            y=[yy1[k],yy2[k]],
                            
                            mode='lines',
                            line=dict(color='black', width=1),
                            uid = 't1'
                            ),
                       dict(x=[xx3[k],xx4[k]], 
                            y=[yy3[k],yy4[k]],
                            
                            mode='lines',
                            line=dict(color='black', width=1),
                            uid = 't1'
                            )
                       
                      
    
                      ]) for k in range(N)
            ]


      
    #######################################
    # Putting it all together and plotting.
    figure1=dict(data=data,layout=layout, frames=frames)           
    iplot(figure1)


# In[8]:


def invertTrans(mat):
    
    '''
    Inverts a Transformation Matrix
    '''
    
    R = sym.Matrix([[mat[0,0],mat[0,1],mat[0,2]],[mat[1,0],mat[1,1],mat[1,2]],[mat[2,0],mat[2,1],mat[2,2]]])
    Rinv = R.T

    p = sym.Matrix([mat[0,3],mat[1,3],mat[2,3]])
    insert = -Rinv*p 

    result = sym.Matrix([[Rinv[0,0],Rinv[0,1],Rinv[0,2],insert[0]],[Rinv[1,0],Rinv[1,1],Rinv[1,2],insert[1]],[Rinv[2,0],Rinv[2,1],Rinv[2,2],insert[2]],[0,0,0,1]])
    
    return result


def se3ToVec(v_se3):
    '''
    Used to convert an se3 twist to a 6-Vector twist
    '''
    
    v_six = sym.Matrix([v_se3[0, 3],
                        v_se3[1, 3],
                        v_se3[2, 3],
                        v_se3[3, 1],
                        v_se3[0, 2],
                        v_se3[1, 0]])

    return v_six


# In[9]:


####################################################
# RIGID BODY PROPERTIES
####################################################

# pendulum length
l = 1

# pendulum mass
m = 1

# pendulum inertia
j = 1

# pendulum first axle height
h = 1

# gravity (yes this technically isn't a rigid body property ¯\_(ツ)_/¯)
g = 9.81

# Create spacial inertia matrix (assuming identical pendulums)
Ipend1 = sym.Matrix(np.diag([m,m,m,j,j,j]))
Ipend2 = Ipend1
Icart = Ipend1


## Configuration Variable Setup #################################

# Set up State Variables

th1 = sym.Function(r'\theta_1')(t)    # angle of first pendulum from horizontal
th2 = sym.Function(r'\theta_2')(t)    # angle of second pendulum from horizontal
x = sym.Function('x')(t)              # x position of first pendulum axle

q = sym.Matrix([x,th1,th2])
qdot = q.diff(t)
qddot = qdot.diff(t)

####################################################
# TRANSFORMATION SETUP

    # Frame w - world frame
    # Frame b - base frame at first pendulum axle
    # Frame p1 - frame aligned with first pendulum
    # Frame p2 - frame aligned with second pendulum
    # Frame x - Cart COM frame directly below pendulum axle
    
####################################################

## Translational Transformations #################################

g_transb = sym.Matrix([[1,0,0,x],
                       [0,1,0,h],
                       [0,0,1,0],
                       [0,0,0,1]])

g_transp2 = sym.Matrix([[1,0,0,l/2],
                       [0,1,0,0],
                       [0,0,1,0],
                       [0,0,0,1]])

g_transx = sym.Matrix([[1,0,0,0],
                       [0,1,0,-h],
                       [0,0,1,0],
                       [0,0,0,1]])

## Rotational Transformations ####################################

g_rotp1 = sym.Matrix([[sym.cos(th1),-sym.sin(th1),0,0],
                      [sym.sin(th1),sym.cos(th1),0,0],
                      [0,0,1,0],
                      [0,0,0,1]])

g_rotp2 = sym.Matrix([[sym.cos(th2),-sym.sin(th2),0,0],
                      [sym.sin(th2),sym.cos(th2),0,0],
                      [0,0,1,0],
                      [0,0,0,1]])

## System Transformations - all bodies relative to the base frame ##########################

g_bp1 = g_rotp1

g_bp2 = g_bp1*g_transp2*g_rotp2

g_bx = g_transx

## Final Transformations - all bodies relative to the world frame ##########################

g_wb = g_transb

g_wp1 = g_wb*g_bp1

g_wp2 = g_wb*g_bp2

g_wx = g_wb*g_bx

print("Nice Job Setting Up! Now Let's do some math!")

####################################################
# ASSEMBLE THE LAGRANGIAN!!!
####################################################

## Calculate the Kinetic Energy of each body #################################
# KE = .5 * V.T * I * V

Vp1 = invertTrans(g_wp1)*g_wp1.diff(t)
Vp1 = se3ToVec(Vp1)

Vp2 = invertTrans(g_wp2)*g_wp2.diff(t)
Vp2 = se3ToVec(Vp2)

Vx = invertTrans(g_wb)*g_wb.diff(t)
Vx = se3ToVec(Vx)

KEp1 = sym.simplify(0.5 * Vp1.T * Ipend1 * Vp1)
KEp2 = sym.simplify(0.5 * Vp2.T * Ipend2 * Vp2)
KEx = sym.simplify(0.5 * Vx.T * Icart * Vx)
KE = sym.simplify(KEp1[0] + KEp2[0] + KEx[0])

## Calculate the Potential Energy of each body ###############################
# PE = m * g * h

PEp2 = m * g * g_wp2[1,3]
PE = sym.simplify(PEp2)

## Calculate the Lagrangian!!! ###############################################

L = sym.simplify(KE-PE)

####################################################
# DEFINE CONSTRAINTS AND EXTERNAL FORCES
####################################################

fx = qdot[0]/sym.Abs(qdot[0])*5
#### Delete the zero and '#' if you want add the swinging stick force.
fth1 = 0#qdot[1]/abs(qdot[1])*sym.exp(-((th1+90/180*np.pi)%(2*np.pi))*5)
f_m = sym.Matrix([fx,fth1,0])

####################################################
# CALCULATE EULER LAGRANGE AND HAMILTONIAN EQUATIONS
####################################################

## Calculate Euler Lagrange Equations ###################################
dLdq = sym.Matrix([L]).jacobian(q).T
dLdqdot = sym.Matrix([L]).jacobian(qdot).T
dLdqdot_dt = dLdqdot.diff(t)

EL_eq = sym.Eq(dLdqdot_dt-dLdq, sym.Matrix(np.zeros(3))+f_m)
print('Euler Lagrange Equations Calculated, Calculating Hamiltonian...')

## Calculate Conserved Hamiltonian ######################################

p = sym.Matrix([L]).jacobian(qdot).T 
Ham = -L
for i in range(len(p)):
    Ham+=p[i]*qdot[i]
print('Hamiltonian Calculated, Solving for EOM...')
## Solve Euler Lagrange Equations for EOM ###############################
sols = sym.solve(EL_eq, [qddot[0],qddot[1],qddot[2]])

qddx = sols[qddot[0]]
qddth1 = sols[qddot[1]]
qddth2 = sols[qddot[2]]

print('Equations of Motion Calculated!')
####################################################
# DERIVE IMPACT UPDATE
####################################################

## Set up Impact Location and Calculate Momentum Constraint ############
phi = g_wb[0,3]-(6-l)             # This is the surface location
dphidq = derivative(phi,q)
lam = sym.symbols(r'\lambda')

def impact_condition(x):
    '''
    This function detects whether or not an impact has occured.
    '''
    q_subber = {q[0]:x[0],qdot[0]:x[1],q[1]:x[2],qdot[1]:x[3],q[2]:x[4],qdot[2]:x[5]}
    test = abs((phi.subs(q_subber)))
    if test<.05 or test>(2*(6-l)-0.05):
        result = True
    else:
        result = False
        
    return result

## Create Impact Update Law #############################################

def impact_update(x):
    '''
    This function updates the velocities due to a collision. The direction from which the system is approaching
    the impact surface is first checked, then the impact update law is calculated and applied. It is important to
    know which direction the system is coming from so that the impulse is applied in the opposite direction.
    '''
    
    # Checking approach direction
    if x[0]>0:
        case = 0
    elif x[0]<0:
        case = 1
        
    if x[3]>0:
        case2 = 0
    elif x[3]<0:
        case2 = 1
        
    # Calculating Impact update law
    q_subber = {q[0]:x[0],qdot[0]:x[1],q[1]:x[2],qdot[1]:x[3],q[2]:x[4],qdot[2]:x[5]}
    pdiff=p.subs(q_subber)-p
    hamdiff = Ham.subs(q_subber)-Ham

    impactleft = sym.Matrix([pdiff[0],pdiff[1],pdiff[2],hamdiff])
    impactright = sym.Matrix([lam*dphidq[0],lam*dphidq[1],lam*dphidq[2],0])

    impact_eqs = sym.Eq(impactleft,impactright)
    ans = sym.solve(impact_eqs,[qdot[0],qdot[1],qdot[2],lam])[case]
    ans2 = sym.solve(impact_eqs,[qdot[0],qdot[1],qdot[2],lam])[case2]
    
    #Updating velocities...
    xv = float(ans[0].subs(q_subber))
    th1v = float(ans2[1].subs(q_subber))
    th2v = float(ans2[2].subs(q_subber))
    
    update_x = np.array([x[0],xv,x[2],th1v,x[4],th2v])
    return(update_x)


# In[10]:


from sympy import lambdify 
####################################################
# Compute Trajectories
####################################################

## Lambdify EOM to restore subbing capability ##################

v1, vd1, v2, vd2, v3, vd3 = sym.symbols(r'v_1 \dot{v_1} v_2 \dot{v_2} v_3 \dot{v_3}')
qsubber = {q[0]:v1,q[1]:v2,q[2]:v3,qdot[0]:vd1,qdot[1]:vd2,qdot[2]:vd3}

xdum = qddx.subs(qsubber)
th1dum = qddth1.subs(qsubber)
th2dum = qddth2.subs(qsubber)

th1_lamb = lambdify([v1,vd1,v2,vd2,v3,vd3],th1dum)
th2_lamb = lambdify([v1,vd1,v2,vd2,v3,vd3],th2dum)
x_lamb = lambdify([v1,vd1,v2,vd2,v3,vd3],xdum)

## Create Function to simulate dynamics #########################

# This should be changed to dyn(t,x) when using solve_ivp
def dyn(x):
    '''
    This is the impact update code when using solve_ivp. I couldn't get it to work properly.
    impact = impact_condition(x)
    
    if impact:
        x = impact_update(x)
    '''
    ddx = x_lamb(x[0],x[1],x[2],x[3],x[4],x[5])
    ddth1 = th1_lamb(x[0],x[1],x[2],x[3],x[4],x[5])
    ddth2 = th2_lamb(x[0],x[1],x[2],x[3],x[4],x[5])
    xdot = np.array([x[1],ddx,x[3],ddth1,x[5],ddth2])
    
    return xdot

## Set up and Solve Trajectory using solve_ivp integrator #######

print('Computing Trajectory')
tspan = (0,10)

'''
I tried using solve_ivp to integrate the trajectory, but it didn't handle the impact updates well. 
I'm not sure why. I didn't like the idea of using events to stop the integrator and start a new one using
The terminated conditions as initial conditions so I used my own simulation/integration function.
At least this way it's less black boxy...

If you do try to solve using solve_ivp expect the integrator to never finish solving. It will get stuck at the first
impact. Incorporating event handling as I decided I didn't want to do should also be a viable solution.
'''

# These are the dt and N for my simulator/integrator
dt = 0.001
N = int((max(tspan)-min(tspan))/dt)

# These are the dt and N for solve_ivp
#N=500
#dt = (tspan[-1]-tspan[0])/N

tvec = np.linspace(min(tspan),max(tspan),N)
x0 = np.array([0,0.00000000001,-np.pi/2,.00000000001,0,0.0])

#xvec = solve_ivp(dyn,tspan,x0,t_eval = tvec).y  #I could not get solve_ivp to properly handle the impacts
xvec = simulate_impact(dyn,x0,tspan,dt) 

print('Trajectory Computed Successfully')


# In[ ]:


####################################################
# Animate Trajectories
####################################################


x1 = np.zeros(N)
y1 = np.zeros(N)
x2 = np.zeros(N)
y2 = np.zeros(N)
x3 = np.zeros(N)
y3 = np.zeros(N)
x4 = np.zeros(N)
y4 = np.zeros(N)

lx1 = np.zeros(N)
ly1 = np.zeros(N)
lx2 = np.zeros(N)
ly2 = np.zeros(N)
lx3 = np.zeros(N)
ly3 = np.zeros(N)

cx1 = np.zeros(N)
cy1 = np.zeros(N)
cx2 = np.zeros(N)
cy2 = np.zeros(N)



d35 = 35/180*np.pi

for i in range(N):
    x1[i] = xvec[0][i]-l/2*np.cos(xvec[2][i])
    y1[i] = l*np.cos(d35)-l/2*np.sin(xvec[2][i])
    x2[i] = xvec[0][i]+l/2*np.cos(xvec[2][i])
    y2[i] = l*np.cos(d35)+l/2*np.sin(xvec[2][i])
    x3[i] = x2[i]+l/2*np.cos(xvec[4][i])
    y3[i] = y2[i]+l/2*np.sin(xvec[4][i])
    x4[i] = x2[i]-l/2*np.cos(xvec[4][i])
    y4[i] = y2[i]-l/2*np.sin(xvec[4][i])

    lx1[i] = xvec[0][i]-l*np.sin(d35)
    ly1[i] = 0
    lx2[i] = xvec[0][i]
    ly2[i] = l*np.cos(d35)
    lx3[i] = xvec[0][i]+l*np.sin(d35)
    ly3[i] = 0


    cx1[i] = xvec[0][i]-l
    cy1[i] = 0
    cx2[i] = xvec[0][i]+l
    cy2[i] = 0
  

rect_array = [x1,y1,x2,y2,x3,y3,x4,y4,lx1,ly1,lx2,ly2,lx3,ly3,cx1,cy1,cx2,cy2]
print('Animating Trajectory...')
animate(rect_array)