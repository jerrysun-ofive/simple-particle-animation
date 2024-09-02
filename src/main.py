import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import RK45
from matplotlib import animation, colors, colormaps
from simulation import *

DT = 1e-19 # time step in seconds
N = 3 # number of particles
SIM_LEN = 5000 # number of steps to simulate
E_PLOT_N = 100 # number of X and Y values in the electric field plot
SIM_SPEED = 8 # number of steps to skip in the animation

# setting the initial state of the simulation
# x position, y position, x velocity, y velocity
state = np.zeros((3, 4))
# mass of each particle. Unit: MeV
mass = np.array([938, 0.511, 938])
# charge of each particle. Unit: q_e
q = np.array([1, -1, 1])

# set initial state
dist = 52.9
# starting x and y of the electron
state[1, 0] = dist * np.cos(1.1)
state[1, 1] = dist * np.sin(1.1)
# starting speed
vi = np.sqrt(np.abs(COULOMB_K * q[0] / dist * mass[1]))
# x and y velocity
state[1, 2] = vi * (state[1, 1] - state[0, 1]) / dist
state[1, 3] = vi * (state[1, 0] - state[0, 0]) / dist
# the second proton starts off to the left and moves to the right
state[2, 0] = -190
state[2, 2] = vi * 0.4 
# shift everything to the left a little for better visualization 
state[:, 0] -= 20

simulation = simulate_steps(state, mass, q, DT, SIM_LEN)

# custome colormap for positive and negative charges
seismic = colormaps['seismic'].resampled(255)
newcolor = seismic(np.linspace(0, 1, 255))
for i in range(3):
    newcolor[:127, i] -= np.linspace(0, 80, 127)/256
    newcolor[127: , i] -= np.linspace(80, 0, 128)/256

newcolor[np.where(newcolor < 0)] = 0
cmap = colors.ListedColormap(newcolor)

# calculate initial electric field
bound = dist * 3
x = np.linspace(-bound, bound, E_PLOT_N)
y = np.linspace(-bound, bound, E_PLOT_N)
X, Y = np.meshgrid(x, y)
Ex, Ey = E_field(state, q, bound, E_PLOT_N)
E_strength = np.log(Ex**2 + Ey**2 + EPS)

# plot initial particles
fig = plt.figure()
mesh = plt.pcolormesh(X, Y, E_strength, cmap='inferno')
scatter = plt.scatter(state[:, 0], state[:, 1], s=np.log(mass/np.min(mass) + 1) * 15, c=q, cmap='seismic', vmin=-2, vmax=2)
axs = fig.get_axes()

def animate_func(i):
    Ex, Ey = E_field(simulation[i * SIM_SPEED], q, bound, E_PLOT_N)
    E_strength = np.log(Ex**2 + Ey**2)
    mesh.set_array(E_strength)
    scatter.set_offsets(simulation[i * SIM_SPEED])
    return scatter, mesh

anim = animation.FuncAnimation(
    fig, animate_func, frames=range(SIM_LEN//SIM_SPEED), interval=40
)

axs[0].set_xlim(-bound, bound)
axs[0].set_ylim(-bound, bound)
fig.set_size_inches(6, 6)
fig.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, wspace = None, hspace = None)
plt.gca().set_aspect('equal')
plt.axis('off')

plt.show()







