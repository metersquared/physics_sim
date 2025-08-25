# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from physics_sim.body import Body
from physics_sim.mechanics import UniformGField
from physics_sim.system import System
from physics_sim.solver import ExplicitEulerSolver

# %%
time = 30
mass = 1
frame_rate = 30
dt = 1 / frame_rate

system = System(name="Projectile Motion System", dim=2, bounding_box=[[0, 40], [0, 30]])
print(system.bounding_box)
pm = Body(mass=mass, dim=2, x=np.array([1.0, 10.0]), v=np.array([20.0, 10.0]))

system.add_body(pm)

gravity = UniformGField(name="Uniform Gravity Field")

system.add_field(gravity)
# %%
state_store = ExplicitEulerSolver(system, dt, time)
print(state_store[system.bodies[0]].x[:, 0])
# %%
import matplotlib as mpl

mpl.rcParams["text.usetex"] = True  # Enable LaTeX
mpl.rcParams["font.family"] = "serif"

fig = plt.figure(figsize=(5, 4))
fig.patch.set_facecolor("darkslategray")  # Set full figure background

ax = fig.add_subplot(autoscale_on=False, xlim=(0, 40), ylim=(0, 30))
ax.set_facecolor("darkslategray")  # Match axes background

# Set white ticks and labels
ax.tick_params(colors="white", labelsize=12)
ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")

# Set white spines
for spine in ax.spines.values():
    spine.set_color("white")

# Axis labels (in LaTeX)
ax.set_xlabel(r"$x$ (m)", fontsize=14)
ax.set_ylabel(r"$y$ (m)", fontsize=14)

# Use a high-contrast color for visibility
scat = ax.scatter(
    state_store[system.bodies[0]].x[0, 0],
    state_store[system.bodies[0]].x[1, 0],
    c="white",
)


def animate(i):
    scat.set_offsets(
        (
            state_store[system.bodies[0]].x[0, i],
            state_store[system.bodies[0]].x[1, i],
        )
    )
    return (scat,)


ani = anim.FuncAnimation(
    fig,
    animate,
    repeat=True,
    interval=dt * 1000,
    blit=True,
    frames=len(state_store[system.bodies[0]].x[0]),
)

# No transparency — just a baked-in dark green background
writer = anim.FFMpegWriter(fps=frame_rate, metadata=dict(artist="Arya"), bitrate=1800)
ani.save("projectile_motion.mp4", writer=writer)


plt.close()
# %%
