# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from physics_sim.mechanics import MassiveBody
from physics_sim.system import System
from physics_sim.solver import ExplicitEulerSolver


# %%

sim_duration = 2560
frame_rate = 120
frame_skip = 60  # Adjust this to control the number of frames in the animation
# frame_skip = 500  # Adjust this to control the number of frames in the animation
dt = 1 / (frame_rate)

# %%
rng = np.random.default_rng(42)  # For reproducibility
system = System(name="Three Body Problem", dim=2)

G_val = 10.0  # much stronger gravitational interaction
epsilon_val = 1e-7  # small softening factor for stability

bodies = [
    MassiveBody(
        mass=1.0,
        dim=2,
        x=np.array([0.0, 10.0]),
        v=np.array([-1.0, 0.0]),  # Initial velocity to create a stable orbit
        G=G_val,
        epsilon=epsilon_val,
    ),
    MassiveBody(
        mass=1.0,
        dim=2,
        x=np.array([-10.0, 0.0]),
        v=np.array([0.0, -1.0]),  # Slightly perturbed velocity
        G=G_val,
        epsilon=epsilon_val,
    ),
]

# Third body placed at origin to balance momentum
x = -sum(body.mass * body.x.copy() for body in bodies) / 1.5
v = -sum(body.mass * body.v.copy() for body in bodies) / 1.5

bodies.append(MassiveBody(mass=1.5, dim=2, x=x, v=v, G=G_val, epsilon=epsilon_val))

# Duplicate the bodies for a second simulation with a velocity perturbation on body 1
bodies_perturbed = [
    MassiveBody(
        mass=body.mass,
        dim=2,
        x=body.x.copy(),
        v=(body.v.copy() + (body.v.copy() * 0.01 if i == 1 else np.array([0.0, 0.0]))),
        G=G_val,
        epsilon=epsilon_val,
    )
    for i, body in enumerate(bodies)
]

# Create second system and run simulation
system_perturbed = System(name="Perturbed Three Body Problem", dim=2)
for body in bodies_perturbed:
    system_perturbed.add_body(body)

for i, body in enumerate(bodies):
    system.add_body(body)

state_store = ExplicitEulerSolver(system, dt, sim_duration)

state_store_perturbed = ExplicitEulerSolver(system_perturbed, dt, sim_duration)

# %%
print(len(state_store[bodies[0]].x[0]), "steps")
# %%
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"

fig = plt.figure(figsize=(6, 5))
fig.patch.set_facecolor("darkslategray")

colors = ["r", "g", "b"]

ax = fig.add_subplot(autoscale_on=False)
ax.set_facecolor("darkslategray")
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.set_xlabel(r"$x$ (m)", fontsize=14, color="white")
ax.set_ylabel(r"$y$ (m)", fontsize=14, color="white")
ax.tick_params(colors="white", labelsize=12)
for spine in ax.spines.values():
    spine.set_color("white")

# === Original system ===
scat_orig = [
    ax.scatter([], [], s=50, color=colors[i], label=f"Body {i+1}")
    for i in range(len(bodies))
]
lines_orig = [
    ax.plot([], [], color=colors[i], lw=1, alpha=0.7, ls="-")[0]
    for i, sc in enumerate(scat_orig)
]

# === Perturbed system ===
scat_pert = [
    ax.scatter([], [], s=50, color=colors[i], marker="D")
    for i in range(len(bodies_perturbed))
]
lines_pert = [
    ax.plot([], [], lw=1, alpha=0.7, ls="--", color=colors[i])[0]
    for i, sc in enumerate(scat_pert)
]


def update(frame):
    for i in range(len(bodies)):
        # Original
        pos_orig = state_store[bodies[i]].x[:, frame]
        scat_orig[i].set_offsets(pos_orig)
        lines_orig[i].set_data(
            state_store[bodies[i]].x[0, :frame],
            state_store[bodies[i]].x[1, :frame],
        )

        # Perturbed
        pos_pert = state_store_perturbed[bodies_perturbed[i]].x[:, frame]
        scat_pert[i].set_offsets(pos_pert)
        lines_pert[i].set_data(
            state_store_perturbed[bodies_perturbed[i]].x[0, :frame],
            state_store_perturbed[bodies_perturbed[i]].x[1, :frame],
        )

    return (*scat_orig, *lines_orig, *scat_pert, *lines_pert)


frame_indices = range(0, len(state_store[bodies[0]].x[0]), frame_skip)

ani = anim.FuncAnimation(
    fig,
    update,
    frames=frame_indices,
    interval=1000 / frame_rate,
    blit=True,
    repeat=True,
)

fig.legend(loc="upper center", ncol=3, fontsize=10)
writer = anim.FFMpegWriter(fps=frame_rate, metadata=dict(artist="Arya"), bitrate=1800)
ani.save("three_body_problem.mp4", writer=writer)

# %%
