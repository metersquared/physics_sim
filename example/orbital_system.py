# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from physics_sim.mechanics import MassiveBody
from physics_sim.system import System
from physics_sim.solver import ExplicitEulerSolver


# %%

years = 2
day = 24 * 60 * 60  # seconds in a day
sim_duration = years * 365 * day
animation_duration = 30  # seconds
animation_to_sim = sim_duration / animation_duration
frame_rate = 30
frame_skip = 500  # Adjust this to control the number of frames in the animation
dt = animation_to_sim / (frame_rate * frame_skip)

# %%
system = System(name="Orbital System", dim=2)
print(system.bounding_box)

sun = MassiveBody(mass=1.989e30, dim=2, x=np.array([0.0, 0.0]), v=np.array([0.0, 0.0]))
earth = MassiveBody(
    mass=5.972e24,
    dim=2,
    x=np.array([1.496e11, 0.0]),
    v=np.array([0.0, 29780.0]),
)
moon = MassiveBody(
    mass=7.34767309e22,
    dim=2,
    x=np.array([1.496e11 + 3.844e8, 0.0]),
    v=np.array([0.0, 29780.0 + 1022.0]),
)
system.add_body(sun)
system.add_body(earth)
system.add_body(moon)

print(system.bodies)
state_store = ExplicitEulerSolver(system, dt, sim_duration)
print(state_store[earth])  # Earth position

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
fig.patch.set_facecolor("darkslategray")

# === Global View ===
ax1.set_facecolor("darkslategray")
ax1.set_xlim(-2e11, 2e11)
ax1.set_ylim(-2e11, 2e11)
ax1.set_xlabel(r"$x$ (m)", fontsize=14, color="white")
ax1.set_ylabel(r"$y$ (m)", fontsize=14, color="white")
ax1.tick_params(colors="white", labelsize=12)
for spine in ax1.spines.values():
    spine.set_color("white")

scat_sun = ax1.scatter([], [], c="yellow", s=100, label="Sun")
scat_earth = ax1.scatter([], [], c="blue", s=10)
scat_moon = ax1.scatter([], [], c="gray", s=5)

# === Zoomed View: Earth-Centered ===
ax2.set_facecolor("darkslategray")
zoom_radius = 5e8  # +-500,000 km around Earth
ax2.set_xlim(-zoom_radius, zoom_radius)
ax2.set_ylim(-zoom_radius, zoom_radius)
ax2.set_xlabel(r"$x_\mathrm{local}$ (m)", fontsize=14, color="white")
ax2.set_ylabel(r"$y_\mathrm{local}$ (m)", fontsize=14, color="white")
ax2.tick_params(colors="white", labelsize=12)
for spine in ax2.spines.values():
    spine.set_color("white")

scat_earth_local = ax2.scatter([], [], c="blue", s=50, label="Earth")
scat_moon_local = ax2.scatter([], [], c="gray", s=30, label="Moon")


def update(frame):
    # === Global view positions ===
    sun_pos = state_store[sun].x[:, frame]
    earth_pos = state_store[earth].x[:, frame]
    moon_pos = state_store[moon].x[:, frame]

    scat_sun.set_offsets(sun_pos)
    scat_earth.set_offsets(earth_pos)
    scat_moon.set_offsets(moon_pos)

    # === Zoomed-in view centered on Earth ===
    moon_local = moon_pos - earth_pos
    scat_earth_local.set_offsets([0, 0])
    scat_moon_local.set_offsets(moon_local)

    return scat_sun, scat_earth, scat_moon, scat_earth_local, scat_moon_local


frame_indices = range(0, len(state_store[earth].x[0]), frame_skip)

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
ani.save("orbital_system.mp4", writer=writer)

# %%
