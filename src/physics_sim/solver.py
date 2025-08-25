from physics_sim.system import System
import numpy as np
from dataclasses import dataclass


@dataclass
class BodyState:
    x: np.ndarray  # Shape: (dim, steps)
    v: np.ndarray  # Shape: (dim, steps)


class StateStore:
    def __init__(self):
        self._data = {}

    def __getitem__(self, body):
        return self._data[body]

    def __setitem__(self, body, state: BodyState):
        self._data[body] = state

    def items(self):
        return self._data.items()

    def bodies(self):
        return self._data.keys()


def ExplicitEulerSolver(system: System, dt: float, t: float) -> dict:
    """Explicit Euler solver for the system.

    Parameters
    ----------
    system : System
        The system to solve.
    dt : float
        Time step.
    t : float
        Current time.

    Returns
    -------
    dict
        Dictionary containing the state of the system at the next time step.
    """
    ts = np.linspace(0, t, int(t / dt) + 1)
    state_store = StateStore()

    for body in system.bodies:
        # Initialize state for each body
        x = np.zeros((system.dim, len(ts)))
        v = np.zeros((system.dim, len(ts)))
        x[:, 0] = body.x.copy()
        v[:, 0] = body.v.copy()
        state_store[body] = BodyState(x=x, v=v)

    for i, time in enumerate(ts):
        if i == 0:
            continue
        # Update position and velocity for each body

        for body in system.bodies:
            # simulate forward using local temp variables only
            x_prev = state_store[body].x[:, i - 1]
            v_prev = state_store[body].v[:, i - 1]

            new_x = x_prev + dt * v_prev
            new_v = v_prev.copy()

            # Apply boundary reflections
            bound_min, bound_max = system.bounding_box[:, 0], system.bounding_box[:, 1]

            below_min = new_x < bound_min
            above_max = new_x > bound_max

            new_v[below_min] *= -1
            new_x[below_min] = bound_min[below_min] + (
                bound_min[below_min] - new_x[below_min]
            )

            new_v[above_max] *= -1
            new_x[above_max] = bound_max[above_max] - (
                new_x[above_max] - bound_max[above_max]
            )

            # Apply forces
            for field in system.all_fields(body):
                force = field.force(body)
                new_v += dt * force / body.mass

            # Store
            state_store[body].x[:, i] = new_x
            state_store[body].v[:, i] = new_v

            # Optionally update body.x and body.v persistently (depends on how your system uses them)
            body.x = new_x
            body.v = new_v

    return state_store
