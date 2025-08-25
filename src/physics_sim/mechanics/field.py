from physics_sim.field import Field
from physics_sim.body import Body
import numpy as np


class UniformGField(Field):
    """A uniform gravitational field."""

    def __init__(self, g=9.8, system=None, name=None):
        """Initialize a uniform gravitational field.

        Parameters
        ----------
        g : float, optional
            The acceleration due to gravity, by default 9.8
        system : System, optional
            The system to which this field belongs, by default None
        name : str, optional
            The name of the field, by default None
        """
        super(UniformGField, self).__init__(system=system, name=name)
        self.g = g

    def force(self, body):
        """Calculate the force on a body due to this field."""
        force = np.zeros(self.system.dim)
        force[1] = -body.mass * self.g
        return force


class GravitationalField(Field):
    """A gravitational field source of a mass."""

    def __init__(self, body: Body, G=6.67430e-11, epsilon=1e7, system=None, name=None):
        """Initialize a gravitational field.

        Parameters
        ----------
        mass : float
            The mass that generates the gravitational field.
        G : float, optional
            The gravitational constant, by default 6.67430e-11.
        system : System, optional
            The system to which this field belongs, by default None
        name : str, optional
            The name of the field, by default None
        """
        super(GravitationalField, self).__init__(system=system, name=name)
        self.body = body
        self.G = G
        self.epsilon = epsilon

    def force(self, body):
        """Calculate the gravitational force on a body.

        Parameters
        ----------
        body : Body
            The body experiencing the force.
        epsilon : float, optional
            A small value to prevent singularities, by default 1e7

        Returns
        -------
        np.ndarray
            The gravitational force vector.
        """
        x1 = self.body.x
        x2 = body.x
        r_vec = x2 - x1
        r = np.linalg.norm(r_vec)
        r_soft = np.sqrt(r**2 + self.epsilon**2)
        return -self.G * self.body.mass * body.mass * r_vec / r_soft**3
