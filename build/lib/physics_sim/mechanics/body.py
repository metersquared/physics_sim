from physics_sim.mechanics.field import GravitationalField
from physics_sim.body import Body


class MassiveBody(Body):
    """A massive body generating a gravitational field."""

    def __init__(
        self, mass, dim, x=None, v=None, system=None, G=6.67430e-11, epsilon=1e7
    ):
        """Initialize a massive body.

        Parameters
        ----------
        mass : float
            The mass of the body.
        dim : int
            The dimension of the body.
        x : np.ndarray, optional
            The position of the body, by default None
        v : np.ndarray, optional
            The velocity of the body, by default None
        system : System, optional
            The system to which this body belongs, by default None
        """
        super(MassiveBody, self).__init__(mass=mass, dim=dim, x=x, v=v, system=system)
        self.fields.append(
            GravitationalField(body=self, system=system, G=G, epsilon=epsilon)
        )
