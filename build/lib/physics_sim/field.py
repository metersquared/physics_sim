from physics_sim.system import System


class Field:
    """A class representing a field in the simulation."""

    def __init__(self, system: System = None, name=None):
        """Initialize a field in the simulation.

        Parameters
        ----------
        system : System
            The system to which this field belongs.
        name : str, optional
            The name of the field, by default None
        """
        super(Field, self).__init__()
        self.name = name if name is not None else "Default Field"
        self.system = system

    def force(self, body):
        """Calculate the force on a body due to this field."""
        raise NotImplementedError("This method should be overridden by subclasses")
