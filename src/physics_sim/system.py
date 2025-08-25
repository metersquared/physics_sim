import numpy as np
from typing import List, Tuple


class System:
    """A class representing a physical system."""

    def __init__(self, name, dim, bounding_box: List[List[float]] = None):
        """Initialize the system.

        Parameters
        ----------
        name : str
            The name of the system.
        dim : int
            The dimension of the system.
        bounding_box : List[List[float]], optional
            The bounding box of the system, by default None
        """
        self.name = name
        self.dim = dim
        self.bodies = []
        self.fields = []
        if bounding_box is None:
            self.set_bounding_box([[-np.inf, np.inf]] * dim)
        else:
            self.set_bounding_box(bounding_box)

    def set_bounding_box(self, bounding_box: List[List[float]]):
        """Set the bounding box of the system.

        Parameters
        ----------
        bounding_box : List[List[float]]
            The bounding box of the system.
        """
        self.bounding_box = np.array(bounding_box)

    def add_body(self, body):
        """Add a body to the system.

        Parameters
        ----------
        body : Body
            The body to add to the system.

        Raises
        ------
        ValueError
            If the body is already part of another system or if the dimensions do not match.
        """
        from physics_sim.body import Body  # late import to avoid circular import

        assert isinstance(body, Body), "Expected a Body instance"
        assert body.dim == self.dim, "Body dimension must match system dimension"
        if body.system is not None and body.system != self:
            raise ValueError("Body already belongs to another system")
        if body not in self.bodies:
            self.bodies.append(body)
        if body.system is None:
            body.system = self

    def add_field(self, field):
        """Add a field to the system.

        Parameters
        ----------
        field : Field
            The field to add to the system.
        """
        from physics_sim.field import Field

        assert isinstance(field, Field), "Expected a Field instance"
        assert (
            field.system is None or field.system == self
        ), "Field already belongs to another system"
        if field.system is None:
            field.system = self
        if field not in self.fields:
            self.fields.append(field)

    def all_fields(self, body):
        """Get all fields in the system, induced on a body.

        Returns
        -------
        List[Field]
            The list of fields in the system.
        """
        fields = list(self.fields)
        for b in self.bodies:
            if b is not body:
                fields.extend(b.fields)
        return fields

    def run(self):
        print(f"Running system: {self.name}")
        for body in self.bodies:
            body.update()
