# Physics_SIM

This is a library to perform Molecular Dynamics simulation of particles in Physics. This is a personal project to review my Physics while applying numerical methods.

## Installation

Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

Go to the root directory of this library (where pyproject.toml is located), and install with pip

```
pip install .
```

Use option `-e` for editable install.

## Usage

The library takes care the heavy-lifting of numerics, so that one can focus on setting up the physical system that are studied.

Classes:

- System : Stores an entire system with optional boundaries, dimension and so on.
- Field : Provide a potential that interacts with the body in the system.
- Body : Represents a particle, may contain mass and charge. Can also generate field.

## Example

Example of its usage are shown in the example directory.

<figure class="video_container">
  <iframe src="example/three_body_problem.mp4" frameborder="0" allowfullscreen="true"> 
</iframe>
</figure>
