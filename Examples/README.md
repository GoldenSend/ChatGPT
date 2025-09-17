# Cave Engine Python example scripts

This folder contains standalone gameplay components that can be added to an entity in a Cave Engine scene. Each script is self-contained and demonstrates different engine systems such as input, physics queries, timers, and entity factories.

- `orbit_camera.py` – Implements a third-person orbit camera with smooth mouse look and scroll wheel zooming.
- `patrol_platform.py` – Moves an entity along configurable waypoints, pausing at each stop and optionally drawing the path in the editor.
- `line_of_sight_turret.py` – Tracks the closest target with a matching tag, performs line-of-sight checks, and fires projectiles or debug shots on a cadence.
- `steering_wheel.py` – Turns a steering wheel mesh with A/D input while clamping the rotation to realistic stop angles.

Copy the scripts into your project and attach the desired component to an entity via the Cave editor. Customize the exposed class attributes to match your scene setup.
