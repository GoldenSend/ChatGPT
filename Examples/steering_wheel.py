import cave


class SteeringWheelController(cave.Component):
        """Rotates a steering wheel mesh with A/D input and clamps to a maximum angle."""

        maxAngle = 450.0
        turnSpeed = 360.0
        autoCenter = True
        autoCenterSpeed = 120.0

        def start(self, scene: cave.Scene):
                self.transform = self.entity.getTransform()
                self._baseEuler = self.transform.getEuler().copy()
                self._currentAngle = 0.0

        def _apply_input(self, dt: float):
                events = cave.getEvents()

                direction = 0.0
                if events.active(cave.event.KEY_A):
                        direction -= 1.0
                if events.active(cave.event.KEY_D):
                        direction += 1.0

                if direction != 0.0:
                        self._currentAngle += direction * self.turnSpeed * dt
                        self._currentAngle = cave.math.clamp(
                                self._currentAngle,
                                -self.maxAngle,
                                self.maxAngle,
                        )
                        return True

                if not self.autoCenter or abs(self._currentAngle) <= 0.0:
                        return False

                step = self.autoCenterSpeed * dt
                if self._currentAngle > 0.0:
                        self._currentAngle = max(0.0, self._currentAngle - step)
                else:
                        self._currentAngle = min(0.0, self._currentAngle + step)

                return False

        def _apply_rotation(self):
                new_euler = self._baseEuler.copy()
                new_euler.z += self._currentAngle
                self.transform.setEuler(new_euler)
                self.entity.submitTransformToWorld()

        def update(self):
                dt = cave.getDeltaTime()
                self._apply_input(dt)
                self._apply_rotation()

        def editorUpdate(self):
                if not hasattr(self, "transform"):
                        self.transform = self.entity.getTransform()
                        self._baseEuler = self.transform.getEuler().copy()
