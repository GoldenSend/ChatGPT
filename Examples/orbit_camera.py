import cave


class OrbitCameraController(cave.Component):
        """Smooth third-person orbit camera that reacts to the mouse and scroll wheel."""

        targetName = ""
        autoAnchorParent = True
        focusOffset = cave.Vector3(0.0, 1.6, 0.0)
        distance = 6.0
        minDistance = 2.5
        maxDistance = 14.0
        zoomSpeed = 2.0
        mouseSensitivity = cave.Vector2(0.2, 0.16)
        invertY = False
        pitchMin = 10.0
        pitchMax = 80.0
        smoothing = 0.18
        lockCursorWhileOrbiting = True

        def start(self, scene: cave.Scene):
                self.scene = scene
                self.transform = self.entity.getTransform()
                self.anchorEntity = None
                self.anchorTransform = None

                self._yaw = self.transform.getEuler().y
                self._pitch = cave.math.clamp(self.transform.getEuler().x, self.pitchMin, self.pitchMax)

                self._currentDistance = cave.math.clamp(self.distance, self.minDistance, self.maxDistance)
                self._desiredDistance = self._currentDistance

                self._findAnchor()

        def _findAnchor(self):
                scene = self.scene if hasattr(self, "scene") else cave.getScene()
                anchor = None

                if self.targetName:
                        anchor = scene.get(self.targetName)

                if not anchor and self.autoAnchorParent:
                        anchor = self.entity.getParent()

                if not anchor:
                        anchor = self.entity

                self.anchorEntity = anchor
                self.anchorTransform = anchor.getTransform() if anchor else self.transform

        def _get_focus_position(self) -> cave.Vector3:
                if not self.anchorTransform or not self.anchorEntity or not self.anchorEntity.isAlive():
                        self._findAnchor()

                return self.anchorTransform.getWorldPosition() + self.focusOffset

        def _get_smoothing_factor(self, dt: float) -> float:
                smooth = cave.math.clamp(self.smoothing, 0.0, 0.99)
                if smooth <= 0.0:
                        return 1.0
                # Convert the smoothing value (0..1) into a framerate-independent lerp weight.
                return 1.0 - cave.math.pow(1.0 - smooth, dt * 60.0)

        def _apply_mouse_input(self):
                events = cave.getEvents()
                orbiting = events.active(cave.event.MOUSE_RIGHT)
                events.setRelativeMouse(orbiting and self.lockCursorWhileOrbiting)

                if not orbiting:
                        return

                motion = events.getMouseMotion()
                invert = -1.0 if self.invertY else 1.0

                self._yaw -= motion.x * self.mouseSensitivity.x
                self._pitch += motion.y * self.mouseSensitivity.y * invert
                self._pitch = cave.math.clamp(self._pitch, self.pitchMin, self.pitchMax)

        def _apply_zoom(self):
                scroll = cave.getEvents().getMouseScroll()
                if scroll == 0.0:
                        return

                self._desiredDistance = cave.math.clamp(
                        self._desiredDistance - scroll * self.zoomSpeed,
                        self.minDistance,
                        self.maxDistance,
                )

        def _update_camera_transform(self, dt: float):
                focus = self._get_focus_position()

                yaw_rad = cave.math.toRadians(self._yaw)
                pitch_rad = cave.math.toRadians(self._pitch)

                cos_pitch = cave.math.cos(pitch_rad)
                forward = cave.Vector3(
                        cave.math.sin(yaw_rad) * cos_pitch,
                        cave.math.sin(pitch_rad),
                        cave.math.cos(yaw_rad) * cos_pitch,
                )

                self._currentDistance = cave.math.lerp(
                        self._currentDistance,
                        self._desiredDistance,
                        self._get_smoothing_factor(dt),
                )

                desired_position = focus - forward * self._currentDistance

                current_position = self.transform.getWorldPosition()
                new_position = cave.math.lerp(
                        current_position,
                        desired_position,
                        self._get_smoothing_factor(dt),
                )

                self.transform.setWorldPosition(new_position)
                self.transform.lookAtPosition(focus)
                self.entity.submitTransformToWorld()

        def update(self):
                dt = cave.getDeltaTime()
                self._apply_mouse_input()
                self._apply_zoom()
                self._update_camera_transform(dt)

        def editorUpdate(self):
                # Keep the anchor updated while adjusting things in the editor.
                if not hasattr(self, "anchorTransform"):
                        self._findAnchor()
