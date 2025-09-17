import cave


class PatrolPlatform(cave.Component):
        """Moves its entity along named waypoints with optional waits and debug drawing."""

        waypointRootName = "Waypoints"
        includeCurrentPosition = True
        speed = 3.0
        waitTime = 0.25
        loop = True
        pingPong = False
        alignToMotion = True
        debugPath = True
        debugColor = cave.Vector3(0.2, 0.8, 1.0)

        def start(self, scene: cave.Scene):
                self.scene = scene
                self.transform = self.entity.getTransform()
                self._waitTimer = cave.SceneTimer()
                self._waiting = False

                positions = self._collect_waypoints(scene)
                if self.includeCurrentPosition:
                        start_pos = self.transform.getWorldPosition().copy()
                        positions.insert(0, start_pos)
                        start_index = 1 if len(positions) > 1 else 0
                else:
                        start_index = 0

                self._path = [pos.copy() for pos in positions]
                self._currentIndex = start_index
                self._direction = 1
                self._completed = len(self._path) <= 1

        def _collect_waypoints(self, scene: cave.Scene):
                positions = []
                holder = None

                if self.waypointRootName:
                        holder = self.entity.getChild(self.waypointRootName)
                        if not holder and scene:
                                holder = scene.get(self.waypointRootName)

                if holder:
                        for child in holder.getChildren():
                                positions.append(child.getTransform().getWorldPosition().copy())

                if not positions:
                        positions.append(self.entity.getTransform().getWorldPosition().copy())

                return positions

        def _advance_index(self):
                if len(self._path) <= 1:
                        self._completed = True
                        return

                next_index = self._currentIndex + self._direction

                if self.pingPong and len(self._path) > 1:
                        if next_index >= len(self._path):
                                self._direction = -1
                                next_index = len(self._path) - 2
                        elif next_index < 0:
                                self._direction = 1
                                next_index = 1
                else:
                        if next_index >= len(self._path):
                                if self.loop:
                                        next_index = 0
                                else:
                                        self._completed = True
                                        next_index = len(self._path) - 1
                        elif next_index < 0:
                                if self.loop:
                                        next_index = len(self._path) - 1
                                else:
                                        self._completed = True
                                        next_index = 0

                self._currentIndex = cave.math.clamp(next_index, 0, len(self._path) - 1)

        def _move_towards_target(self, dt: float):
                if self._completed or self.speed <= 0.0:
                        return

                if self._waiting:
                        if self._waitTimer.get() >= self.waitTime:
                                self._waiting = False
                                self._waitTimer.reset()
                                self._advance_index()
                        else:
                                return

                if self._completed:
                        return

                target = self._path[self._currentIndex]
                current = self.transform.getWorldPosition()
                delta = target - current
                distance = delta.length()

                if distance <= 0.01:
                        if self.waitTime > 0.0:
                                self._waiting = True
                                self._waitTimer.reset()
                        else:
                                self._advance_index()
                        return

                direction = delta.copy()
                if direction.length() > 0.0001:
                        direction.normalize()

                move_amount = self.speed * dt
                if move_amount >= distance:
                        new_position = target.copy()
                else:
                        new_position = current + direction * move_amount

                self.transform.setWorldPosition(new_position)
                self.entity.submitTransformToWorld()

                if self.alignToMotion and direction.length() > 0.0001:
                        planar = cave.Vector3(direction.x, 0.0, direction.z)
                        if planar.length() > 0.0001:
                                planar.normalize()
                                look_point = cave.Vector3(
                                        new_position.x + planar.x,
                                        new_position.y,
                                        new_position.z + planar.z,
                                )
                                self.transform.lookAtPosition(look_point, cave.Vector3(0.0, 1.0, 0.0))

        def update(self):
                dt = cave.getDeltaTime()
                self._move_towards_target(dt)

        def editorUpdate(self):
                if not self.debugPath:
                        return

                scene = cave.getScene()
                if not scene:
                        return

                positions = self._collect_waypoints(scene)
                if self.includeCurrentPosition:
                        positions.insert(0, self.entity.getTransform().getWorldPosition().copy())

                if len(positions) < 2:
                        return

                for i in range(len(positions) - 1):
                        scene.addDebugLine(positions[i], positions[i + 1], self.debugColor)

                if self.loop and not self.pingPong:
                        scene.addDebugLine(positions[-1], positions[0], self.debugColor)
