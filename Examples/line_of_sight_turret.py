import cave


class LineOfSightTurret(cave.Component):
        """Rotating turret that tracks tagged targets, raycasts, and fires projectiles."""

        targetTag = "Player"
        targetName = ""
        reacquireInterval = 0.35
        maxDistance = 25.0
        rotationSpeed = 6.0
        fireInterval = 0.75
        muzzleName = "Muzzle"
        targetOffset = cave.Vector3(0.0, 1.2, 0.0)
        projectileTemplate = ""
        projectileSpeed = 28.0
        projectileLifetime = 6.0
        fireSound = ""
        raycastMaskBit = -1
        requireLineOfSight = True
        drawDebugAim = False
        debugColor = cave.Vector3(1.0, 0.35, 0.15)

        def start(self, scene: cave.Scene):
                self.scene = scene
                self.transform = self.entity.getTransform()
                self.muzzleEntity = self.entity.getChild(self.muzzleName) if self.muzzleName else None
                self.muzzleTransform = self.muzzleEntity.getTransform() if self.muzzleEntity else self.transform

                self.fireTimer = cave.SceneTimer()
                self.retargetTimer = cave.SceneTimer()
                self.fireTimer.set(self.fireInterval)

                self.target = None
                self._raycastMask = None

                if self.raycastMaskBit >= 0:
                        mask = cave.BitMask(False)
                        mask.enable(self.raycastMaskBit)
                        self._raycastMask = mask

        def _muzzle_position(self) -> cave.Vector3:
                return self.muzzleTransform.getWorldPosition()

        def _target_position(self, entity: cave.Entity) -> cave.Vector3:
                return entity.getTransform().getWorldPosition() + self.targetOffset

        def _has_line_of_sight(self, target_pos: cave.Vector3, entity: cave.Entity = None) -> bool:
                if not self.requireLineOfSight:
                        return True

                origin = self._muzzle_position()
                if self._raycastMask:
                        result = self.scene.rayCast(origin, target_pos, self._raycastMask)
                else:
                        result = self.scene.rayCast(origin, target_pos)

                if not result.hit:
                        return True

                if entity and result.entity == entity:
                        return True

                if result.entity == self.entity and entity is None:
                        return True

                return False

        def _is_target_valid(self, entity: cave.Entity) -> bool:
                if not entity or not entity.isAlive() or not entity.isActive():
                        return False

                muzzle_pos = self._muzzle_position()
                target_pos = self._target_position(entity)
                if (target_pos - muzzle_pos).length() > self.maxDistance:
                        return False

                return self._has_line_of_sight(target_pos, entity)

        def _acquire_target(self) -> cave.Entity:
                scene = self.scene if hasattr(self, "scene") else cave.getScene()
                if not scene:
                        return None

                muzzle_pos = self._muzzle_position()
                best_entity = None
                best_distance = self.maxDistance

                if self.targetName:
                        candidate = scene.get(self.targetName)
                        if candidate and candidate != self.entity and self._is_target_valid(candidate):
                                return candidate

                candidates = scene.getEntitiesWithTag(self.targetTag) if self.targetTag else []
                for entity in candidates:
                        if entity == self.entity or not entity.isActive():
                                continue

                        target_pos = self._target_position(entity)
                        distance = (target_pos - muzzle_pos).length()
                        if distance > self.maxDistance:
                                continue

                        if not self._has_line_of_sight(target_pos, entity):
                                continue

                        if best_entity is None or distance < best_distance:
                                best_entity = entity
                                best_distance = distance

                return best_entity

        def _update_target(self):
                if self.target and not self._is_target_valid(self.target):
                        self.target = None

                if self.retargetTimer.get() >= self.reacquireInterval or not self.target:
                        self.retargetTimer.reset()
                        new_target = self._acquire_target()
                        if new_target:
                                self.target = new_target

        def _aim_at_target(self, dt: float):
                if not self.target:
                        return

                muzzle_pos = self._muzzle_position()
                aim_pos = self._target_position(self.target)

                if self.drawDebugAim:
                        self.scene.addDebugLine(muzzle_pos, aim_pos, self.debugColor)

                direction = aim_pos - muzzle_pos
                if direction.length() <= 0.0001:
                        return

                direction.normalize()
                look_point = muzzle_pos + direction

                lerp = cave.math.clamp(self.rotationSpeed * dt, 0.0, 1.0)
                self.transform.lookAtPositionSmooth(look_point, lerp, cave.Vector3(0.0, 1.0, 0.0))
                self.entity.submitTransformToWorld()

        def _spawn_projectile(self):
                muzzle_pos = self._muzzle_position()
                forward = self.muzzleTransform.getForwardVector(True)

                if forward.length() <= 0.0001:
                        if self.target:
                                forward = self._target_position(self.target) - muzzle_pos
                        else:
                                forward = cave.Vector3(0.0, 0.0, -1.0)

                if forward.length() > 0.0001:
                        forward.normalize()

                if self.projectileTemplate:
                        projectile = self.scene.addFromTemplate(self.projectileTemplate, muzzle_pos.copy())
                        projectile_transform = projectile.getTransform()
                        projectile_transform.lookAtPosition(muzzle_pos + forward)
                        projectile.submitTransformToWorld()

                        if self.projectileSpeed > 0.0:
                                rb = projectile.get("Rigid Body")
                                if isinstance(rb, cave.RigidBodyComponent):
                                        velocity = forward * self.projectileSpeed
                                        rb.setLinearVelocity(velocity.x, velocity.y, velocity.z)

                        if self.projectileLifetime > 0.0:
                                projectile.scheduleKill(self.projectileLifetime)
                else:
                        self.scene.addDebugLine(
                                muzzle_pos,
                                muzzle_pos + forward * (self.maxDistance * 0.5),
                                self.debugColor,
                        )

                if self.fireSound:
                        cave.playSound(self.fireSound)

        def _try_fire(self):
                if not self.target or self.fireInterval <= 0.0:
                        return

                if self.fireTimer.get() < self.fireInterval:
                        return

                aim_pos = self._target_position(self.target)
                if not self._has_line_of_sight(aim_pos, self.target):
                        return

                self.fireTimer.reset()
                self._spawn_projectile()

        def update(self):
                dt = cave.getDeltaTime()
                self._update_target()

                if not self.target:
                        return

                self._aim_at_target(dt)
                self._try_fire()
