"""Kinematic collision resolver for player AABB against static obstacles."""
import numpy as np
import config
from physics.aabb import AABB
from world.spatial import candidates


class ContactInfo:
    """Result of one swept movement step."""
    __slots__ = ("on_ground", "wall_normal", "wall_obstacle", "hit_ceiling", "approach_speed", "stepped")

    def __init__(self):
        self.on_ground = False
        self.stepped = False           # walked up a curb / stair step this tick
        self.wall_normal = None        # horizontal unit normal of the wall that was hit (points away from wall)
        self.wall_obstacle = None
        self.hit_ceiling = False
        self.approach_speed = 0.0      # horizontal speed into the wall before it was cancelled


class CollisionSystem:
    def __init__(self):
        pass

    @staticmethod
    def player_box(pos, player_size, shrink=0.0):
        half = np.asarray(player_size, dtype=np.float32) * 0.5 - shrink
        pos = np.asarray(pos, dtype=np.float32)
        return AABB(pos - half, pos + half)

    @staticmethod
    def overlaps_any(box, obstacles, ignore=None):
        """Returns the first obstacle overlapping the box (or None)."""
        for obs in candidates(obstacles, box.min, box.max):
            if obs is ignore:
                continue
            if box.intersects(obs.bounds):
                return obs
        return None

    @staticmethod
    def overlapping(box, obstacles, ignore=None):
        """Every obstacle overlapping the box."""
        return [obs for obs in candidates(obstacles, box.min, box.max)
                if obs is not ignore and box.intersects(obs.bounds)]

    @staticmethod
    def ground_height_below(pos, player_size, obstacles, max_drop=50.0):
        """Highest obstacle top directly under the player's footprint and below the feet.

        Returns None if nothing is found within max_drop.
        """
        half_w = player_size[0] * 0.5 * 0.9
        half_d = player_size[2] * 0.5 * 0.9
        feet_y = pos[1] - player_size[1] * 0.5
        best = None
        lo = (pos[0] - half_w, 0.0, pos[2] - half_d)
        hi = (pos[0] + half_w, 0.0, pos[2] + half_d)
        for obs in candidates(obstacles, lo, hi):
            b = obs.bounds
            if b.max[0] <= pos[0] - half_w or b.min[0] >= pos[0] + half_w:
                continue
            if b.max[2] <= pos[2] - half_d or b.min[2] >= pos[2] + half_d:
                continue
            top = float(b.max[1])
            if top <= feet_y + 1e-3 and top >= feet_y - max_drop:
                if best is None or top > best:
                    best = top
        return best

    @staticmethod
    def resolve_movement(player_pos, player_size, velocity, dt, obstacles):
        """
        Moves the player by velocity * dt and resolves collisions against obstacles.
        Returns:
            new_pos: np.ndarray (center position of the player)
            new_vel: np.ndarray (corrected velocity)
            contact: ContactInfo
        """
        pos = np.asarray(player_pos, dtype=np.float32).copy()
        vel = np.asarray(velocity, dtype=np.float32).copy()
        half_w = player_size[0] * 0.5
        half_h = player_size[1] * 0.5
        half_d = player_size[2] * 0.5

        contact = ContactInfo()

        # Broadphase: only obstacles near the swept volume of this step
        sweep = np.abs(vel) * dt + 0.1
        lo = (pos[0] - half_w - sweep[0], 0.0, pos[2] - half_d - sweep[2])
        hi = (pos[0] + half_w + sweep[0], 0.0, pos[2] + half_d + sweep[2])
        obstacles = candidates(obstacles, lo, hi)

        def make_box():
            return AABB(
                [pos[0] - half_w, pos[1] - half_h, pos[2] - half_d],
                [pos[0] + half_w, pos[1] + half_h, pos[2] + half_d]
            )

        can_step = float(vel[1]) <= 0.01       # grounded or falling: curbs and stairs are walked over

        def try_step_up(obs):
            """Lift the body onto a low obstacle it walked into (stairs, curbs, bumpers)."""
            if not can_step:
                return False
            lift = float(obs.bounds.max[1]) - (float(pos[1]) - half_h)
            if lift <= 0.0 or lift > config.STEP_HEIGHT:
                return False
            trial_y = float(pos[1]) + lift + 0.002
            box = AABB([pos[0] - half_w, trial_y - half_h, pos[2] - half_d],
                       [pos[0] + half_w, trial_y + half_h, pos[2] + half_d])
            for other in obstacles:
                if box.intersects(other.bounds):
                    return False
            pos[1] = trial_y
            contact.stepped = True
            return True

        # --- 1. Move X ---
        pos[0] += vel[0] * dt
        for _ in range(4):
            player_box = make_box()
            hit = next((o for o in obstacles if player_box.intersects(o.bounds)), None)
            if hit is None:
                break
            if try_step_up(hit):
                continue
            if vel[0] > 0.0:
                pos[0] = hit.bounds.min[0] - half_w - 0.001
                contact.wall_normal = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
            elif vel[0] < 0.0:
                pos[0] = hit.bounds.max[0] + half_w + 0.001
                contact.wall_normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                break
            contact.wall_obstacle = hit
            contact.approach_speed = max(contact.approach_speed, abs(float(vel[0])))
            vel[0] = 0.0

        # --- 2. Move Z ---
        pos[2] += vel[2] * dt
        for _ in range(4):
            player_box = make_box()
            hit = next((o for o in obstacles if player_box.intersects(o.bounds)), None)
            if hit is None:
                break
            if try_step_up(hit):
                continue
            if vel[2] > 0.0:
                pos[2] = hit.bounds.min[2] - half_d - 0.001
                contact.wall_normal = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            elif vel[2] < 0.0:
                pos[2] = hit.bounds.max[2] + half_d + 0.001
                contact.wall_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                break
            contact.wall_obstacle = hit
            contact.approach_speed = max(contact.approach_speed, abs(float(vel[2])))
            vel[2] = 0.0

        # --- 3. Move Y ---
        pos[1] += vel[1] * dt
        player_box = make_box()
        for obs in obstacles:
            if player_box.intersects(obs.bounds):
                if vel[1] < 0.0:
                    # Landing on top of surface
                    pos[1] = obs.bounds.max[1] + half_h
                    vel[1] = 0.0
                    contact.on_ground = True
                elif vel[1] > 0.0:
                    # Head bumped ceiling
                    pos[1] = obs.bounds.min[1] - half_h - 0.001
                    vel[1] = 0.0
                    contact.hit_ceiling = True
                player_box.move_to(pos)

        # --- 4. Ground probe (small skin downward) ---
        if not contact.on_ground:
            feet_box = AABB(
                [pos[0] - half_w * 0.9, pos[1] - half_h - 0.05, pos[2] - half_d * 0.9],
                [pos[0] + half_w * 0.9, pos[1] - half_h, pos[2] + half_d * 0.9]
            )
            for obs in obstacles:
                if feet_box.intersects(obs.bounds):
                    contact.on_ground = True
                    break

        return pos, vel, contact
