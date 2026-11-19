"""Random-input fuzzing of the full playground: physical invariants must never break."""
import numpy as np
from harness import *  # noqa
from physics.collision import CollisionSystem

L1, L2 = config.UPPER_ARM_LEN, config.FOREARM_LEN


def _fuzz(seed, seconds=10.0):
    rng = np.random.default_rng(seed)
    ls, obs = build_playground()
    spawn = (float(rng.uniform(-25, 25)), 0.9, float(rng.uniform(-25, 45)))
    while not BodyModel.pose_is_free(np.array(spawn, dtype=np.float32), obs):
        spawn = (float(rng.uniform(-25, 25)), 0.9, float(rng.uniform(-25, 45)))
    sim = Sim(ls, obs, spawn=spawn, yaw=float(rng.uniform(0, 360)), with_arms=True)
    keys = list(KEYS.keys())
    n = int(seconds / DT)
    for i in range(n):
        if i % 12 == 0:
            # change the chord every 0.2 s
            sim.release_all()
            for k in keys:
                if rng.random() < 0.45:
                    sim.hold(k)
            sim.player.yaw += float(rng.normal(0, 25))
            sim.player.pitch = float(np.clip(rng.normal(0, 20), -80, 80))
        sim.step()
        pos = sim.pos
        assert np.all(np.isfinite(pos)) and np.all(np.isfinite(sim.vel)), "NaN in simulation"
        if sim.state not in ParkourState.ROOT_MOTION and sim.player.respawn_count == 0:
            box = CollisionSystem.player_box(pos, BodyModel.SIZE, shrink=0.02)
            assert CollisionSystem.overlaps_any(box, obs) is None, f"body inside geometry at {pos} in {sim.state}"
        if sim.state in (ParkourState.LEDGE_HANG, ParkourState.SHIMMY) and sim.fsm.settle_time >= config.GRAB_SETTLE_TIME:
            assert BodyModel.pose_is_free(pos, obs), "hang pose intersects the world"
            assert sim.feet_y >= -0.01, "feet below the floor while hanging"
        L = sim.arm_rig.last
        for side in ("left", "right"):
            sh, el, wr = L[side + "_shoulder"], L[side + "_elbow"], L[side + "_wrist"]
            assert abs(np.linalg.norm(el - sh) - L1) < 1e-3
            assert abs(np.linalg.norm(wr - el) - L2) < 1e-3
            assert config.ELBOW_MIN_ANGLE_DEG - 0.01 <= L[side + "_elbow_angle"] <= config.ELBOW_MAX_ANGLE_DEG + 0.01
        assert sim.trace[-1]["step"] <= 0.25, "teleport"
    return sim


def test_fuzz_seed_1():
    _fuzz(1)


def test_fuzz_seed_2():
    _fuzz(2)


def test_fuzz_seed_3():
    _fuzz(3)
