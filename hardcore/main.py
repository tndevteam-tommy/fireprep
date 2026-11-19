"""Entry point and master game loop for HARDCORE 3D Parkour Engine."""
import sys
import time
import numpy as np
import config
from math3d.transform import perspective, look_at
from math3d.quaternion import quat_from_axis_angle, quat_rotate_vec
from core.gamepad_device import GamepadManager, PadEvent   # sets SDL controller hints: import before pygame.init
from core.window import WindowManager, load_display_settings, save_display_settings, fullscreen_from_args
from core.input_buffer import InputBuffer, Action
from core.input_router import InputRouter
from core.controls import make_scheme
from core.keymap import keycode_map
from core.gamepad import ControllerBindings, default_settings_path, control_label
from core.gamepad_input import GamepadInput, Haptics, PAUSE
from core.mp_menu import MultiplayerMenuModel
from game.session import MultiplayerSession, parse_mp_args
from parkour.ledge_system import LedgeSystem
from parkour.controller import PlayerController
from parkour.body import BodyModel
from parkour import acrobatics
from world.generator import LevelGenerator
from animation.arm_rig import FirstPersonArmRig
from render.renderer import MasterRenderer
from render.menu import PauseMenu

try:
    import pygame
except ImportError:
    pass


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    # 1. Initialize Display & Modern OpenGL 3.3 Context (windowed or fullscreen: saved choice / flags)
    display = load_display_settings()
    window = WindowManager(fullscreen=fullscreen_from_args(argv, display["fullscreen"]))
    key_mappings = keycode_map(pygame)          # keyboard layout: core/keymap.py

    # 2. Build Procedural Environment & Ledge Graph
    ledge_system = LedgeSystem()
    t_build = time.time()
    obstacles, world_verts, world_indices, city = LevelGenerator.build_city(ledge_system, seed=config.CITY_SEED)
    print(f"[world] city built in {time.time() - t_build:.1f}s: {len(obstacles)} boxes, {len(ledge_system.rails)} edges, "
          f"{city.stats['buildings']} buildings, {city.stats['vehicles']} vehicles, {city.stats['props']} props")

    # 3. Initialize First-Person Skeletal Arm/Finger Rig
    arm_rig = FirstPersonArmRig()

    # 4. Initialize Modern GL Master Renderer
    renderer = MasterRenderer(world_verts, world_indices, ledge_system.rails, arm_rig)

    # 5. Initialize Controllers & State (start at beginning of Zone 1 runway)
    input_buf = InputBuffer()
    caps_on = bool(pygame.key.get_mods() & pygame.KMOD_CAPS)
    player = PlayerController(ledge_system, obstacles, spawn_pos=(-6.0, 0.9, -24.0), yaw=180.0,
                              scheme=make_scheme(run_mode=caps_on))
    fsm = player.fsm
    print(f"[controls] {player.scheme.description}")
    print(f"[controls] run mode {'ON' if caps_on else 'OFF'} (Caps Lock toggles)")

    # Keyboard, controller buttons and the stick all feed the input buffer through one router
    router = InputRouter(input_buf, lambda: player.scheme)

    # Controllers: bindings + settings persist per user; the pause menu's Controller screen edits them
    settings_path = default_settings_path()
    bindings = ControllerBindings.load(settings_path)
    pads = GamepadManager()
    pad_input = GamepadInput(bindings, router)
    haptics = Haptics()
    toasts = renderer.toasts
    last_input = "keyboard"             # which the player used last: "keyboard" (and mouse) or "pad"
    print(f"[controls] controller settings: {settings_path}")

    current_fov = config.BASE_FOV
    renderer.resize(*window.size)
    aspect = float(window.size[0]) / float(window.size[1])

    # Head Bob State
    head_bob_timer = 0.0
    head_bob_offset = np.zeros(3, dtype=np.float32)

    # Clock & Timers
    clock = window.clock
    accumulator = 0.0
    last_time = time.time()

    menu = renderer.menu
    menu.set_fullscreen_state(window.fullscreen)

    # Multiplayer: host (a server in this process, bots) or join; settings remembered
    mp_args = parse_mp_args(argv)
    mp_menu = MultiplayerMenuModel.load()
    mp_menu.load_values({k: mp_args[k] for k in ("name", "max_players", "emblems", "bots", "bot_level", "mode",
                                                  "teams", "minutes", "capture_limit", "team", "goal")
                         if k in mp_args})
    if "port" in mp_args:
        mp_menu.values["port"] = str(mp_args["port"])
    if "join" in mp_args:
        mp_menu.values["address"] = mp_args["join"]
    menu.attach_multiplayer(mp_menu)
    session = None
    pending_host = None
    menu.attach_controller(bindings, on_change=lambda b: b.save(settings_path))
    menu.set_pad(pads.active)
    if pads.active is not None:
        start = control_label("start", pads.active.family)
        toasts.push(f"Controller: {pads.active.describe()}  -  {start} > Controller to customise", key="pad")
    paused = False

    def set_paused(flag):
        nonlocal paused, accumulator
        paused = flag
        if flag:
            menu.open()
            router.release_all()
            pad_input.release_all()
            input_buf.clear()
            window.set_mouse_grab(False)
        else:
            menu.close()
            window.set_mouse_grab(True)
            pygame.mouse.get_rel()          # swallow the cursor jump caused by re-grabbing
            # a button still held from the menu (A on "Resume", a trigger) must not act until released
            pad_input.suppress_held(pads.active_state)
            accumulator = 0.0

    def run_mode_notice():
        if hasattr(player.scheme, "run_mode"):
            on = player.scheme.run_mode
            print(f"[controls] run mode {'ON' if on else 'OFF'}")
            toasts.push("Sprint lock ON" if on else "Sprint lock OFF", key="run")

    def toggle_fullscreen():
        on = window.toggle_fullscreen()
        display["fullscreen"] = on
        save_display_settings(display)
        menu.set_fullscreen_state(on)
        if paused:
            window.set_mouse_grab(False)       # the menu keeps its cursor

    def is_fullscreen_key(event):
        if event.type != pygame.KEYDOWN:
            return False
        if menu.in_controller_screen and menu.controller.waiting is not None:
            return False                      # binding a key: F11 / Enter are keys like any other
        alt_enter = event.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and (event.mod & pygame.KMOD_ALT)
        return event.key == pygame.K_F11 or bool(alt_enter)

    def start_session(action):
        """Host or join from the Multiplayer screen (or the command line).  Hosting starts the server
        in a process of its own, which takes a few seconds (it builds its copy of the city): the
        game goes on meanwhile and the session comes up in poll_pending_host()."""
        nonlocal session, pending_host
        if session is not None:
            leave_session()
        if pending_host is not None:
            pending_host.cancel()
            pending_host = None
        name = mp_menu.values["name"] or "Player"
        team = mp_menu.values["team"]
        try:
            if action[0] == "host":
                pending_host = MultiplayerSession.start_host(("game.match", "city_world", [config.CITY_SEED]),
                                                             action[1], name, team=team)
                mp_menu.say("Starting the server - building its copy of the city ...", "info")
                print("[multiplayer] starting the server ...")
                if action[1].rules().never_ends():
                    toasts.push("No time limit and no capture limit: rounds only end when you leave", key="mp-warn")
                return False
            session = MultiplayerSession.join(action[1], name, team=team)
        except (ConnectionError, OSError, ValueError) as exc:
            reason = getattr(exc, "strerror", None) or str(exc)
            mp_menu.say(f"Could not {'host' if action[0] == 'host' else 'join'}: {reason}", "warn")
            print(f"[multiplayer] {mp_menu.status}")
            return False
        session_started()
        return True

    def session_started():
        mp_menu.in_session = True
        mp_menu.say(session.describe(), "ok")
        toasts.push(session.describe(), key="mp")
        toasts.push(session.describe_game(), key="mp-game")
        print(f"[multiplayer] {session.describe()} - {session.describe_game()}")

    def poll_pending_host():
        """The server process we started: up yet?"""
        nonlocal session, pending_host
        try:
            ready = pending_host.poll()
        except (ConnectionError, OSError) as exc:
            pending_host = None
            mp_menu.say(f"Could not host: {exc}", "warn")
            print(f"[multiplayer] {mp_menu.status}")
            return
        if ready is None:
            dots = "." * (1 + int(pending_host.elapsed * 2) % 3)
            mp_menu.say(f"Starting the server - building its copy of the city {dots}", "info")
            return
        pending_host = None
        session = ready
        session_started()
        if paused and menu.screen in ("main", "multiplayer"):
            set_paused(False)

    def leave_session(reason=None):
        nonlocal session
        if session is None:
            return
        session.leave()
        session = None
        mp_menu.in_session = False
        renderer.mp.clear()
        menu.results = None
        if menu.screen == "results":
            menu.screen = "main"
        mp_menu.say(reason or "Left the game", "info")

    def menu_choice(choice):
        nonlocal running
        if choice == PauseMenu.RESUME:
            set_paused(False)
        elif choice == PauseMenu.FULLSCREEN:
            toggle_fullscreen()
        elif choice == PauseMenu.QUIT:
            running = False
        elif isinstance(choice, tuple) and choice[0] == "mp":
            action = choice[1]
            if action[0] in ("host", "join"):
                if start_session(action):
                    set_paused(False)
            elif action[0] == "leave":
                leave_session()
                if menu.screen != "main":
                    menu.screen = "main"
            elif action[0] == "join_next" and session is not None:
                session.join_next()
            elif action[0] == "team" and session is not None:
                if not session.switch_team(action[1]):
                    mp_menu.say("This game is a free for all - your team counts in teams games", "info")

    def on_pad_event(pe):
        nonlocal last_input
        if pe.kind == PadEvent.ADDED:
            toasts.push(f"Controller connected: {pe.device.describe()}", key="pad")
            menu.set_pad(pads.active)
        elif pe.kind == PadEvent.REMOVED:
            toasts.push(f"Controller disconnected: {pe.device.name}", key="pad")
            if pe.was_active:
                pad_input.release_all()
                if not paused and last_input == "pad":
                    set_paused(True)          # like a console: never keep running with no one at the controls
            menu.set_pad(pads.active)
        elif pe.kind == PadEvent.ACTIVATED:
            pad_input.release_all()
            menu.set_pad(pe.device)
        elif pe.kind == PadEvent.BUTTON:
            if pe.device is not pads.active:
                return
            if pe.down:
                last_input = "pad"
            if paused:
                menu_choice(menu.handle_pad(pe.control, pe.down, pad_input.menu_event(pe.control, pe.down)))
                return
            result = pad_input.on_button(pe.control, pe.down)
            if result == PAUSE:
                set_paused(True)
            elif result == Action.RUN_TOGGLE:
                run_mode_notice()

    # Command line: --host / --join ADDRESS (see game/session.py parse_mp_args)
    if mp_args.get("host"):
        request = mp_menu.host_request()
        if request is not None:
            start_session(request)
        else:
            print(f"[multiplayer] {mp_menu.status}")
    elif "join" in mp_args:
        start_session(("join", mp_args["join"]))

    running = True
    while running:
        now = time.time()
        frame_dt = min(0.1, now - last_time)
        last_time = now
        if not paused:
            accumulator += frame_dt

        # =====================================================================
        # 1. EVENT POLLING & INPUT
        # =====================================================================
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue

            pad_events = pads.process(event)
            if pad_events is not None:
                for pe in pad_events:
                    on_pad_event(pe)
                continue

            if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                last_input = "keyboard"

            if is_fullscreen_key(event):
                toggle_fullscreen()
                continue

            if paused:
                menu_choice(menu.handle_event(event))

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    set_paused(True)
                elif event.key in key_mappings:
                    action = key_mappings[event.key]
                    router.press(("key", event.key), action)
                    if action == Action.RUN_TOGGLE:
                        run_mode_notice()

            elif event.type == pygame.KEYUP:
                if event.key in key_mappings:
                    router.release(("key", event.key), key_mappings[event.key])

            elif event.type == pygame.MOUSEMOTION and window.mouse_grabbed:
                dx, dy = event.rel
                player.apply_mouse(dx, dy)

        # Multiplayer: send our pose, hear about everyone else, rounds and results
        if pending_host is not None:
            poll_pending_host()
        if session is not None:
            for kind, text in session.update(frame_dt, player):
                toasts.push(text, key="mp-pickup" if kind == "pickup" else None)
            if session.client.team and mp_menu.values["team"] != session.client.team and session.client.teams_game:
                mp_menu.values["team"] = session.client.team       # the side you are actually on
            if session.ended and not session.client.connected:
                leave_session(session.ended)
            elif session.results is not None:
                if not paused:
                    set_paused(True)
                menu.show_results(session.results_view())
            elif menu.screen == "results":
                menu.results = None
                menu.screen = "main"
                set_paused(False)
            if session is not None:
                renderer.mp.set_frame(session.remote_poses(), session.emblems(), session.hud_state(player.pos))
        # zombie tag: a zombie's own arms are green and ragged too
        zombie_me = session is not None and session.client.living is not None and session.client.living != session.client.id
        renderer.skin_color = renderer.ZOMBIE_SKIN if zombie_me else renderer.SKIN
        renderer.shirt_color = renderer.ZOMBIE_SHIRT if zombie_me else renderer.SHIRT
        if session is None:
            player.set_speed_scales(1.0, 1.0)

        # The window may have changed size (fullscreen / back): re-layout the HUD, fix the aspect
        if renderer.resize(*window.size):
            aspect = float(window.size[0]) / float(window.size[1])

        # Controller sticks: analog movement + camera (play), held-direction repeats (menus)
        if not paused:
            turn, look_up = pad_input.update(pads.active_state, frame_dt)
            if turn or look_up:
                player.apply_look(turn, look_up)
            if turn or look_up or pad_input.move != (0.0, 0.0):
                last_input = "pad"
        elif running:
            for command in pad_input.menu_update(pads.active_state, frame_dt):
                menu_choice(menu.navigate(command))
                if not paused:
                    break

        # =====================================================================
        # 2. FIXED TIMESTEP SIMULATION LOOP (60 Hz): FSM + Swept AABB collision
        # =====================================================================
        while accumulator >= config.FIXED_DT:
            was_on_ground, prev_vy = player.on_ground, float(player.vel[1])
            player.step(config.FIXED_DT, input_buf)
            accumulator -= config.FIXED_DT
            if bindings.vibration and last_input == "pad" and pads.active is not None:
                buzz = haptics.on_tick(fsm.last_event, was_on_ground, player.on_ground, prev_vy)
                if buzz is not None:
                    pads.rumble(*buzz)
        ui_dt = frame_dt
        if paused:
            frame_dt = 0.0      # freeze bob / reticle / arm animation while the menu is up

        # Draw the body between the last two physics ticks (the fraction of a tick the clock is into
        # the next one): even motion whether this frame ran two ticks, one or none
        pos = player.render_pos(accumulator / config.FIXED_DT)
        vel, on_ground = player.vel, player.on_ground
        forward_3d = player.forward_3d
        right_h = player.right_h

        # Base camera up vector with Dutch roll tilt (lateral dynos / wall push-offs)
        cam_up = np.cross(right_h, forward_3d)
        cam_up /= (np.linalg.norm(cam_up) + 1e-7)
        if abs(fsm.dutch_tilt) > 0.01:
            tilt_q = quat_from_axis_angle(forward_3d, np.radians(fsm.dutch_tilt))
            cam_up = quat_rotate_vec(tilt_q, cam_up)

        # =====================================================================
        # 3. CAMERA KINEMATICS & HEAD BOBBING
        # =====================================================================
        speed_h = float(np.linalg.norm([vel[0], vel[2]]))
        spin = fsm.spin
        if on_ground and speed_h > 0.5 and spin is None:
            bob_speed_scale = (speed_h / config.WALK_SPEED)
            head_bob_timer += frame_dt * config.HEAD_BOB_FREQ * bob_speed_scale
            head_bob_offset[1] = np.sin(head_bob_timer) * config.HEAD_BOB_AMP
            head_bob_offset[0] = np.cos(head_bob_timer * 0.5) * (config.HEAD_BOB_AMP * 0.5)
        else:
            head_bob_offset += (0.0 - head_bob_offset) * min(1.0, 10.0 * frame_dt)

        eye_pos = BodyModel.camera_pos(pos, player.forward_h)
        eye_pos = eye_pos + right_h * head_bob_offset[0] + np.array([0.0, head_bob_offset[1], 0.0], dtype=np.float32)

        current_fov += (fsm.target_fov - current_fov) * min(1.0, config.FOV_LERP_SPEED * frame_dt)

        # Flips and rolls: the eyes turn with the body about its centre of mass (the body mesh gets
        # the same transform in the renderer; the rig itself is posed upright from rig_eye)
        rig_eye = eye_pos
        eye_pos, view_fwd, cam_up, body_T = acrobatics.view_for(spin, pos, eye_pos, forward_3d, cam_up, right_h)
        renderer.body_transform = body_T
        view_mat = look_at(eye_pos, eye_pos + view_fwd, cam_up)
        proj_mat = perspective(current_fov, aspect, 0.05, 500.0)

        # Soft-lock cursor: nearest rail in the view cone; "locked" only when it is humanly grabbable
        targeted_rail_info = ledge_system.find_best_rail(
            eye_pos, forward_3d, max_dist=config.RETICLE_MAX_DIST, cone_deg=config.DETECTION_CONE_DEG
        )
        if targeted_rail_info[0] is None:
            targeted_rail_info = None
        grab_candidate = player.grab_candidate() if spin is None else None
        climb_down = None
        if spin is not None:
            targeted_rail_info = None          # tumbling: no targeting
        elif grab_candidate is not None:
            targeted_rail_info = (grab_candidate.rail, grab_candidate.grip, 0.0, grab_candidate.s)
        else:
            # Standing at a drop: offer to lower yourself over the edge (E)
            climb_down = player.climb_down_candidate()
            if climb_down is not None:
                rail, s = climb_down
                targeted_rail_info = (rail, rail.get_point(s), 0.0, s)

        # =====================================================================
        # 4. PROCEDURAL SKELETAL ARMS & FINGERS UPDATE
        # =====================================================================
        arm_rig.update(
            frame_dt, pos, rig_eye, player.yaw, player.pitch,
            fsm.state, fsm.active_rail, targeted_rail_info, vel, on_ground,
            grip_point=fsm.grip_point, climb_phase=fsm.climb_phase,
            climb_next=fsm.climb_next, grab_candidate=grab_candidate,
            shimmy_dir=fsm.shimmy_dir, corner_link=fsm.corner_link,
            corner_entry_s=fsm.corner_entry_s, mantle_reverse=fsm.mantle_reverse,
            climb_direction=fsm.climb_direction, supported_hang=fsm.supported_hang, spin=spin,
            zombie=renderer.skin_color == renderer.ZOMBIE_SKIN, move_rate=fsm.move_rate,
            move_time_left=fsm.move_time_left
        )

        renderer.reticle.update(frame_dt, targeted_rail_info, view_mat, proj_mat,
                                locked=grab_candidate is not None, climb_down=climb_down is not None)

        # =====================================================================
        # 5. MASTER RENDER PASS
        # =====================================================================
        renderer.render_frame(view_mat, proj_mat, eye_pos, targeted_rail_info, fsm.active_rail, ui_dt=ui_dt)

        window.swap()
        clock.tick(config.FPS)

    if session is not None:
        session.leave()
    if pending_host is not None:
        pending_host.cancel()
    pads.close()
    window.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
