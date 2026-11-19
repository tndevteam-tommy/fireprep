"""Contextual Soft-Lock Reticle Renderer."""
import numpy as np
import config
from math3d.transform import ortho


def ui_scale(screen_w, screen_h):
    """HUD / menu scale: the UI is designed for config.UI_DESIGN_SIZE and grows uniformly with the screen."""
    dw, dh = config.UI_DESIGN_SIZE
    return max(0.5, min(screen_w / float(dw), screen_h / float(dh)))
from render.buffer import MeshBuffer
try:
    from OpenGL.GL import *
except ImportError:
    pass


class ContextualReticle:
    def __init__(self, screen_w=config.WINDOW_WIDTH, screen_h=config.WINDOW_HEIGHT):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.ui_scale = ui_scale(screen_w, screen_h)
        self.cursor_pos = np.array([screen_w * 0.5, screen_h * 0.5], dtype=np.float32)
        self.target_pos = self.cursor_pos.copy()
        self.is_locked = False
        self.is_tracking = False
        self.is_climb_down = False

        # Build 2D procedural UI primitives (unit square centered at 0,0)
        # Vertices: 2D pos
        verts = np.array([
            [-0.5, -0.5],
            [ 0.5, -0.5],
            [ 0.5,  0.5],
            [-0.5,  0.5]
        ], dtype=np.float32)
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

        # Build diamond primitive for locked mode
        diamond_verts = np.array([
            [ 0.0, -0.7],
            [ 0.7,  0.0],
            [ 0.0,  0.7],
            [-0.7,  0.0]
        ], dtype=np.float32)
        diamond_indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

        # Downward chevron for the "climb down over this edge" prompt
        chevron_verts = np.array([
            [-1.0, -0.6],
            [-0.7, -0.9],
            [ 0.0, -0.2],
            [ 0.7, -0.9],
            [ 1.0, -0.6],
            [ 0.0,  0.4],
        ], dtype=np.float32)
        chevron_indices = np.array([0, 1, 2, 2, 3, 4, 0, 2, 5, 2, 4, 5], dtype=np.uint32)

        layout = [(0, 2, GL_FLOAT if 'GL_FLOAT' in globals() else 5126, False)]
        self.quad_buf = MeshBuffer(verts, indices, layout)
        self.diamond_buf = MeshBuffer(diamond_verts, diamond_indices, layout)
        self.chevron_buf = MeshBuffer(chevron_verts, chevron_indices, layout)

    def resize(self, screen_w, screen_h):
        """The window changed size (fullscreen / back): keep the cursor at the same relative spot."""
        fx = self.cursor_pos[0] / max(1.0, self.screen_w)
        fy = self.cursor_pos[1] / max(1.0, self.screen_h)
        self.screen_w, self.screen_h = screen_w, screen_h
        self.ui_scale = ui_scale(screen_w, screen_h)
        self.cursor_pos = np.array([fx * screen_w, fy * screen_h], dtype=np.float32)
        self.target_pos = self.cursor_pos.copy()

    def update(self, dt, targeted_rail_info, view_mat, proj_mat, locked=None, climb_down=False):
        """
        Projects targeted rail to screen space and updates magnetic spring lerp.
        targeted_rail_info: tuple (rail, pt_on_rail, dist, s) or None
        locked: optional explicit "grabbable right now" flag from the body reach model
        climb_down: the target is an edge below the feet the player can lower themselves over (E)
        """
        center_x = self.screen_w * 0.5
        center_y = self.screen_h * 0.5
        default_center = np.array([center_x, center_y], dtype=np.float32)

        self.is_locked = False
        self.is_tracking = False
        self.is_climb_down = False

        if targeted_rail_info is not None:
            rail, pt_world, dist, s = targeted_rail_info
            # Project point into clip space
            pt4 = np.array([pt_world[0], pt_world[1], pt_world[2], 1.0], dtype=np.float32)
            clip = proj_mat @ (view_mat @ pt4)

            w = clip[3]
            if w > 0.1:
                ndc_x = clip[0] / w
                ndc_y = clip[1] / w
                if -1.2 <= ndc_x <= 1.2 and -1.2 <= ndc_y <= 1.2:
                    screen_x = (ndc_x + 1.0) * 0.5 * self.screen_w
                    screen_y = (1.0 - ndc_y) * 0.5 * self.screen_h
                    self.target_pos = np.array([screen_x, screen_y], dtype=np.float32)

                    in_reach = locked if locked is not None else (dist <= config.MAX_GRAB_REACH)
                    if climb_down:
                        self.is_climb_down = True
                        self.target_pos[1] = min(self.target_pos[1], self.screen_h - 70.0 * self.ui_scale)
                    elif in_reach:
                        self.is_locked = True
                    else:
                        self.is_tracking = True
                elif climb_down:
                    # The edge is right under the feet (below the view): keep the prompt
                    # pinned to the bottom of the screen
                    screen_x = float(np.clip((ndc_x + 1.0) * 0.5 * self.screen_w, 60.0 * self.ui_scale, self.screen_w - 60.0 * self.ui_scale))
                    self.target_pos = np.array([screen_x, self.screen_h - 70.0 * self.ui_scale], dtype=np.float32)
                    self.is_climb_down = True
                else:
                    self.target_pos = default_center
            elif climb_down:
                self.target_pos = np.array([center_x, self.screen_h - 70.0 * self.ui_scale], dtype=np.float32)
                self.is_climb_down = True
            else:
                self.target_pos = default_center
        else:
            self.target_pos = default_center

        # Smooth magnetic lerp toward target
        lerp_speed = 18.0 if (self.is_locked or self.is_tracking or self.is_climb_down) else 12.0
        self.cursor_pos += (self.target_pos - self.cursor_pos) * min(1.0, lerp_speed * dt)

    def draw(self, shader):
        """Draws the contextual reticle using orthographic projection."""
        ortho_mat = ortho(0.0, self.screen_w, self.screen_h, 0.0, -1.0, 1.0)
        shader.use()
        shader.set_mat4("u_Proj", ortho_mat)

        # 1. Neutral Center Aim Dot (always present so player knows where forward is)
        center_x = self.screen_w * 0.5
        center_y = self.screen_h * 0.5
        shader.set_vec2("u_Offset", (center_x, center_y))
        shader.set_float("u_Scale", 4.0 * self.ui_scale)
        shader.set_vec4("u_Color", (1.0, 1.0, 1.0, 0.40))
        self.quad_buf.draw()

        # 2. Magnetic Soft-Lock Cursor
        dist_from_center = float(np.linalg.norm(self.cursor_pos - np.array([center_x, center_y])))

        if self.is_climb_down:
            # Green downward chevron: press E to lower yourself over this edge
            shader.set_vec2("u_Offset", (self.cursor_pos[0], self.cursor_pos[1]))
            shader.set_float("u_Scale", 18.0 * self.ui_scale)
            shader.set_vec4("u_Color", (0.35, 1.0, 0.45, 0.95))
            self.chevron_buf.draw()

        elif self.is_locked:
            # High-intensity Amber Diamond (In Reach)
            shader.set_vec2("u_Offset", (self.cursor_pos[0], self.cursor_pos[1]))
            shader.set_float("u_Scale", 16.0 * self.ui_scale)
            shader.set_vec4("u_Color", (1.0, 0.58, 0.05, 0.95))
            self.diamond_buf.draw()

            # Center target pip
            shader.set_float("u_Scale", 4.0 * self.ui_scale)
            shader.set_vec4("u_Color", (1.0, 1.0, 1.0, 1.0))
            self.quad_buf.draw()

        elif self.is_tracking:
            # Faint Cyan Tracking Reticle (Within view cone but out of reach)
            shader.set_vec2("u_Offset", (self.cursor_pos[0], self.cursor_pos[1]))
            shader.set_float("u_Scale", 22.0 * self.ui_scale)
            shader.set_vec4("u_Color", (0.0, 0.88, 1.0, 0.65))
            self.diamond_buf.draw()

            shader.set_float("u_Scale", 5.0 * self.ui_scale)
            shader.set_vec4("u_Color", (0.0, 0.88, 1.0, 0.90))
            self.quad_buf.draw()

        elif dist_from_center > 2.0:
            # Transitioning smoothly back to center
            shader.set_vec2("u_Offset", (self.cursor_pos[0], self.cursor_pos[1]))
            shader.set_float("u_Scale", 8.0 * self.ui_scale)
            shader.set_vec4("u_Color", (1.0, 1.0, 1.0, 0.25))
            self.quad_buf.draw()
