#version 330 core
layout (location = 0) in vec2 a_Corner;     // -1 .. 1

uniform mat4 u_View;
uniform mat4 u_Proj;
uniform vec3 u_Center;
uniform vec3 u_Right;     // camera right / up in world space: the quad always faces the viewer
uniform vec3 u_Up;        // (a light beam passes the world's up instead, and stretches)
uniform float u_Size;
uniform vec2 u_Stretch;   // (1, 1) = a round glow; (1, n) = a beam n times taller than wide

out vec2 v_Corner;

void main() {
    vec3 wp = u_Center + (u_Right * a_Corner.x * u_Stretch.x + u_Up * a_Corner.y * u_Stretch.y) * u_Size;
    v_Corner = a_Corner;
    gl_Position = u_Proj * u_View * vec4(wp, 1.0);
}
