#version 330 core
layout (location = 0) in vec3 a_Position;
layout (location = 1) in vec3 a_Normal;

uniform mat4 u_Model;
uniform mat4 u_View;
uniform mat4 u_Proj;

out vec3 v_Normal;
out vec3 v_Pos;

void main() {
    vec4 wp = u_Model * vec4(a_Position, 1.0);
    v_Pos = wp.xyz;
    v_Normal = mat3(u_Model) * a_Normal;
    gl_Position = u_Proj * u_View * wp;
}
