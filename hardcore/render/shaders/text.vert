#version 330 core
layout (location = 0) in vec2 a_Position;
layout (location = 1) in vec2 a_UV;

uniform mat4 u_Proj;
uniform vec2 u_Offset;
uniform vec2 u_Size;

out vec2 v_UV;

void main() {
    vec2 pos = u_Offset + a_Position * u_Size;
    v_UV = a_UV;
    gl_Position = u_Proj * vec4(pos, 0.0, 1.0);
}
