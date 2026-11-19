#version 330 core
layout (location = 0) in vec3 a_Position;
layout (location = 1) in vec3 a_Normal;
layout (location = 2) in vec2 a_UV;
layout (location = 3) in vec3 a_Color;
layout (location = 4) in vec3 a_Extra;   // variant * 1000 + material * 10 + face role, face width, face height

uniform mat4 u_Model;
uniform mat4 u_View;
uniform mat4 u_Proj;

out vec3 v_FragPos;
out vec3 v_Normal;
out vec2 v_UV;
out vec3 v_Color;
flat out vec3 v_Extra;

void main() {
    vec4 worldPos = u_Model * vec4(a_Position, 1.0);
    v_FragPos = worldPos.xyz;
    v_Normal = mat3(transpose(inverse(u_Model))) * a_Normal;
    v_UV = a_UV;
    v_Color = a_Color;
    v_Extra = a_Extra;
    gl_Position = u_Proj * u_View * worldPos;
}
