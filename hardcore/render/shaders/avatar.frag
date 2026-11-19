#version 330 core
in vec3 v_Normal;
in vec3 v_Pos;

uniform vec3 u_Color;
uniform vec3 u_LightDir;
uniform vec3 u_CamPos;
uniform float u_Emissive;    // 0 = lit surface, 1 = glows (emblem cores, the bot's visor)

out vec4 FragColor;

void main() {
    vec3 n = normalize(v_Normal);
    float diffuse = max(dot(n, normalize(u_LightDir)), 0.0);
    vec3 view = normalize(u_CamPos - v_Pos);
    float rim = pow(1.0 - max(dot(n, view), 0.0), 2.0);
    vec3 lit = u_Color * (0.38 + 0.62 * diffuse) + rim * 0.08;
    vec3 glow = u_Color * (0.85 + 0.6 * rim);
    FragColor = vec4(mix(lit, glow, u_Emissive), 1.0);
}
