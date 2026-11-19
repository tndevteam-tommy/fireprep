#version 330 core
in vec2 v_Corner;

uniform vec3 u_Color;
uniform float u_Alpha;
uniform float u_Beam;      // 0 = round glow, 1 = a column of light fading toward its top

out vec4 FragColor;

void main() {
    float a;
    if (u_Beam > 0.5) {
        float across = 1.0 - abs(v_Corner.x);
        float up = 1.0 - smoothstep(-1.0, 1.0, v_Corner.y);
        a = pow(across, 2.2) * up * u_Alpha;
    } else {
        float r = length(v_Corner);
        if (r > 1.0) discard;
        float core = pow(1.0 - r, 3.0);
        float halo = pow(1.0 - r, 1.2) * 0.45;
        a = (core + halo) * u_Alpha;
    }
    FragColor = vec4(u_Color * a, a);          // additive: premultiplied glow
}
