#version 330 core
in vec2 v_UV;

uniform sampler2D u_Tex;
uniform vec4 u_Color;

out vec4 FragColor;

void main() {
    vec4 t = texture(u_Tex, v_UV);
    FragColor = vec4(u_Color.rgb * t.rgb, u_Color.a * t.a);
}
