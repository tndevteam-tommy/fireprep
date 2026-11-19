#version 330 core
in vec3 v_FragPos;
in vec3 v_Normal;
in vec2 v_UV;
flat in float v_BoneID;

uniform vec3 u_LightDir;
uniform vec3 u_CamPos;
uniform vec3 u_SkinColor;      // (0.78, 0.58, 0.46) - green for a zombie in zombie tag
uniform vec3 u_ShirtColor;     // (0.14, 0.15, 0.18)

out vec4 FragColor;

void main() {
    vec3 norm = normalize(v_Normal);
    vec3 lightDir = normalize(u_LightDir);
    vec3 viewDir = normalize(u_CamPos - v_FragPos);

    // Material: a_UV.x carries it (0 = by bone: sleeve / skin; 1 = skin; 2 = black cloth; 3 = brown boot; 4 = shirt)
    vec3 baseColor;
    int b = int(v_BoneID + 0.5);
    int mat = int(v_UV.x + 0.5);
    if (mat == 2) {
        baseColor = vec3(0.05, 0.05, 0.06);          // black pants
    } else if (mat == 3) {
        baseColor = vec3(0.38, 0.22, 0.10);          // brown leather boots
    } else if (mat == 4 || (mat == 0 && (b == 0 || b == 1 || b == 19 || b == 20))) {
        baseColor = u_ShirtColor;                    // shirt / sleeves (torso, upper arm, clavicle)
    } else {
        baseColor = u_SkinColor;                     // skin (forearms, palms, fingers)
    }

    // Ambient
    vec3 ambient = 0.55 * baseColor;

    // Diffuse
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * baseColor * 0.75;

    // Specular
    vec3 halfDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(norm, halfDir), 0.0), 16.0);
    vec3 specular = vec3(0.25) * spec;

    FragColor = vec4(ambient + diffuse + specular, 1.0);
}
