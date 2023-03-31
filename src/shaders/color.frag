#version 330

out vec4 color;

in vec3 v_normal;
in vec2 v_tex_coords;

void main() {
    color.xyz = vec3(v_tex_coords.xy, 0.8) * (max(dot(v_normal, vec3(0.0, 0.0, 1.0)), 0.0) + 0.3);
    color.w = gl_FragCoord.z;
}