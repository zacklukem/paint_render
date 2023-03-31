#version 330

out vec4 color;

in vec3 v_normal;
in vec2 v_tex_coords;

uniform sampler2D albedo_texture;

void main() {
    vec3 tex = texture(albedo_texture, v_tex_coords).rgb;
    color.xyz = tex * (max(dot(v_normal, vec3(0.0, 0.0, 1.0)), 0.0) + 0.3);
    color.w = gl_FragCoord.z;
}