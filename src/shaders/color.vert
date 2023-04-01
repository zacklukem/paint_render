uniform mat4 view;
uniform mat4 perspective;
uniform mat4 model;

in vec3 position;
in vec3 normal;
in vec2 tex_coords;

out vec3 v_normal;
out vec2 v_tex_coords;

void main() {
    gl_Position = perspective * view * model * vec4(position, 1.0);
    v_normal = normal;
    v_tex_coords = tex_coords;
}