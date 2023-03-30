#version 330

uniform mat4 view;
uniform mat4 perspective;
uniform mat4 model;

in vec3 position;
in vec3 normal;

out vec3 v_normal;

void main() {
    gl_Position = perspective * view * model * vec4(position, 1.0);
    v_normal = normal;
}