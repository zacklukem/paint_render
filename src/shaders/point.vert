#version 330

uniform mat4 view;
uniform mat4 perspective;
uniform mat4 model;

in vec3 position;

void main() {
    gl_Position = perspective * view * model * vec4(position, 1.0);
}