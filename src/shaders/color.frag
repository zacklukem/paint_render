#version 330

out vec4 color;
in vec3 v_normal;

void main() {
    color = vec4(1.0, 1.0, 1.0, 1.0) * max(dot(v_normal, vec3(0.0, 0.0, 1.0)), 0.3);
}