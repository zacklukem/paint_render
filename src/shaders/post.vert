in vec2 position;

out vec2 v_pos;

void main() {
    gl_Position = vec4(position, 0.0, 1.0);
    v_pos = (position + 1.0) / 2.0;
}