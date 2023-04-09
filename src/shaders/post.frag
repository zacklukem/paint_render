uniform sampler2D post_process_texture;

out vec4 color;
in vec2 v_pos;

void main() {
    color = texture(post_process_texture, v_pos);
}