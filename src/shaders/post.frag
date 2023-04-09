uniform sampler2D post_process_texture;
uniform sampler2D canvas_texture;
uniform bool enable_canvas;

out vec4 color;
in vec2 v_pos;

void main() {
    color = texture(post_process_texture, v_pos);
    if (enable_canvas) {
        color.xyz *= texture(canvas_texture, v_pos).x;
    }
}