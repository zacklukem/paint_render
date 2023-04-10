uniform sampler2D post_process_texture;
uniform sampler2D canvas_texture;
uniform bool enable_canvas;
uniform float saturation;

out vec4 color;
in vec2 v_pos;

void main() {
    vec3 c = texture(post_process_texture, v_pos).xyz;
    if (enable_canvas) {
        c *= texture(canvas_texture, v_pos).x;
    }

    vec3 c_lum_scale = vec3(0.2126, 0.7152, 0.0722) * c;
    vec3 c_lum = vec3(c_lum_scale.x + c_lum_scale.y + c_lum_scale.z);

    c = mix(c_lum, c, saturation);

    color = vec4(c, 1.0);
}