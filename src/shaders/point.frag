uniform sampler2D camera_texture;
uniform sampler2D brush_stroke;

out vec4 color;

in vec4 g_color;
in float g_brush_index;

void main() {
    vec2 coord = gl_PointCoord;
    coord.x /= PR_NUM_BRUSHES;
    coord.x += g_brush_index / PR_NUM_BRUSHES;
    vec4 brush = texture(brush_stroke, coord);
    if (coord.x > 1.0) {
        discard;
    }

    float intensity = 1.0 - brush.x;

    color = g_color;
    color.a = intensity;
}