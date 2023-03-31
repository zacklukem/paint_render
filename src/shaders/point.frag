#version 330

uniform sampler2D camera_texture;
uniform sampler2D brush_stroke;

out vec4 color;

in vec4 g_color;

void main() {
    vec2 coord = gl_PointCoord;
    coord.x *= 320.0 / 136.0;
    vec4 brush = texture(brush_stroke, coord);
    if (coord.x > 1.0) {
        discard;
    }

    float intensity = 1.0 - brush.x;

    color = g_color;
    color.a = intensity;
}