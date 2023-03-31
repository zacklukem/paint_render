#version 330

uniform sampler2D camera_texture;
uniform sampler2D brush_stroke;

out vec4 color;

in vec4 g_color;

void main() {
    vec4 brush = texture(brush_stroke, gl_PointCoord.xy);

    float intensity = 1.0 - brush.x;

    color = g_color;
    color.a = intensity;
}