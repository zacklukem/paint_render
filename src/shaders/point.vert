#version 330

uniform mat4 view;
uniform mat4 perspective;
uniform mat4 model;
uniform sampler2D color_texture;

in vec3 position;

out vec4 v_color;
out float v_model_depth;

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    gl_Position = perspective * view * model * vec4(position, 1.0);

    // the w was a z...
    vec3 raw_pos = gl_Position.xyz / gl_Position.w;
    raw_pos = (raw_pos + 1.0) / 2.0;

    raw_pos.x = clamp(raw_pos.x, 0.0, 1.0);
    raw_pos.y = clamp(raw_pos.y, 0.0, 1.0);

    vec4 sample = texture(color_texture, raw_pos.xy);

    v_color = vec4(sample.rgb, 1.0);
    v_model_depth = sample.w;
}