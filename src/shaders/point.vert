uniform mat4 view;
uniform mat4 perspective;
uniform mat4 model;
uniform vec3 camera_pos;
uniform sampler2D color_texture;
uniform sampler2D albedo_texture;

in vec3 position;
in vec3 normal;
in vec3 tangent;
in vec3 bitangent;
in vec2 uv;
in int brush_index;

out float v_brush_index;
out vec4 v_color;
out float v_model_depth;
out vec3 v_tangent;
out vec3 v_bitangent;

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

const vec3 TO_LIGHT_DIR = normalize(vec3(-1.0, 1.0, 1.0));

void main() {
    v_brush_index = float(brush_index);

    gl_Position = perspective * view * model * vec4(position, 1.0);

    v_tangent = (view * model * vec4(tangent, 0.0)).xyz;
    v_bitangent = (view * model * vec4(bitangent, 0.0)).xyz;

    // the w was a z...
    vec3 raw_pos = gl_Position.xyz / gl_Position.w;
    raw_pos = (raw_pos + 1.0) / PR_NUM_BRUSHES;

    raw_pos.x = clamp(raw_pos.x, 0.0, 1.0);
    raw_pos.y = clamp(raw_pos.y, 0.0, 1.0);

    vec4 sample = texture(color_texture, raw_pos.xy);
    v_model_depth = sample.w;

    // Shading

    vec3 n = normalize((model * vec4(normal, 0.0)).xyz);

    vec3 p = (model * vec4(position, 1.0)).xyz;

    vec3 to_view = normalize(p - camera_pos);

    vec3 r = normalize(reflect(TO_LIGHT_DIR, n));

    float kS = pow(max(dot(r, to_view), 0.0), 20.0);

    float kD = max(dot(n, TO_LIGHT_DIR), 0.0);

    v_color = texture(albedo_texture, uv) * (kD + 0.2) + vec4(1.0, 1.0, 1.0, 1.0) * kS;

    // Apply quantization to brightness
    vec3 hsv = rgb2hsv(v_color.xyz);
    float quantize = 5.0;
    hsv.z = floor(hsv.z * (quantize - 1.0) + 0.5) / (quantize - 1.0);
    v_color.xyz = hsv2rgb(hsv);
}