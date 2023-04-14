layout(points) in;
layout(triangle_strip, max_vertices = 6) out;

uniform float brush_size;
uniform bool enable_brush_tbn;

in float v_brush_index[];
in vec4 v_color[];
in vec3 v_tangent[];
in vec3 v_bitangent[];

out float g_brush_index;
out vec4 g_color;
out vec2 g_uv;

void main() {
    vec4 position = gl_in[0].gl_Position;

    // vec2 direction = normalize(vec2(0.0, 1.0));
    vec2 direction = normalize(v_tangent[0].xy);

    // clang-format off
    mat4 rot = mat4(
        direction.x, -direction.y, 0.0, 0.0,
        direction.y, direction.x, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );
    // clang-format on

    g_brush_index = v_brush_index[0];
    float point_size = brush_size;

    g_color = v_color[0];

    mat4 tbn = mat4(1.0);
    if (enable_brush_tbn) {
        tbn = mat4(vec4(normalize(v_tangent[0]), 0.0), vec4(normalize(v_bitangent[0]), 0.0),
                   vec4(0.0, 0.0, 0.0, 0.0), vec4(0.0, 0.0, 0.0, 0.0));
    }

    // TL -- TR
    // |  \  |
    // BL -- BR

    g_uv = vec2(0.0, 0.0);
    vec4 p = tbn * rot * vec4(-point_size, -point_size, 0.0, 0.0); // BL
    gl_Position = p + position;
    EmitVertex();
    g_uv = vec2(1.0, 0.0);
    p = tbn * rot * vec4(point_size, -point_size, 0.0, 0.0); // BL
    gl_Position = p + position;
    EmitVertex();
    g_uv = vec2(0.0, 1.0);
    p = tbn * rot * vec4(-point_size, point_size, 0.0, 0.0); // BL
    gl_Position = p + position;
    EmitVertex();

    g_uv = vec2(1.0, 0.0);
    p = tbn * rot * vec4(point_size, -point_size, 0.0, 0.0); // BL
    gl_Position = p + position;
    EmitVertex();
    g_uv = vec2(1.0, 1.0);
    p = tbn * rot * vec4(point_size, point_size, 0.0, 0.0); // BL
    gl_Position = p + position;
    EmitVertex();
    g_uv = vec2(0.0, 1.0);
    p = tbn * rot * vec4(-point_size, point_size, 0.0, 0.0); // BL
    gl_Position = p + position;
    EmitVertex();
    EndPrimitive();
}
