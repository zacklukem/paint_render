layout (points) in;
layout (triangle_strip, max_vertices = 6) out;

in float v_brush_index[];
in vec4 v_color[];
in float v_model_depth[];
in vec3 v_tangent[];
in vec3 v_bitangent[];

out float g_brush_index;
out vec4 g_color;
out vec2 g_uv;

void main() {
    vec4 position = gl_in[0].gl_Position;
    // Equivilant to gl_FragCoord.z
    float point_depth = ((position.z / position.w) + 1.0) / 2.0;
    // float model_depth = v_model_depth[0];

    // vec2 direction = normalize(vec2(1.0, 1.5));
    vec2 direction = normalize(v_tangent[0].xy);


    mat4 rot = mat4(
        direction.x, -direction.y, 0.0, 0.0,
        direction.y, direction.x, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    );

    // TODO: fix depth
    // if (point_depth - 0.01 <= model_depth) {
    g_brush_index = v_brush_index[0];
    float point_size = 0.04;

    g_color = v_color[0];


    // TL -- TR
    // |  \  |
    // BL -- BR


    g_uv = vec2(0.0, 0.0);
    vec4 p = rot * vec4(-point_size, -point_size, 0.0, 0.0); // BL
    gl_Position = p + position;
    EmitVertex(); 
    g_uv = vec2(1.0, 0.0);
    p = rot * vec4(point_size, -point_size, 0.0, 0.0); // BL
    gl_Position = p + position;
    EmitVertex(); 
    g_uv = vec2(0.0, 1.0);
    p = rot * vec4(-point_size, point_size, 0.0, 0.0); // BL
    gl_Position = p + position;
    EmitVertex(); 

    g_uv = vec2(1.0, 0.0);
    p = rot * vec4(point_size, -point_size, 0.0, 0.0); // BL
    gl_Position = p + position;
    EmitVertex(); 
    g_uv = vec2(1.0, 1.0);
    p = rot * vec4(point_size, point_size, 0.0, 0.0); // BL
    gl_Position = p + position;
    EmitVertex(); 
    g_uv = vec2(0.0, 1.0);
    p = rot * vec4(-point_size, point_size, 0.0, 0.0); // BL
    gl_Position = p + position;
    EmitVertex(); 
    EndPrimitive();
    // }
}
