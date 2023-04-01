layout (points) in;
layout (points, max_vertices = 1) out;

in float v_brush_index[];
in vec4 v_color[];
in float v_model_depth[];

out float g_brush_index;
out vec4 g_color;

void main() {
    vec4 position = gl_in[0].gl_Position;
    // Equivilant to gl_FragCoord.z
    float point_depth = ((position.z / position.w) + 1.0) / 2.0;
    float model_depth = v_model_depth[0];

    if (point_depth - 0.01 <= model_depth) {
        g_brush_index = v_brush_index[0];
        gl_PointSize = 50.0;
        g_color = v_color[0];
        gl_Position = gl_in[0].gl_Position;
        EmitVertex(); 
        EndPrimitive();
    }
}

