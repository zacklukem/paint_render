use glium::{
    implement_vertex,
    index::{NoIndices, PrimitiveType},
    Display, IndexBuffer, VertexBuffer,
};
use tobj::Mesh;

use crate::point_gen::Point;

#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    tex_coords: [f32; 2],
}
implement_vertex!(Vertex, position, normal, tex_coords);

pub fn debug_points(display: &Display, points: &[Point]) -> (VertexBuffer<Point>, NoIndices) {
    (
        VertexBuffer::new(display, points).unwrap(),
        NoIndices(PrimitiveType::Points),
    )
}

pub fn gen_buffers(display: &Display, mesh: &Mesh) -> (VertexBuffer<Vertex>, IndexBuffer<u32>) {
    let mut vertices = vec![];

    let has_normals = !mesh.normals.is_empty();
    let has_tex_coords = !mesh.texcoords.is_empty();

    if has_normals {
        assert_eq!(mesh.positions.len() / 3, mesh.normals.len() / 3);
    }

    if has_tex_coords {
        assert_eq!(mesh.positions.len() / 3, mesh.texcoords.len() / 2);
    }

    for position in mesh.positions.chunks_exact(3) {
        let position = [position[0], position[1], position[2]];
        vertices.push(Vertex {
            position,
            normal: [0.0, 0.0, 0.0],
            tex_coords: [0.0, 0.0],
        });
    }

    if has_normals {
        for (i, normal) in mesh.normals.chunks_exact(3).enumerate() {
            let normal = [normal[0], normal[1], normal[2]];
            vertices[i].normal = normal;
        }
    }

    if has_tex_coords {
        for (i, tex_coord) in mesh.texcoords.chunks_exact(2).enumerate() {
            let tex_coord = [tex_coord[0], tex_coord[1]];
            vertices[i].tex_coords = tex_coord;
        }
    }

    let vb = VertexBuffer::new(display, &vertices).unwrap();
    let ib = IndexBuffer::new(display, PrimitiveType::TrianglesList, &mesh.indices).unwrap();
    (vb, ib)
}
