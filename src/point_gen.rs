use cgmath::{prelude::*, Vector3};
use log::{info, warn};
use tobj::Model;

/// Generates points on the surface of a model with a density of `density` points per unit squared
pub fn gen_point_list(model: &Model, density: f32) -> Vec<Vector3<f32>> {
    let mesh = &model.mesh;

    let mut points = vec![];

    let mut total_area = 0.0;

    for triangle in mesh.indices.chunks(3) {
        if triangle.len() != 3 {
            warn!(
                "Found a polygon with {} vertices ({triangle:?})",
                triangle.len()
            );
            continue;
        }
        let a = &mesh.positions[(triangle[0] * 3) as usize..(triangle[0] * 3 + 3) as usize];
        let b = &mesh.positions[(triangle[1] * 3) as usize..(triangle[1] * 3 + 3) as usize];
        let c = &mesh.positions[(triangle[2] * 3) as usize..(triangle[2] * 3 + 3) as usize];
        let a = Vector3::new(a[0], a[1], a[2]);
        let b = Vector3::new(b[0], b[1], b[2]);
        let c = Vector3::new(c[0], c[1], c[2]);

        let ab = b - a;
        let ac = c - a;

        let area = ab.cross(ac).magnitude() / 2.0;
        total_area += area;
        let num_points_f32 = area * density;
        let mut num_points = num_points_f32.floor() as usize;
        let num_points_remainder = num_points_f32 - num_points as f32;

        if fastrand::f32() < num_points_remainder {
            num_points += 1;
        }
        for _ in 0..num_points {
            let mut r1 = fastrand::f32();
            let mut r2 = fastrand::f32();
            if r1 + r2 >= 1.0 {
                r1 = 1.0 - r1;
                r2 = 1.0 - r2;
            }
            points.push(a + ab * r1 + ac * r2)
        }
    }

    let actual_density = points.len() as f32 / total_area;

    let error = (100.0 * (actual_density - density) / density).abs();

    info!(
        "{}:\n\tTotal area: {total_area}\n\texpected density: {density}\n\tactual density: {actual_density}\n\terror: {error}%",
        model.name,
    );

    points
}
