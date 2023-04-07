use cgmath::{prelude::*, Vector2, Vector3};
use glium::implement_vertex;
use log::{info, warn};
use tobj::Model;

#[derive(Copy, Clone, Debug)]
pub struct Point {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 3],
    pub bitangent: [f32; 3],
    pub uv: [f32; 2],
    pub brush_index: i32,
}
implement_vertex!(Point, position, normal, tangent, bitangent, uv, brush_index);

/// Generates points on the surface of a model with a density of `density` points per unit squared
pub fn gen_point_list(model: &Model, density: f32) -> Vec<Point> {
    let num_brushes = env!("PR_NUM_BRUSHES").parse::<u32>().unwrap();

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

        let an = &mesh.normals[(triangle[0] * 3) as usize..(triangle[0] * 3 + 3) as usize];
        let bn = &mesh.normals[(triangle[1] * 3) as usize..(triangle[1] * 3 + 3) as usize];
        let cn = &mesh.normals[(triangle[2] * 3) as usize..(triangle[2] * 3 + 3) as usize];
        let an = Vector3::new(an[0], an[1], an[2]);
        let bn = Vector3::new(bn[0], bn[1], bn[2]);
        let cn = Vector3::new(cn[0], cn[1], cn[2]);

        let auv = &mesh.texcoords[(triangle[0] * 2) as usize..(triangle[0] * 2 + 2) as usize];
        let buv = &mesh.texcoords[(triangle[1] * 2) as usize..(triangle[1] * 2 + 2) as usize];
        let cuv = &mesh.texcoords[(triangle[2] * 2) as usize..(triangle[2] * 2 + 2) as usize];
        let auv = Vector2::new(auv[0], auv[1]);
        let buv = Vector2::new(buv[0], buv[1]);
        let cuv = Vector2::new(cuv[0], cuv[1]);

        let ab = b - a;
        let ac = c - a;

        let duv_ab = buv - auv;
        let duv_ac = cuv - auv;

        let r = 1.0 / (duv_ab.x * duv_ac.y - duv_ab.y * duv_ac.x);
        let tangent = (ab * duv_ac.y - ac * duv_ab.y) * r;
        let bitangent = (ac * duv_ab.x - ab * duv_ac.x) * r;

        let area = ab.cross(ac).magnitude() / 2.0;
        total_area += area;
        let num_points_f32 = area * density;
        let mut num_points = num_points_f32.floor() as usize;
        let num_points_remainder = num_points_f32 - num_points as f32;

        if rand::random::<f32>() < num_points_remainder {
            num_points += 1;
        }
        for _ in 0..num_points {
            let mut r1 = rand::random();
            let mut r2 = rand::random();
            if r1 + r2 >= 1.0 {
                r1 = 1.0 - r1;
                r2 = 1.0 - r2;
            }

            let p = a + ab * r1 + ac * r2;

            let ap = p - a;
            let bp = p - b;

            let u = (ac.cross(ap).magnitude() / 2.0) / area;
            let v = (ab.cross(bp).magnitude() / 2.0) / area;
            let w = 1.0 - u - v;

            let n = an * u + bn * v + cn * w;
            let uv = auv * u + buv * v + cuv * w;

            points.push(Point {
                position: p.into(),
                normal: n.into(),
                tangent: tangent.into(),
                bitangent: bitangent.into(),
                uv: uv.into(),
                brush_index: (rand::random::<u32>() % num_brushes) as i32,
            })
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
