use std::{path::Path, process::exit, time::Instant};

use glium::{index::NoIndices, Display, IndexBuffer, VertexBuffer};
use log::{error, info};
use tobj::{LoadOptions, Model};

use crate::{
    mesh::{gen_buffers, gen_point_buffers, Vertex},
    point_gen::{gen_point_list, Point},
};

pub struct ModelData {
    #[allow(dead_code)]
    pub model: Model,
    pub model_buffers: (VertexBuffer<Vertex>, IndexBuffer<u32>),
    #[allow(dead_code)]
    pub points: Vec<Point>,
    pub point_buffers: (VertexBuffer<Point>, NoIndices),
}

pub fn gen_models(
    obj_file: impl AsRef<Path>,
    stroke_density: f32,
    display: &Display,
) -> Vec<ModelData> {
    let obj_file = obj_file.as_ref();
    let (models, _materials) = tobj::load_obj(
        obj_file,
        &LoadOptions {
            single_index: true,
            triangulate: true,
            ignore_points: true,
            ignore_lines: true,
        },
    )
    .unwrap_or_else(|e| {
        error!("Failed to load obj file '{}': {e}", obj_file.display());
        exit(1);
    });

    for model in &models {
        info!(
            "Loaded model {} with {} triangles",
            model.name,
            model.mesh.indices.len() / 3,
        );
    }

    // Generate buffers and point lists for each model
    models
        .into_iter()
        .map(|model| {
            let start = Instant::now();
            let points = gen_point_list(&model, stroke_density);
            info!(
                "Generated {} points for model {} ({:?})",
                points.len(),
                model.name,
                start.elapsed()
            );
            let model_buffers = gen_buffers(display, &model.mesh);
            let point_buffers = gen_point_buffers(display, &points);
            ModelData {
                model,
                model_buffers,
                points,
                point_buffers,
            }
        })
        .collect::<Vec<_>>()
}
