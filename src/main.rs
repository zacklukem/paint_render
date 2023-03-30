mod camera;
mod mesh;
mod point_gen;

use std::{
    path::PathBuf,
    process::exit,
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use camera::Camera;
use cgmath::{point3, prelude::*, vec3, Deg, Matrix4};
use clap::Parser;
use glium::{
    draw_parameters::DepthTest,
    glutin::{
        event::{Event, MouseScrollDelta, TouchPhase, WindowEvent},
        event_loop::EventLoop,
        window::WindowBuilder,
        ContextBuilder,
    },
    uniform, BackfaceCullingMode, Depth, Display, DrawParameters, Surface,
};
use log::{error, info};
use mesh::debug_points;
use point_gen::gen_point_list;
use tobj::LoadOptions;

use crate::mesh::gen_buffers;

#[derive(Parser, Debug)]
struct Args {
    /// The path to the obj file to view
    #[arg(short, long)]
    obj_file: PathBuf,

    /// Average number of stroke points per unit squared
    #[arg(short, long, default_value = "3000.0")]
    stroke_density: f32,
}

mod shaders {
    pub const COLOR_VERT: &str = include_str!("./shaders/color.vert");
    pub const COLOR_FRAG: &str = include_str!("./shaders/color.frag");
    pub const POINT_VERT: &str = include_str!("./shaders/point.vert");
    pub const POINT_FRAG: &str = include_str!("./shaders/point.frag");
}

fn main() {
    env_logger::init();

    let args = Args::parse();

    let (models, _materials) = tobj::load_obj(
        &args.obj_file,
        &LoadOptions {
            single_index: true,
            triangulate: true,
            ignore_points: true,
            ignore_lines: true,
        },
    )
    .unwrap_or_else(|e| {
        error!("Failed to load obj file '{}': {e}", args.obj_file.display());
        exit(1);
    });

    for model in &models {
        info!(
            "Loaded model {} with {} triangles",
            model.name,
            model.mesh.indices.len() / 3,
        );
    }

    let event_loop = EventLoop::new();
    let wb = WindowBuilder::new();
    let cb = ContextBuilder::new().with_depth_buffer(24);
    let display = Display::new(wb, cb, &event_loop).unwrap();

    // Shader program
    let color_program =
        glium::Program::from_source(&display, shaders::COLOR_VERT, shaders::COLOR_FRAG, None)
            .unwrap();

    let point_program =
        glium::Program::from_source(&display, shaders::POINT_VERT, shaders::POINT_FRAG, None)
            .unwrap();

    // Generate buffers and point lists for each model
    let models = models
        .into_iter()
        .map(|model| {
            let start = Instant::now();
            let points = gen_point_list(&model, args.stroke_density);
            info!(
                "Generated {} points for model {} ({:?})",
                points.len(),
                model.name,
                start.elapsed()
            );
            let buffers = gen_buffers(&display, &model.mesh);
            let point_buffer = debug_points(&display, &points);
            (model, buffers, points, point_buffer)
        })
        .collect::<Vec<_>>();

    let aspect = display.gl_window().window().inner_size().width as f32
        / display.gl_window().window().inner_size().height as f32;

    let camera = Arc::new(Mutex::new(Camera::new(
        point3(10.0, 10.0, 10.0),
        vec3(-10.0, -10.0, -10.0),
        Deg(100.0),
        aspect,
        0.01,
        100.0,
    )));

    let model = Arc::new(Mutex::new(Matrix4::identity()));
    let wheel_delta = Arc::new(Mutex::new(None));

    // Handle constant time loop
    {
        let wheel_delta = wheel_delta.clone();
        let camera = camera.clone();
        let model = model.clone();

        thread::spawn(move || loop {
            let start = Instant::now();
            {
                let wheel_delta = wheel_delta.lock().unwrap();
                let mut camera = camera.lock().unwrap();
                if let Some(wheel_delta) = *wheel_delta {
                    camera.zoom(wheel_delta * 0.01);
                }
                let mut model = model.lock().unwrap();
                *model = Matrix4::from_angle_y(Deg(0.3)) * *model;
            }
            thread::sleep(Duration::from_millis(16).saturating_sub(start.elapsed()));
        });
    }

    event_loop.run(move |ev, _, control_flow| {
        let mut target = display.draw();
        target.clear_color_and_depth((0.0, 0.0, 0.0, 1.0), 1.0);

        let model: [[f32; 4]; 4] = { <Matrix4<f32> as Into<_>>::into(*model.lock().unwrap()) };

        let uniforms = {
            let camera = camera.lock().unwrap();
            uniform! {
                view: camera.view(),
                perspective: camera.perspective(),
                model: model,
            }
        };

        for (_, (vb, ib), _, (point_vb, point_ib)) in &models {
            target
                .draw(
                    vb,
                    ib,
                    &color_program,
                    &uniforms,
                    &DrawParameters {
                        depth: Depth {
                            test: DepthTest::IfLess,
                            write: true,
                            ..Default::default()
                        },
                        backface_culling: BackfaceCullingMode::CullClockwise,
                        ..Default::default()
                    },
                )
                .unwrap();

            target
                .draw(
                    point_vb,
                    point_ib,
                    &point_program,
                    &uniforms,
                    &DrawParameters {
                        point_size: Some(10.0),
                        depth: Depth {
                            test: DepthTest::IfLess,
                            write: false,
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                )
                .unwrap();
        }

        target.finish().unwrap();

        let next_frame_time = Instant::now() + Duration::from_nanos(16_666_667);
        control_flow.set_wait_until(next_frame_time);
        match ev {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    control_flow.set_exit();
                    return;
                }
                WindowEvent::MouseWheel {
                    phase: TouchPhase::Ended,
                    ..
                } => {
                    *wheel_delta.lock().unwrap() = None;
                }
                WindowEvent::MouseWheel {
                    delta,
                    phase: TouchPhase::Moved | TouchPhase::Started,
                    ..
                } => {
                    let delta = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(pos) => pos.y as f32,
                    };
                    *wheel_delta.lock().unwrap() = Some(delta);
                }
                _ => return,
            },
            _ => (),
        }
    });
}
