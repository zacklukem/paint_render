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
use point_gen::gen_point_list;
use tobj::LoadOptions;

use crate::mesh::gen_buffers;

#[derive(Parser, Debug)]
struct Args {
    /// The path to the obj file to view
    #[arg(short, long)]
    obj_file: PathBuf,

    /// Average number of stroke points per unit squared
    #[arg(short, long, default_value = "100.0")]
    stroke_density: f32,
}

const VERTEX_SHADER_SRC: &str = r#"
    #version 330

    uniform mat4 view;
    uniform mat4 perspective;
    uniform mat4 model;

    in vec3 position;
    in vec3 normal;

    out vec3 v_normal;

    void main() {
        gl_Position = perspective * view * model * vec4(position, 1.0);
        v_normal = normal;
    }
"#;

const FRAGMENT_SHADER_SRC: &str = r#"
    #version 330

    out vec4 color;
    in vec3 v_normal;

    void main() {
        color = vec4(1.0, 1.0, 1.0, 1.0) * max(dot(v_normal, vec3(0.0, 0.0, 1.0)), 0.3);
    }
"#;

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
    let program =
        glium::Program::from_source(&display, VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC, None)
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
            (model, buffers, points)
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

        for (_, (vb, ib), _) in &models {
            target
                .draw(
                    vb,
                    ib,
                    &program,
                    &uniforms,
                    &DrawParameters {
                        depth: Depth {
                            test: DepthTest::IfLess,
                            write: true,
                            ..Default::default()
                        },
                        backface_culling: BackfaceCullingMode::CullingDisabled,
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
