mod camera;
mod mesh;
mod point_gen;

use std::{
    collections::HashSet,
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
    framebuffer::{DepthStencilRenderBuffer, SimpleFrameBuffer},
    glutin::{
        event::{
            ElementState, Event, MouseScrollDelta, StartCause, TouchPhase, VirtualKeyCode,
            WindowEvent,
        },
        event_loop::EventLoop,
        window::WindowBuilder,
        ContextBuilder,
    },
    index::NoIndices,
    program::ProgramCreationInput,
    texture::{CompressedSrgbTexture2d, DepthStencilFormat, SrgbTexture2d},
    uniform, BackfaceCullingMode, Blend, Depth, Display, DrawParameters, IndexBuffer, Program,
    Surface, VertexBuffer,
};
use log::{error, info};
use mesh::{debug_points, Vertex};
use point_gen::{gen_point_list, Point};
use tobj::{LoadOptions, Model};

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
    pub const POINT_GEOM: &str = include_str!("./shaders/point.geom");
    pub const POINT_FRAG: &str = include_str!("./shaders/point.frag");
}

fn main() {
    env_logger::init();

    let args = Args::parse();

    let event_loop = EventLoop::new();
    let wb = WindowBuilder::new();
    let cb = ContextBuilder::new().with_depth_buffer(24);
    let display = Display::new(wb, cb, &event_loop).unwrap();

    // Shader programs
    let color_program =
        Program::from_source(&display, shaders::COLOR_VERT, shaders::COLOR_FRAG, None).unwrap();

    let point_program = Program::new(
        &display,
        ProgramCreationInput::SourceCode {
            vertex_shader: shaders::POINT_VERT,
            fragment_shader: shaders::POINT_FRAG,
            geometry_shader: Some(shaders::POINT_GEOM),
            tessellation_control_shader: None,
            tessellation_evaluation_shader: None,
            transform_feedback_varyings: None,
            outputs_srgb: false,
            uses_point_size: true,
        },
    )
    .unwrap();

    let brush_stroke = image::open("textures/0.png").unwrap().into_rgba8();
    let image_dimensions = brush_stroke.dimensions();
    let brush_stroke = glium::texture::RawImage2d::from_raw_rgba_reversed(
        &brush_stroke.into_raw(),
        image_dimensions,
    );
    let brush_stroke = CompressedSrgbTexture2d::new(&display, brush_stroke).unwrap();

    let models = gen_models(args, &display);

    // Camera

    let aspect = display.get_framebuffer_dimensions().0 as f32
        / display.get_framebuffer_dimensions().1 as f32;

    let camera = Arc::new(Mutex::new(Camera::new(
        point3(2.0, 2.0, 2.0),
        vec3(-10.0, -10.0, -10.0),
        Deg(100.0),
        aspect,
        0.1,
        10.0,
    )));

    let model = Arc::new(Mutex::new(Matrix4::identity()));
    let wheel_delta = Arc::new(Mutex::new(None));
    let keys = Arc::new(Mutex::new(HashSet::new()));

    // Handle fixed time loop
    {
        let wheel_delta = wheel_delta.clone();
        let camera = camera.clone();
        let model = model.clone();
        let keys = keys.clone();

        thread::spawn(move || loop {
            let start = Instant::now();
            {
                let wheel_delta = wheel_delta.lock().unwrap();
                let mut camera = camera.lock().unwrap();
                let keys = keys.lock().unwrap();
                if let Some(wheel_delta) = *wheel_delta {
                    camera.zoom(wheel_delta * 0.01);
                }
                let mut model = model.lock().unwrap();
                if keys.contains(&VirtualKeyCode::Left) {
                    *model = Matrix4::from_angle_y(Deg(-0.3)) * *model;
                }
                if keys.contains(&VirtualKeyCode::Right) {
                    *model = Matrix4::from_angle_y(Deg(0.3)) * *model;
                }
                if keys.contains(&VirtualKeyCode::Up) {
                    *model = Matrix4::from_angle_x(Deg(-0.3)) * *model;
                }
                if keys.contains(&VirtualKeyCode::Down) {
                    *model = Matrix4::from_angle_x(Deg(0.3)) * *model;
                }
            }
            thread::sleep(Duration::from_millis(16).saturating_sub(start.elapsed()));
        });
    }

    let depth_render_buffer = DepthStencilRenderBuffer::new(
        &display,
        DepthStencilFormat::I24I8,
        display.get_framebuffer_dimensions().0,
        display.get_framebuffer_dimensions().1,
    )
    .unwrap();

    let color_texture = SrgbTexture2d::empty(
        &display,
        display.get_framebuffer_dimensions().0,
        display.get_framebuffer_dimensions().1,
    )
    .unwrap();

    event_loop.run(move |ev, _, control_flow| {
        match ev {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    control_flow.set_exit();
                    return;
                }
                WindowEvent::KeyboardInput { input, .. } => {
                    let key = input.virtual_keycode.unwrap();
                    if input.state == ElementState::Pressed {
                        keys.lock().unwrap().insert(key);
                    } else {
                        keys.lock().unwrap().remove(&key);
                    }
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
                    {
                        *wheel_delta.lock().unwrap() = Some(delta);
                    }
                }
                _ => return,
            },
            Event::NewEvents(cause) => match cause {
                StartCause::ResumeTimeReached { .. } => (),
                StartCause::Init => (),
                _ => return,
            },
            _ => return,
        }

        let next_frame_time = Instant::now() + Duration::from_nanos(16_666_667);
        control_flow.set_wait_until(next_frame_time);

        draw(
            &model,
            &camera,
            &display,
            &color_texture,
            &depth_render_buffer,
            &models,
            &color_program,
            &point_program,
            &brush_stroke,
        );
    });
}

struct ModelData {
    #[allow(dead_code)]
    model: Model,
    model_buffers: (VertexBuffer<Vertex>, IndexBuffer<u32>),
    #[allow(dead_code)]
    points: Vec<Point>,
    point_buffers: (VertexBuffer<Point>, NoIndices),
}

fn draw(
    model: &Mutex<Matrix4<f32>>,
    camera: &Mutex<Camera>,
    display: &Display,
    color_texture: &SrgbTexture2d,
    depth_render_buffer: &DepthStencilRenderBuffer,
    models: &[ModelData],
    color_program: &Program,
    point_program: &Program,
    brush_stroke: &CompressedSrgbTexture2d,
) {
    let model: [[f32; 4]; 4] = { <Matrix4<f32> as Into<_>>::into(*model.lock().unwrap()) };

    // render color buffer
    {
        let camera_uniforms = {
            let camera = camera.lock().unwrap();
            uniform! {
                view: camera.view(),
                perspective: camera.perspective(),
                model: model,
            }
        };
        let mut target = SimpleFrameBuffer::with_depth_stencil_buffer(
            display,
            color_texture,
            depth_render_buffer,
        )
        .unwrap();

        target.clear_color_and_depth((0.0, 0.0, 0.0, 1.0), 1.0);

        for model in models {
            let (vb, ib) = &model.model_buffers;
            target
                .draw(
                    vb,
                    ib,
                    color_program,
                    &camera_uniforms,
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
        }
    }

    {
        let mut target = display.draw();
        target.clear_color_and_depth((0.0, 0.0, 0.0, 1.0), 1.0);

        let camera_uniforms = {
            let camera = camera.lock().unwrap();
            uniform! {
                view: camera.view(),
                perspective: camera.perspective(),
                model: model,
                color_texture: color_texture,
                brush_stroke: brush_stroke,
            }
        };

        for model in models {
            let (vb, ib) = &model.point_buffers;
            target
                .draw(
                    vb,
                    ib,
                    point_program,
                    &camera_uniforms,
                    &DrawParameters {
                        blend: Blend::alpha_blending(),
                        ..Default::default()
                    },
                )
                .unwrap();
        }

        target.finish().unwrap();
    }
}

fn gen_models(args: Args, display: &Display) -> Vec<ModelData> {
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

    // Generate buffers and point lists for each model
    models
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
            let model_buffers = gen_buffers(display, &model.mesh);
            let point_buffers = debug_points(display, &points);
            ModelData {
                model,
                model_buffers,
                points,
                point_buffers,
            }
        })
        .collect::<Vec<_>>()
}
