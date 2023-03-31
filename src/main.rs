mod camera;
mod mesh;
mod objects;
mod point_gen;

use std::{
    cmp::Reverse,
    collections::HashSet,
    path::PathBuf,
    sync::{
        mpsc::{channel, Sender},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use camera::Camera;
use cgmath::{point3, prelude::*, vec3, vec4, Deg, Matrix4, Vector3};
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
    program::ProgramCreationInput,
    texture::{CompressedSrgbTexture2d, DepthStencilFormat, SrgbTexture2d},
    uniform, BackfaceCullingMode, Blend, Depth, Display, DrawParameters, Program, Surface,
};

use mesh::gen_point_buffers;
use objects::{gen_models, ModelData};
use point_gen::Point;
use rayon::{prelude::IntoParallelRefIterator, slice::ParallelSliceMut};

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

#[derive(Debug, Copy, Clone)]
enum ViewState {
    Raster,
    Full,
}

#[derive(Debug)]
struct State {
    view_state: Mutex<ViewState>,
    wheel_delta: Mutex<Option<f32>>,
    camera: Mutex<Camera>,
    keys: Mutex<HashSet<VirtualKeyCode>>,
    model: Mutex<Matrix4<f32>>,
}

struct DrawData {
    models: Vec<ModelData>,
    color_texture: SrgbTexture2d,
    depth_render_buffer: DepthStencilRenderBuffer,
    color_program: Program,
    point_program: Program,
    brush_stroke: CompressedSrgbTexture2d,
}

fn main() {
    env_logger::init();

    let args = Args::parse();

    let event_loop = EventLoop::new();
    let wb = WindowBuilder::new();
    let cb = ContextBuilder::new().with_depth_buffer(24);
    let display = Display::new(wb, cb, &event_loop).unwrap();

    // Shader programs
    let mut data = init_draw_data(&display, &args);

    // Camera

    let aspect = display.get_framebuffer_dimensions().0 as f32
        / display.get_framebuffer_dimensions().1 as f32;

    let state = Arc::new(State {
        view_state: Mutex::new(ViewState::Full),
        camera: Mutex::new(Camera::new(
            point3(2.0, 2.0, 2.0),
            vec3(-10.0, -10.0, -10.0),
            Deg(100.0),
            aspect,
            0.1,
            10.0,
        )),
        wheel_delta: Mutex::new(None),
        keys: Mutex::new(HashSet::new()),
        model: Mutex::new(Matrix4::identity()),
    });

    let (tx, rx) = channel();

    // Handle fixed time loop
    fixed_update(
        state.clone(),
        data.models.iter().map(|p| p.points.clone()).collect(),
        tx,
    );

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
                        if key == VirtualKeyCode::V {
                            let mut view = state.view_state.lock().unwrap();
                            *view = match *view {
                                ViewState::Full => ViewState::Raster,
                                ViewState::Raster => ViewState::Full,
                            };
                        }
                        state.keys.lock().unwrap().insert(key);
                    } else {
                        state.keys.lock().unwrap().remove(&key);
                    }
                }
                WindowEvent::MouseWheel {
                    phase: TouchPhase::Ended,
                    ..
                } => {
                    *state.wheel_delta.lock().unwrap() = None;
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
                    *state.wheel_delta.lock().unwrap() = Some(delta);
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

        {
            let mut last_points = None;
            while let Ok(points) = rx.try_recv() {
                last_points = Some(points);
            }
            if let Some(points) = last_points {
                for (i, points) in points.into_iter().enumerate() {
                    data.models[i].point_buffers = gen_point_buffers(&display, &points);
                    data.models[i].points = points;
                }
            }
        }

        draw(&state, &display, &data);
    });
}

fn init_draw_data(display: &Display, args: &Args) -> DrawData {
    let color_program =
        Program::from_source(display, shaders::COLOR_VERT, shaders::COLOR_FRAG, None).unwrap();

    let point_program = Program::new(
        display,
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
    let brush_stroke = CompressedSrgbTexture2d::new(display, brush_stroke).unwrap();

    let models = gen_models(&args.obj_file, args.stroke_density, display);

    let depth_render_buffer = DepthStencilRenderBuffer::new(
        display,
        DepthStencilFormat::I24I8,
        display.get_framebuffer_dimensions().0,
        display.get_framebuffer_dimensions().1,
    )
    .unwrap();

    let color_texture = SrgbTexture2d::empty(
        display,
        display.get_framebuffer_dimensions().0,
        display.get_framebuffer_dimensions().1,
    )
    .unwrap();

    DrawData {
        color_program,
        point_program,
        brush_stroke,
        models,
        depth_render_buffer,
        color_texture,
    }
}

fn fixed_update(
    state: Arc<State>,
    mut points_m: Vec<Vec<Point>>,
    points_sender: Sender<Vec<Vec<Point>>>,
) {
    thread::spawn(move || {
        let mut reverse_sort = true;
        let mut changed = true;
        loop {
            let start = Instant::now();
            {
                let wheel_delta = state.wheel_delta.lock().unwrap();
                let keys = state.keys.lock().unwrap();
                let mut model = state.model.lock().unwrap();
                let mut camera = state.camera.lock().unwrap();
                if let Some(wheel_delta) = *wheel_delta {
                    camera.zoom(wheel_delta * 0.01);
                    // Disable update on mouse wheel because it's too slow
                    // changed = true;
                }
                if keys.contains(&VirtualKeyCode::Left) {
                    *model = Matrix4::from_angle_y(Deg(-0.3)) * *model;
                    changed = true;
                }
                if keys.contains(&VirtualKeyCode::Right) {
                    *model = Matrix4::from_angle_y(Deg(0.3)) * *model;
                    changed = true;
                }
                if keys.contains(&VirtualKeyCode::Up) {
                    *model = Matrix4::from_angle_x(Deg(-0.3)) * *model;
                    changed = true;
                }
                if keys.contains(&VirtualKeyCode::Down) {
                    *model = Matrix4::from_angle_x(Deg(0.3)) * *model;
                    changed = true;
                }
                if keys.contains(&VirtualKeyCode::R) {
                    reverse_sort = !reverse_sort;
                    changed = true;
                }
                if changed {
                    changed = false;
                    let model = *model;
                    let view = Matrix4::from(camera.view());
                    let perspective = Matrix4::from(camera.perspective());

                    #[derive(PartialOrd, PartialEq)]
                    #[repr(transparent)]
                    struct Ord<T>(T);

                    impl std::cmp::Eq for Ord<f32> {}

                    impl std::cmp::Ord for Ord<f32> {
                        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                            self.partial_cmp(other).unwrap().reverse()
                        }
                    }

                    for points in &mut points_m {
                        if reverse_sort {
                            points.par_sort_by_cached_key(|p| {
                                let p = perspective
                                    * view
                                    * model
                                    * vec4(p.position[0], p.position[1], p.position[2], 1.0);
                                Reverse(Ord(p.z / p.w))
                            });
                        } else {
                            points.par_sort_by_cached_key(|p| {
                                let p = perspective
                                    * view
                                    * model
                                    * vec4(p.position[0], p.position[1], p.position[2], 1.0);
                                Ord(p.z / p.w)
                            });
                        }
                    }

                    points_sender.send(points_m.clone()).unwrap();
                }
            }
            thread::sleep(Duration::from_millis(16).saturating_sub(start.elapsed()));
        }
    });
}

fn draw_model(target: &mut impl Surface, state: &State, data: &DrawData, model: [[f32; 4]; 4]) {
    let camera_uniforms = {
        let camera = state.camera.lock().unwrap();
        uniform! {
            view: camera.view(),
            perspective: camera.perspective(),
            model: model,
        }
    };

    target.clear_color_and_depth((0.0, 0.0, 0.0, 1.0), 1.0);

    for model in &data.models {
        let (vb, ib) = &model.model_buffers;
        target
            .draw(
                vb,
                ib,
                &data.color_program,
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

fn draw_points(target: &mut impl Surface, state: &State, data: &DrawData, model: [[f32; 4]; 4]) {
    let camera_uniforms = {
        let camera = state.camera.lock().unwrap();
        uniform! {
            view: camera.view(),
            perspective: camera.perspective(),
            model: model,
            color_texture: &data.color_texture,
            brush_stroke: &data.brush_stroke,
        }
    };

    for model in &data.models {
        let (vb, ib) = &model.point_buffers;
        target
            .draw(
                vb,
                ib,
                &data.point_program,
                &camera_uniforms,
                &DrawParameters {
                    blend: Blend::alpha_blending(),
                    ..Default::default()
                },
            )
            .unwrap();
    }
}

fn draw(state: &State, display: &Display, data: &DrawData) {
    let model: [[f32; 4]; 4] = { <Matrix4<f32> as Into<_>>::into(*state.model.lock().unwrap()) };
    let view_state = { *state.view_state.lock().unwrap() };

    match view_state {
        ViewState::Full => {
            // render color buffer
            {
                let mut target = SimpleFrameBuffer::with_depth_stencil_buffer(
                    display,
                    &data.color_texture,
                    &data.depth_render_buffer,
                )
                .unwrap();

                draw_model(&mut target, state, data, model);
            }

            // render points
            {
                let mut target = display.draw();
                target.clear_color_and_depth((0.0, 0.0, 0.0, 1.0), 1.0);

                draw_points(&mut target, state, data, model);

                target.finish().unwrap();
            }
        }
        ViewState::Raster => {
            let mut target = display.draw();

            draw_model(&mut target, state, data, model);

            target.finish().unwrap();
        }
    }
}
