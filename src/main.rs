#![feature(path_file_prefix)]

mod camera;
mod mesh;
mod objects;
mod point_gen;
mod running_average;

use std::{
    cmp::Reverse,
    collections::HashSet,
    fs,
    io::Cursor,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::{channel, Sender},
        Arc, Mutex,
    },
    thread,
    time::{Duration, Instant},
};

use camera::Camera;
use cgmath::{point3, prelude::*, vec3, vec4, Deg, Matrix4, Point3, Vector4};
use clap::Parser;
use egui::{SidePanel, Slider};
use egui_glium::EguiGlium;
use glium::{
    draw_parameters::DepthTest,
    framebuffer::{DepthStencilRenderBuffer, SimpleFrameBuffer},
    glutin::{
        dpi::PhysicalSize,
        event::{
            ElementState, Event, MouseScrollDelta, StartCause, TouchPhase, VirtualKeyCode,
            WindowEvent,
        },
        event_loop::EventLoop,
        window::WindowBuilder,
        ContextBuilder,
    },
    implement_vertex,
    index::{NoIndices, PrimitiveType},
    program::ProgramCreationInput,
    texture::{CompressedSrgbTexture2d, DepthStencilFormat, SrgbTexture2d},
    uniform, BackfaceCullingMode, Blend, Depth, Display, DrawParameters, IndexBuffer, Program,
    Surface, VertexBuffer,
};

use image::io::Reader as ImageReader;
use mesh::gen_point_buffers;
use objects::{gen_models, ModelData};
use point_gen::Point;
use rayon::slice::ParallelSliceMut;
use running_average::RunningAverage;
use serde::Deserialize;

#[derive(Parser, Debug)]
struct Args {
    /// The path to the obj file to view
    scene: PathBuf,
}

const BRUSHES_PNG: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/brushes.png"));

mod shaders {

    macro_rules! include_shader {
        ($file: expr) => {
            concat!(
                "#version 330\n",
                "#define PR_NUM_BRUSHES ",
                env!("PR_NUM_BRUSHES"),
                "\n",
                include_str!($file)
            )
        };
    }

    pub const POST_VERT: &str = include_shader!("./shaders/post.vert");
    pub const POST_FRAG: &str = include_shader!("./shaders/post.frag");

    pub const COLOR_VERT: &str = include_shader!("./shaders/color.vert");
    pub const COLOR_FRAG: &str = include_shader!("./shaders/color.frag");

    pub const POINT_VERT: &str = include_shader!("./shaders/point.vert");
    pub const POINT_GEOM: &str = include_shader!("./shaders/point.geom");
    pub const POINT_FRAG: &str = include_shader!("./shaders/point.frag");
}

#[derive(Debug)]
struct DebugInfo {
    /// Draw time in microseconds
    draw_time: AtomicU64,
    /// Sort time in microseconds
    sort_time: AtomicU64,
    /// Fixed time in microseconds
    fixed_time: AtomicU64,
}

#[derive(Debug, Deserialize)]
struct Scene {
    obj_file: PathBuf,
    albedo_texture: PathBuf,
    stroke_density: f32,
    brush_size: f32,
    quantization: i32,
}

#[derive(Debug, Copy, Clone)]
enum ViewState {
    Raster,
    Full,
}

#[derive(Debug)]
struct State {
    view_state: Mutex<ViewState>,
    wheel_delta: Mutex<Option<(f32, f32)>>,
    camera: Mutex<Camera>,
    keys: Mutex<HashSet<VirtualKeyCode>>,
    model: Mutex<Matrix4<f32>>,
    enable_gui: AtomicBool,
    debug_info: DebugInfo,
}

struct DrawData {
    models: Vec<ModelData>,
    albedo_texture: CompressedSrgbTexture2d,
    post_process_texture: SrgbTexture2d,
    color_texture: SrgbTexture2d,
    #[allow(dead_code)]
    depth_render_buffer: DepthStencilRenderBuffer,
    color_program: Program,
    point_program: Program,
    post_process_program: Program,
    brush_stroke: CompressedSrgbTexture2d,
    post_process_quad: (VertexBuffer<PostProcessVert>, IndexBuffer<u8>),
    params: Params,
}

struct Params {
    quantization: i32,
    brush_size: f32,
}

#[derive(Copy, Clone)]
struct PostProcessVert {
    position: [f32; 2],
}
implement_vertex!(PostProcessVert, position);

fn main() {
    env_logger::init();

    let args = Args::parse();

    let event_loop = EventLoop::new();
    let wb = WindowBuilder::new().with_inner_size(PhysicalSize::new(2880, 1800));
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
        enable_gui: AtomicBool::new(true),
        debug_info: DebugInfo {
            draw_time: AtomicU64::new(0),
            sort_time: AtomicU64::new(0),
            fixed_time: AtomicU64::new(0),
        },
    });

    let mut egui_glium = EguiGlium::new(&display, &event_loop);
    // let mut color_test = ;

    let (tx, rx) = channel();

    // Handle fixed time loop
    fixed_update(
        state.clone(),
        data.models.iter().map(|p| p.points.clone()).collect(),
        tx,
    );

    let mut sort_time_average = RunningAverage::<f64, 32>::new();
    let mut draw_time_average = RunningAverage::<f64, 32>::new();
    let mut fixed_time_average = RunningAverage::<f64, 32>::new();
    let mut true_frame_time_average = RunningAverage::<f64, 32>::new();

    let mut true_frame_time_start = Instant::now();
    let mut true_frame_time = Duration::ZERO;

    event_loop.run(move |ev, _, control_flow| {
        match ev {
            Event::WindowEvent { event, .. } => {
                let response = egui_glium.on_event(&event);
                if !response.consumed {
                    match event {
                        WindowEvent::Resized(size) => {
                            let aspect = size.width as f32 / size.height as f32;
                            let mut camera = state.camera.lock().unwrap();
                            camera.set_aspect(aspect);
                            return;
                        }
                        WindowEvent::CloseRequested => {
                            control_flow.set_exit();
                            return;
                        }
                        WindowEvent::KeyboardInput { input, .. } => {
                            let key = input.virtual_keycode.unwrap();
                            if input.state == ElementState::Pressed {
                                match key {
                                    VirtualKeyCode::V => {
                                        let mut view = state.view_state.lock().unwrap();
                                        *view = match *view {
                                            ViewState::Full => ViewState::Raster,
                                            ViewState::Raster => ViewState::Full,
                                        };
                                    }
                                    VirtualKeyCode::G => {
                                        let v = state.enable_gui.load(Ordering::Acquire);
                                        state.enable_gui.store(!v, Ordering::Release);
                                    }
                                    _ => (),
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
                                MouseScrollDelta::LineDelta(x, y) => (x, y),
                                MouseScrollDelta::PixelDelta(pos) => (pos.x as f32, pos.y as f32),
                            };
                            *state.wheel_delta.lock().unwrap() = Some(delta);
                        }
                        _ => return,
                    };
                }
            }
            Event::NewEvents(cause) => match cause {
                StartCause::ResumeTimeReached { .. } => (),
                StartCause::Init => (),
                _ => return,
            },
            _ => return,
        }

        let next_frame_time = Instant::now() + Duration::from_nanos(16_666_667);
        control_flow.set_wait_until(next_frame_time);

        let start = Instant::now();

        // UI
        if state.enable_gui.load(Ordering::Relaxed) {
            sort_time_average
                .add(state.debug_info.sort_time.load(Ordering::Relaxed) as f64 / 1000.0);
            fixed_time_average
                .add(state.debug_info.fixed_time.load(Ordering::Relaxed) as f64 / 1000.0);
            draw_time_average
                .add(state.debug_info.draw_time.load(Ordering::Relaxed) as f64 / 1000.0);

            true_frame_time_average.add(true_frame_time.as_secs_f64());

            egui_glium.run(&display, |egui_ctx| {
                SidePanel::left("my_side_panel").show(egui_ctx, |ui| {
                    ui.add(Slider::new(&mut data.params.quantization, 0..=20).text("Quantization"));
                    ui.add(
                        Slider::new(&mut data.params.brush_size, 0.01..=0.08).text("Brush Size"),
                    );

                    ui.label(format!("Draw time: {:.3} ms", draw_time_average.average()));

                    ui.label(format!(
                        "Fixed time: {:.3} ms",
                        fixed_time_average.average()
                    ));

                    ui.label(format!("Sort time: {:.3} ms", sort_time_average.average()));

                    ui.label(format!(
                        "FPS: {:.3} fps",
                        1.0 / true_frame_time_average.average()
                    ));
                });
            });
        }

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

        state
            .debug_info
            .draw_time
            .store(start.elapsed().as_micros() as u64, Ordering::Release);

        draw(&state, &display, &data, &mut egui_glium);

        true_frame_time = true_frame_time_start.elapsed();
        true_frame_time_start = Instant::now();
    });
}

fn init_draw_data(display: &Display, args: &Args) -> DrawData {
    let scene: Scene = toml::from_str(&fs::read_to_string(&args.scene).unwrap()).unwrap();
    let scene_base_dir = args.scene.parent().unwrap();

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

    let post_process_program =
        Program::from_source(display, shaders::POST_VERT, shaders::POST_FRAG, None).unwrap();

    let brush_stroke = ImageReader::new(Cursor::new(BRUSHES_PNG))
        .with_guessed_format()
        .unwrap()
        .decode()
        .unwrap()
        .into_rgba8();
    let image_dimensions = brush_stroke.dimensions();
    let brush_stroke = glium::texture::RawImage2d::from_raw_rgba_reversed(
        &brush_stroke.into_raw(),
        image_dimensions,
    );
    let brush_stroke = CompressedSrgbTexture2d::new(display, brush_stroke).unwrap();

    let albedo_texture = image::open(scene_base_dir.join(scene.albedo_texture))
        .unwrap()
        .into_rgba8();
    let image_dimensions = albedo_texture.dimensions();
    let albedo_texture = glium::texture::RawImage2d::from_raw_rgba_reversed(
        &albedo_texture.into_raw(),
        image_dimensions,
    );
    let albedo_texture = CompressedSrgbTexture2d::new(display, albedo_texture).unwrap();

    let models = gen_models(
        scene_base_dir.join(scene.obj_file),
        scene.stroke_density,
        display,
    );

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

    let post_process_texture = SrgbTexture2d::empty(
        display,
        display.get_framebuffer_dimensions().0,
        display.get_framebuffer_dimensions().1,
    )
    .unwrap();

    let params = Params {
        quantization: scene.quantization,
        brush_size: scene.brush_size,
    };

    let post_quad_vert = vec![
        PostProcessVert {
            // BL
            position: [-1.0, -1.0],
        },
        PostProcessVert {
            // BR
            position: [1.0, -1.0],
        },
        PostProcessVert {
            // TR
            position: [1.0, 1.0],
        },
        PostProcessVert {
            // TL
            position: [-1.0, 1.0],
        },
    ];

    // TL ---- TR
    // |  ^     ^
    // v   \    |
    // BL >--- BR

    let post_quad_indices = vec![0u8, 1, 3, 1, 2, 3];

    let post_quad_vertex_buffer = VertexBuffer::new(display, &post_quad_vert).unwrap();
    let post_quad_index_buffer =
        IndexBuffer::new(display, PrimitiveType::TrianglesList, &post_quad_indices).unwrap();

    DrawData {
        color_program,
        point_program,
        brush_stroke,
        albedo_texture,
        models,
        depth_render_buffer,
        color_texture,
        post_process_quad: (post_quad_vertex_buffer, post_quad_index_buffer),
        post_process_texture,
        post_process_program,
        params,
    }
}

fn fixed_update(
    state: Arc<State>,
    mut points_m: Vec<Vec<Point>>,
    points_sender: Sender<Vec<Vec<Point>>>,
) {
    let latest = Arc::new(Mutex::new(
        None::<(Matrix4<f32>, Matrix4<f32>, Matrix4<f32>, bool)>,
    ));

    {
        let latest = latest.clone();
        let state = state.clone();
        thread::spawn(move || loop {
            let latest = { *latest.lock().unwrap() };
            let elapsed = if let Some((model, view, perspective, reverse_sort)) = latest {
                let start = Instant::now();
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
                            let p: Vector4<f32> = perspective
                                * view
                                * model
                                * vec4(p.position[0], p.position[1], p.position[2], 1.0);
                            Reverse(Ord(p.z / p.w))
                        });
                    } else {
                        points.par_sort_by_cached_key(|p| {
                            let p: Vector4<f32> = perspective
                                * view
                                * model
                                * vec4(p.position[0], p.position[1], p.position[2], 1.0);
                            Ord(p.z / p.w)
                        });
                    }
                }

                points_sender.send(points_m.clone()).unwrap();
                let elapsed = start.elapsed();
                state
                    .debug_info
                    .sort_time
                    .store(elapsed.as_micros() as u64, Ordering::Relaxed);
                elapsed
            } else {
                Duration::ZERO
            };
            // TODO: this is awful
            thread::sleep(Duration::from_millis(17).saturating_sub(elapsed));
        });
    }

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
                    *model = Matrix4::from_angle_y(Deg(0.3 * wheel_delta.0)) * *model;
                    camera.rotate_up(Deg(-0.3 * wheel_delta.1));
                    // Disable update on mouse wheel because it's too slow
                    changed = true;
                }
                if keys.contains(&VirtualKeyCode::Up) {
                    camera.zoom(0.01);
                }
                if keys.contains(&VirtualKeyCode::Down) {
                    camera.zoom(-0.01);
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
                    {
                        if let Ok(mut lock) = latest.try_lock() {
                            *lock = Some((model, view, perspective, reverse_sort));
                        }
                    }
                }
            }
            let elapsed = start.elapsed();
            state
                .debug_info
                .fixed_time
                .store(elapsed.as_micros() as u64, Ordering::Relaxed);
            thread::sleep(Duration::from_millis(16).saturating_sub(elapsed));
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
            albedo_texture: &data.albedo_texture,
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
            albedo_texture: &data.albedo_texture,
            brush_stroke: &data.brush_stroke,
            camera_pos: <Point3<_> as Into<[f32; 3]>>::into(camera.position()),
            quantization: data.params.quantization,
            brush_size: data.params.brush_size,
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

fn draw(state: &State, display: &Display, data: &DrawData, egui_glium: &mut EguiGlium) {
    let model: [[f32; 4]; 4] = { <Matrix4<f32> as Into<_>>::into(*state.model.lock().unwrap()) };
    let view_state = { *state.view_state.lock().unwrap() };

    match view_state {
        ViewState::Full => {
            // TODO: fix depth buffer
            // render color buffer
            // {
            //     let mut target = SimpleFrameBuffer::with_depth_stencil_buffer(
            //         display,
            //         &data.color_texture,
            //         &data.depth_render_buffer,
            //     )
            //     .unwrap();

            //     draw_model(&mut target, state, data, model);
            // }

            // render points
            {
                // let mut target = display.draw();
                let mut target =
                    SimpleFrameBuffer::new(display, &data.post_process_texture).unwrap();

                target.clear_color_and_depth((0.0, 0.0, 0.0, 1.0), 1.0);

                draw_points(&mut target, state, data, model);

                // target.finish().unwrap();
            }

            {
                let mut target = display.draw();

                target.clear_color(0.0, 0.0, 0.0, 1.0);

                target
                    .draw(
                        &data.post_process_quad.0,
                        &data.post_process_quad.1,
                        &data.post_process_program,
                        &uniform! {
                            post_process_texture: &data.post_process_texture,
                        },
                        &DrawParameters::default(),
                    )
                    .unwrap();

                if state.enable_gui.load(Ordering::Relaxed) {
                    egui_glium.paint(display, &mut target);
                }

                target.finish().unwrap();
            }
        }
        ViewState::Raster => {
            let mut target = display.draw();

            draw_model(&mut target, state, data, model);

            if state.enable_gui.load(Ordering::Relaxed) {
                egui_glium.paint(display, &mut target);
            }

            target.finish().unwrap();
        }
    }
}
