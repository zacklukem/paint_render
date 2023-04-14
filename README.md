# Paint Render

 * Project for Purdue CS 334
 * Basic painterly renderer
 * Requirements to Build: Rust, OpenGL

## Usage

### Example Run:

You can also run using pre-compiled binaries at [https://github.com/zacklukem/paint_render/releases](https://github.com/zacklukem/paint_render/releases)

`cargo run -- res/scenes/apple.toml`

To build release target with optimizations (recommended if it runs slow):

`cargo run --release -- res/scenes/apple.toml`

### Example scene file (`res/scenes/apple.toml`):
```toml
# paths in scene are relative to dir containing scene
obj_file = "../models/apple.obj"
albedo_texture = "../textures/apple.png"
stroke_density = 2200
brush_size = 0.04
quantization = 8
background = [0.5, 0.5, 0.5]
saturation = 0.8 # optional
```

### While Running

 * Press `g` to toggle UI overlay
 * Press `v` to toggle points/no paint view
 * Scroll left/right up/down to pan around scene (trackpad recommended)
 * Up arrow to zoom in, down arrow to zoom out
