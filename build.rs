use image::RgbImage;
use std::fs;

const BRUSH_DIM: u32 = 320;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let brush_dir = format!("{}/textures/brushes", manifest_dir);
    println!("cargo:rerun-if-changed={}", brush_dir);
    let brushes = fs::read_dir(brush_dir)
        .unwrap()
        .map(|dir| dir.unwrap().path())
        .filter(|p| p.file_name().unwrap().to_string_lossy() != ".DS_Store")
        .collect::<Vec<_>>();
    let out_image_width = BRUSH_DIM * brushes.len() as u32;
    let out_image_height = BRUSH_DIM;
    let mut out_image = RgbImage::new(out_image_width, out_image_height);
    out_image.fill(0xff);

    let num_brushes = brushes.len();

    for (i, brush) in brushes.into_iter().enumerate() {
        println!("cargo:rerun-if-changed={}", brush.to_string_lossy());
        let brush = image::open(brush).unwrap().into_rgb8();
        assert_eq!(brush.width(), BRUSH_DIM);
        let x_offset = i as u32 * BRUSH_DIM;
        let y_offset = (BRUSH_DIM - brush.height()) / 2;

        for (x0, y0, p) in brush.enumerate_pixels() {
            out_image.put_pixel(x_offset + x0, y_offset + y0, *p);
        }
    }

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let out_file = format!("{}/brushes.png", out_dir);
    out_image.save(out_file).unwrap();

    println!("cargo:rustc-env=PR_NUM_BRUSHES={num_brushes}");

    println!("cargo:rerun-if-changed=build.rs");
}
