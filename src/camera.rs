use std::cell::Cell;

use cgmath::{prelude::*, Matrix4, Point3, Rad, Vector3};

#[derive(Debug)]
pub struct Camera {
    position: Point3<f32>,
    direction: Vector3<f32>,
    fov: Rad<f32>,
    aspect_ratio: f32,
    near: f32,
    far: f32,
    view: Cell<Option<[[f32; 4]; 4]>>,
    perspective: Cell<Option<[[f32; 4]; 4]>>,
}

impl Camera {
    pub fn new(
        position: Point3<f32>,
        direction: Vector3<f32>,
        fov: impl Into<Rad<f32>>,
        aspect_ratio: f32,
        near: f32,
        far: f32,
    ) -> Self {
        Self {
            position,
            direction,
            fov: fov.into(),
            aspect_ratio,
            near,
            far,
            view: Cell::new(None),
            perspective: Cell::new(None),
        }
    }

    pub fn zoom(&mut self, amount: f32) {
        self.position += self.direction.normalize() * amount;
        self.reset_view_perspective();
    }

    fn reset_view_perspective(&self) {
        self.view.set(None);
        self.perspective.set(None);
    }

    pub fn view(&self) -> [[f32; 4]; 4] {
        if let Some(view) = self.view.get() {
            view
        } else {
            self.view.set(Some(
                Matrix4::look_at_rh(
                    self.position,
                    self.position + self.direction,
                    Vector3::unit_y(),
                )
                .into(),
            ));
            self.view.get().unwrap()
        }
    }

    pub fn perspective(&self) -> [[f32; 4]; 4] {
        if let Some(perspective) = self.perspective.get() {
            perspective
        } else {
            self.perspective.set(Some(
                cgmath::perspective(self.fov, self.aspect_ratio, self.near, self.far).into(),
            ));
            self.perspective.get().unwrap()
        }
    }
}
