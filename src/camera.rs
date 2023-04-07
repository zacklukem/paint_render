use std::cell::Cell;

use cgmath::{prelude::*, Deg, Matrix4, Point3, Rad, Vector3};

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

    pub fn rotate_up(&mut self, angle: impl Into<Rad<f32>>) {
        let angle = angle.into();
        let theta: Deg<_> = self.position.to_vec().angle(Vector3::unit_y()).into();
        let angle_d: Deg<_> = angle.into();
        if (theta.0 + angle_d.0 < 5.0 && angle.0 < 0.0)
            || (theta.0 + angle_d.0 > 175.0 && angle.0 > 0.0)
        {
            return;
        }
        let distance = self.position.distance(Point3::origin());

        self.position = Point3::from_vec(
            distance
                * (Matrix4::from_axis_angle(self.right(), angle)
                    * self.position.to_vec().normalize().extend(1.0))
                .truncate(),
        );
        self.direction = -self.position.to_vec().normalize();
        self.reset_view_perspective();
    }

    pub fn right(&self) -> Vector3<f32> {
        self.direction.cross(Vector3::unit_y()).normalize()
    }

    pub fn position(&self) -> Point3<f32> {
        self.position
    }

    pub fn zoom(&mut self, amount: f32) {
        self.position += self.direction.normalize() * amount;
        self.reset_view_perspective();
    }

    pub fn set_aspect(&mut self, aspect_ratio: f32) {
        self.aspect_ratio = aspect_ratio;
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
