use three_d::*;
use three_d_asset::Scene;

mod consts {
    // Submodule just so I can limit the scope of rustfmt::skip.
    use std::f32::consts::FRAC_1_SQRT_2;
    use three_d::Mat4;

    #[rustfmt::skip]
    pub const OBLIQUE_SHEAR: Mat4 = Mat4::new(
                    1.0,              0.0,  0.0,  0.0,
                    0.0,              1.0,  0.0,  0.0,
       -FRAC_1_SQRT_2,    FRAC_1_SQRT_2,    1.0,  0.0,
                    0.0,              0.0,  0.0,  1.0,
    );
}

pub fn main() {
    // Get model path from command line.
    let model_path = std::env::args().nth(1).expect("Usage: view <model_path>");

    // Create a window (a canvas on web)
    let window = Window::new(WindowSettings {
        title: "Model viewer".to_string(),

        max_size: Some((1280, 720)),
        ..Default::default()
    })
    .unwrap();

    // Get the graphics context from the window
    let context = window.gl();

    // Create a camera
    let mut camera = Camera::new_orthographic(
        window.viewport(),
        vec3(0.0, 0.0, 100.0),
        vec3(0.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0),
        1.0,
        0.1,
        120.0,
    );

    let mut loaded = three_d_asset::io::load(&[model_path]).expect("Couldn't load OBJ");
    let mut scene: Scene = loaded.deserialize("obj").expect("Couldn't deserialize OBJ");

    scene.children[0].transformation = consts::OBLIQUE_SHEAR;

    let cpu_model: CpuModel = scene.into();

    let model = Model::<PhysicalMaterial>::new(&context, &cpu_model).unwrap();

    let light = DirectionalLight::new(&context, 10.0, Srgba::WHITE, vec3(-1.0, -1.0, -3.0));

    // Start the main render loop
    window.render_loop(
        move |frame_input| // Begin a new frame with an updated frame input
    {
        // Ensure the viewport matches the current window viewport which changes if the window is resized
        camera.set_viewport(frame_input.viewport);

        // Get the screen render target to be able to render something on the screen
        frame_input.screen()
            .clear(ClearState::color_and_depth(0.0, 1.0, 1.0, 1.0, 1.0))
            .render(&camera, &model, &[&light]);

        FrameOutput::default()
    },
    );
}
