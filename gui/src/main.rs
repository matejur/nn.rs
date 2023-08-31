use nn::linear::LayerData;

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "Neural network",
        native_options,
        Box::new(|_cc| Box::new(App::new())),
    )
}

pub struct App {
    layer_data: Vec<LayerData>,
}

impl Default for App {
    fn default() -> Self {
        Self {
            layer_data: vec![LayerData::default()],
        }
    }
}

impl App {
    /// Called once before the first frame.
    pub fn new() -> Self {
        Default::default()
    }
}

fn activation_function_selector(ui: &mut egui::Ui, layer: &mut LayerData, index: usize) {
    ui.label("Activation: ");
    egui::ComboBox::from_id_source(format!("activation_function_layer_{index}"))
        .selected_text(format!("{:?}", layer.activation_function))
        .show_ui(ui, |ui| {
            ui.selectable_value(
                &mut layer.activation_function,
                nn::linear::Activation::None,
                "None",
            );
            ui.selectable_value(
                &mut layer.activation_function,
                nn::linear::Activation::ReLU,
                "ReLU",
            );
            ui.selectable_value(
                &mut layer.activation_function,
                nn::linear::Activation::Sigmoid,
                "Sigmoid",
            );
            ui.selectable_value(
                &mut layer.activation_function,
                nn::linear::Activation::SoftmaxCrossEntropy,
                "SoftmaxCrossEntropy",
            );
        });
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::SidePanel::left("side_panel").show(ctx, |ui| {
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    let layer = self.layer_data.first_mut().unwrap();
                    ui.label("Inputs: ");
                    if ui.button("-").clicked() {
                        layer.input_channels -= 1;
                    }

                    if layer.input_channels < 1 {
                        layer.input_channels = 1;
                    }

                    ui.add(egui::DragValue::new(&mut layer.input_channels).speed(0.0));

                    if ui.button("+").clicked() {
                        layer.input_channels += 1;
                    }
                })
            });

            for index in 0..self.layer_data.len() - 1 {
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        let (before, after) = self.layer_data.split_at_mut(index + 1);
                        let input_layer = before.last_mut().unwrap();
                        let output_layer = after.first_mut().unwrap();
                        let mut current_neurons = output_layer.input_channels;

                        ui.label("Neurons: ");
                        if ui.button("-").clicked() {
                            current_neurons -= 1;
                        }

                        if current_neurons < 1 {
                            current_neurons = 1;
                        }

                        ui.add(egui::DragValue::new(&mut current_neurons).speed(0.0));

                        if ui.button("+").clicked() {
                            current_neurons += 1;
                        }

                        input_layer.output_channels = current_neurons;
                        output_layer.input_channels = current_neurons;

                        activation_function_selector(ui, input_layer, index);
                    });
                });
            }

            ui.group(|ui| {
                ui.horizontal(|ui| {
                    let index = self.layer_data.len();
                    let layer = self.layer_data.last_mut().unwrap();
                    ui.label("Outputs: ");
                    if ui.button("-").clicked() {
                        layer.output_channels -= 1;
                    }

                    if layer.output_channels < 1 {
                        layer.output_channels = 1;
                    }

                    ui.add(egui::DragValue::new(&mut layer.output_channels).speed(0.0));

                    if ui.button("+").clicked() {
                        layer.output_channels += 1;
                    }

                    activation_function_selector(ui, layer, index);
                });
            });

            if ui.button("Add layer").clicked() {
                if self.layer_data.len() == 1 {
                    let last_layer = self.layer_data.last_mut().unwrap();

                    let new_layer_data = LayerData {
                        input_channels: 1,
                        output_channels: last_layer.output_channels,
                        activation_function: last_layer.activation_function,
                    };

                    last_layer.output_channels = 1;
                    last_layer.activation_function = nn::linear::Activation::None;

                    self.layer_data.push(new_layer_data);
                } else {
                    let last_index = self.layer_data.len() - 1;
                    let last_layer = &self.layer_data[last_index];
                    let second_last_layer = &self.layer_data[last_index - 1];

                    let new_layer_data = LayerData {
                        input_channels: second_last_layer.output_channels,
                        output_channels: last_layer.input_channels,
                        activation_function: Default::default(),
                    };

                    self.layer_data.insert(last_index, new_layer_data);
                }
            }

            if ui.button("Debug").clicked() {
                dbg!(&self.layer_data);
            }
        });

        // egui::CentralPanel::default().show(ctx, |ui| {
        //     // The central panel the region left after adding TopPanel's and SidePanel's
        // });
    }
}
