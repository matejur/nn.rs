use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

use egui::{epaint::CircleShape, Color32, Pos2, Shape, Stroke, Vec2};
use nn::{
    cost::CostFunction,
    linear::{Activation, LayerData, Linear},
    network::NeuralNetwork,
};

fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "Neural network",
        native_options,
        Box::new(|_cc| Box::new(App::new())),
    )
}

pub struct App {
    layer_info: Vec<(usize, Activation)>,
    cost_function: CostFunction,
    neural_network: Option<NeuralNetwork>,
    auto_update: bool,
}

impl Default for App {
    fn default() -> Self {
        Self {
            layer_info: vec![(1, Activation::None), (1, Activation::None)],
            neural_network: None,
            cost_function: CostFunction::CrossEntropy,
            auto_update: true,
        }
    }
}

impl App {
    /// Called once before the first frame.
    pub fn new() -> Self {
        Default::default()
    }

    fn network_config(&mut self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.horizontal(|ui| {
                let layer = self.layer_info.first_mut().unwrap();
                ui.label("Inputs: ");
                if ui.button("-").clicked() {
                    layer.0 -= 1;
                }

                if layer.0 < 1 {
                    layer.0 = 1;
                }

                ui.add(egui::DragValue::new(&mut layer.0).speed(0.0));

                if ui.button("+").clicked() {
                    layer.0 += 1;
                }
            })
        });

        let mut to_remove = None;
        for index in 1..self.layer_info.len() - 1 {
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    let layer = &mut self.layer_info[index];

                    ui.label("Neurons: ");
                    if ui.button("-").clicked() {
                        layer.0 -= 1;
                    }

                    if layer.0 < 1 {
                        layer.0 = 1;
                    }

                    ui.add(egui::DragValue::new(&mut layer.0).speed(0.0));

                    if ui.button("+").clicked() {
                        layer.0 += 1;
                    }

                    activation_function_selector(ui, &mut layer.1, index);

                    if ui.button("remove").clicked() {
                        to_remove = Some(index);
                    }
                });
            });
        }

        if let Some(index) = to_remove {
            self.layer_info.remove(index);
        }

        ui.group(|ui| {
            ui.horizontal(|ui| {
                let index = self.layer_info.len();
                let layer = self.layer_info.last_mut().unwrap();
                ui.label("Outputs: ");
                if ui.button("-").clicked() {
                    layer.0 -= 1;
                }

                if layer.0 < 1 {
                    layer.0 = 1;
                }

                ui.add(egui::DragValue::new(&mut layer.0).speed(0.0));

                if ui.button("+").clicked() {
                    layer.0 += 1;
                }

                activation_function_selector(ui, &mut layer.1, index);
            });
        });

        ui.group(|ui| {
            ui.label("Cost function: ");
            egui::ComboBox::from_id_source(format!("cost_function_selector"))
                .selected_text(format!("{:?}", self.cost_function))
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.cost_function,
                        CostFunction::CrossEntropy,
                        "CrossEntropy",
                    );
                    ui.selectable_value(
                        &mut self.cost_function,
                        CostFunction::MeanSquaredError,
                        "MeanSquaredError",
                    );
                });
        });

        ui.horizontal(|ui| {
            if ui.button("Add layer").clicked() {
                self.layer_info
                    .insert(self.layer_info.len() - 1, (1, Activation::None));
            }

            ui.checkbox(&mut self.auto_update, "Auto create");

            if ui.button("Create network").clicked() {
                self.neural_network = Some(create_network(&self.layer_info, self.cost_function))
            }
        });
    }
}

fn create_network(
    layer_info: &Vec<(usize, Activation)>,
    cost_function: CostFunction,
) -> NeuralNetwork {
    let layers = to_layer_data(layer_info)
        .into_iter()
        .map(Linear::from_layer_data)
        .collect();
    NeuralNetwork::create(layers, cost_function)
}

fn to_layer_data(input: &Vec<(usize, Activation)>) -> Vec<LayerData> {
    let mut out = Vec::new();
    let mut data = LayerData::default();
    data.input_channels = input[0].0;

    for i in 1..input.len() {
        data.output_channels = input[i].0;
        data.activation_function = input[i].1;
        out.push(data.clone());
        data.input_channels = input[i].0;
    }

    out
}

fn activation_function_selector(ui: &mut egui::Ui, act: &mut Activation, index: usize) {
    ui.label("Activation: ");
    egui::ComboBox::from_id_source(format!("activation_function_layer_{index}"))
        .selected_text(format!("{:?}", act))
        .show_ui(ui, |ui| {
            ui.selectable_value(act, nn::linear::Activation::None, "None");
            ui.selectable_value(act, nn::linear::Activation::ReLU, "ReLU");
            ui.selectable_value(act, nn::linear::Activation::Sigmoid, "Sigmoid");
            ui.selectable_value(
                act,
                nn::linear::Activation::SoftmaxCrossEntropy,
                "SoftmaxCrossEntropy",
            );
        });
}

fn interpolate_color(color1: Color32, color2: Color32, t: f32) -> Color32 {
    let r = (1.0 - t) * color1.r() as f32 + t * color2.r() as f32;
    let g = (1.0 - t) * color1.g() as f32 + t * color2.g() as f32;
    let b = (1.0 - t) * color1.b() as f32 + t * color2.b() as f32;

    Color32::from_rgb(r as u8, g as u8, b as u8)
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let mut starting_hash = 0;
        if self.auto_update {
            let mut hasher = DefaultHasher::new();
            self.layer_info.hash(&mut hasher);
            starting_hash = hasher.finish();
        }

        egui::Window::new("Network config").show(ctx, |ui| {
            self.network_config(ui);
        });

        // egui::SidePanel::left("side_panel").show(ctx, |ui| {
        //     self.network_config(ui);
        // });

        if self.auto_update {
            let mut hasher = DefaultHasher::new();
            self.layer_info.hash(&mut hasher);
            let ending_hash = hasher.finish();

            if starting_hash != ending_hash {
                self.neural_network = Some(create_network(&self.layer_info, self.cost_function));
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            if let Some(network) = &self.neural_network {
                let painter = ui.painter();

                let Vec2 {
                    x: width,
                    y: height,
                } = ui.available_size();

                let margin_lr = 100.0;
                let margin_ud = 50.0;

                let layers = network.get_layers();
                let columns = layers.len() + 1;
                let column_spacing = (width - 2.0 * margin_lr) / (columns - 1) as f32;

                let number_of_inputs = layers[0].get_input_channels();
                let available_height = height - 2.0 * margin_ud;
                let row_spacing = available_height / (number_of_inputs as f32);

                // Draw all weights
                for index in 0..layers.len() {
                    let layer = &layers[index];
                    let number_of_inputs = layer.get_input_channels();
                    let number_of_neurons = layer.get_output_channels();
                    let current_row_spacing = available_height / (number_of_neurons as f32);
                    let previous_row_spacing = available_height / (number_of_inputs as f32);

                    for neuron_index in 0..number_of_neurons {
                        for input_index in 0..number_of_inputs {
                            let weight_index = neuron_index * number_of_inputs + input_index;
                            let weight = layer.weights.elems[weight_index];
                            let weight_sigmoid = 1.0 / (1.0 + (-weight).exp());

                            let color =
                                interpolate_color(Color32::RED, Color32::BLUE, weight_sigmoid);

                            painter.add(Shape::LineSegment {
                                points: [
                                    Pos2 {
                                        x: margin_lr + (index + 1) as f32 * column_spacing,
                                        y: margin_ud
                                            + (neuron_index as f32 + 0.5) * current_row_spacing,
                                    },
                                    Pos2 {
                                        x: margin_lr + index as f32 * column_spacing,
                                        y: margin_ud
                                            + (input_index as f32 + 0.5) * previous_row_spacing,
                                    },
                                ],
                                stroke: Stroke {
                                    width: 2.0,
                                    color: color,
                                },
                            });
                        }
                    }
                }

                // Draw neurons on top
                for inp_index in 0..number_of_inputs {
                    painter.add(Shape::Circle(CircleShape {
                        radius: 15.0,
                        center: Pos2 {
                            x: margin_lr,
                            y: margin_ud + (inp_index as f32 + 0.5) * row_spacing,
                        },
                        fill: Color32::LIGHT_GRAY,
                        stroke: Stroke::default(),
                    }));
                }

                for index in 0..layers.len() {
                    let layer = &layers[index];
                    let number_of_neurons = layer.get_output_channels();
                    let current_row_spacing = available_height / (number_of_neurons as f32);

                    for neuron_index in 0..number_of_neurons {
                        let bias = layer.biases.elems[neuron_index];
                        let bias_sigmoid = 1.0 / (1.0 + (-bias).exp());

                        let color = interpolate_color(Color32::RED, Color32::BLUE, bias_sigmoid);

                        painter.add(Shape::Circle(CircleShape {
                            radius: 15.0,
                            center: Pos2 {
                                x: margin_lr + (index + 1) as f32 * column_spacing,
                                y: margin_ud + (neuron_index as f32 + 0.5) * current_row_spacing,
                            },
                            fill: color,
                            stroke: Stroke::default(),
                        }));
                    }
                }
            }
        });
    }
}
