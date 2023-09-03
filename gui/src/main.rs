mod visualization;
use visualization::Visualizer;

use std::{
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
};

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

    visualizer: Visualizer,
}

impl Default for App {
    fn default() -> Self {
        Self {
            layer_info: vec![(1, Activation::None), (1, Activation::None)],
            neural_network: None,
            cost_function: CostFunction::CrossEntropy,
            auto_update: true,
            visualizer: Visualizer::default(),
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
            egui::ComboBox::from_id_source("cost_function_selector")
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
    layer_info: &[(usize, Activation)],
    cost_function: CostFunction,
) -> NeuralNetwork {
    let layers = to_layer_data(layer_info)
        .into_iter()
        .map(Linear::from_layer_data)
        .collect();
    NeuralNetwork::create(layers, cost_function)
}

fn to_layer_data(input: &[(usize, Activation)]) -> Vec<LayerData> {
    let mut out = Vec::new();
    let mut data = LayerData {
        input_channels: input[0].0,
        ..Default::default()
    };

    for (neurons, act) in input.iter().skip(1) {
        data.output_channels = *neurons;
        data.activation_function = *act;
        out.push(data.clone());
        data.input_channels = *neurons;
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

        egui::Window::new("Visualization config").show(ctx, |ui| {
            self.visualizer.visual_config(ui);
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
                self.visualizer.visualize_network(ui, network);
            }
        });
    }
}
