use egui::{Color32, Vec2, Shape, epaint::CircleShape, Pos2, Stroke};
use nn::network::NeuralNetwork;

pub struct Visualizer {
    low_color: Color32,
    high_color: Color32,
}

fn interpolate_color(color1: Color32, color2: Color32, t: f32) -> Color32 {
    let r = (1.0 - t) * color1.r() as f32 + t * color2.r() as f32;
    let g = (1.0 - t) * color1.g() as f32 + t * color2.g() as f32;
    let b = (1.0 - t) * color1.b() as f32 + t * color2.b() as f32;

    Color32::from_rgb(r as u8, g as u8, b as u8)
}

impl Visualizer {
    pub fn visual_config(&mut self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.label("Low color: ");
                ui.color_edit_button_srgba(&mut self.low_color);
                ui.label("High color: ");
                ui.color_edit_button_srgba(&mut self.high_color);
            });
        });
    }

    pub fn visualize_network(&self, ui: &mut egui::Ui, network: &NeuralNetwork) {
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
        for (index, layer) in layers.iter().enumerate() {
            let number_of_inputs = layer.get_input_channels();
            let number_of_neurons = layer.get_output_channels();
            let current_row_spacing = available_height / (number_of_neurons as f32);
            let previous_row_spacing = available_height / (number_of_inputs as f32);

            for neuron_index in 0..number_of_neurons {
                for input_index in 0..number_of_inputs {
                    let weight_index = neuron_index * number_of_inputs + input_index;
                    let weight = layer.weights.elems[weight_index];
                    let weight_sigmoid = 1.0 / (1.0 + (-weight).exp());

                    let color = interpolate_color(self.low_color, self.high_color, weight_sigmoid);

                    painter.add(Shape::LineSegment {
                        points: [
                            Pos2 {
                                x: margin_lr + (index + 1) as f32 * column_spacing,
                                y: margin_ud + (neuron_index as f32 + 0.5) * current_row_spacing,
                            },
                            Pos2 {
                                x: margin_lr + index as f32 * column_spacing,
                                y: margin_ud + (input_index as f32 + 0.5) * previous_row_spacing,
                            },
                        ],
                        stroke: Stroke { width: 2.0, color },
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

        for (index, layer) in layers.iter().enumerate() {
            let number_of_neurons = layer.get_output_channels();
            let current_row_spacing = available_height / (number_of_neurons as f32);

            for neuron_index in 0..number_of_neurons {
                let bias = layer.biases.elems[neuron_index];
                let bias_sigmoid = 1.0 / (1.0 + (-bias).exp());

                let color = interpolate_color(self.low_color, self.high_color, bias_sigmoid);

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
}

impl Default for Visualizer {
    fn default() -> Self {
        Self { low_color: Color32::RED, high_color: Color32::BLUE }
    }
}
