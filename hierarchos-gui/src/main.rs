// Hierarchos GUI — Premium egui Interface
// A high-performance native GUI for the Hierarchos architecture

mod app;
mod bridge;
mod embedded;
mod panels;
mod theme;
mod widgets;

use app::HierarchosApp;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Hierarchos — Linear-Complexity Hierarchical Agent")
            .with_inner_size([1440.0, 900.0])
            .with_min_inner_size([900.0, 600.0])
            .with_decorations(true),
        ..Default::default()
    };

    eframe::run_native(
        "Hierarchos",
        options,
        Box::new(|cc| Ok(Box::new(HierarchosApp::new(cc)))),
    )
}
