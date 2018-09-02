use display_graph::DisplayGraph;
use CliResult;
use std::fs::File;
use OutputParameters;

/// Starts a web server for TFVisualizer and opens its webroot in a browser.
pub fn open_web(output: &DisplayGraph, _params: &OutputParameters) -> CliResult<()> {
    use rouille::Response;

    let data = ::serde_json::to_vec(output)?;

    println!("TFVisualizer is now running on http://127.0.0.1:8000/.");
    let _ = ::open::that("http://127.0.0.1:8000/");

    ::rouille::start_server("0.0.0.0:8000", move |request| {
        if request.remove_prefix("/dist").is_some() || request.remove_prefix("/public").is_some() {
            return ::rouille::match_assets(&request, "../visualizer");
        }

        return router!(request,
            (GET) (/) => {
                let index = File::open("../visualizer/index.html").unwrap();
                Response::from_file("text/html", index)
            },

            (GET) (/current) => {
                Response::from_data("application/json", data.clone())
            },

            _ => Response::empty_404(),
        );
    });
}
