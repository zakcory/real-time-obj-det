use axum::{Router, routing};
use crate::handlers::models::post::search_image;

// Custom modules
pub mod post;

pub fn routes() -> Router {
    Router::new()
        .route("/search", routing::post(search_image))
}
