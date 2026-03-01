use axum::{Router, routing};
use chrono::Utc;
use serde_json::json;
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;

// Custom modules
pub mod api;
pub mod models;
use crate::handlers::api::ApiResponse;
use crate::handlers::models::post::ImageSearchRequest;

// Variables
pub const TAG_GENERAL: &str = "General";
pub const TAG_MODELS: &str = "Models";

#[derive(OpenApi)]
#[openapi(
    paths(
        health,
        models::post::search_image
    ),
    components(schemas(
        ImageSearchRequest
    )),
    tags(
        (name = TAG_MODELS, description = "Model Endpoints"),
        (name = TAG_GENERAL, description = "General Endpoints"),
    ),
    info(
        title = "Image Search API",
        version = "1.0.0",
        description = "API for image search",
    ),
)]
struct APIDoc;

pub fn routes() -> Router {
    let openapi = APIDoc::openapi();

    Router::new()
        .merge(
            SwaggerUi::new("/docs")
                .url("/openapi.json", openapi)
        )

        .merge(models::routes())
        .route("/health", routing::get(health))
        .fallback(default)
}

/// Returns health status for whole API
#[utoipa::path(
    get,
    path = "/health",
    tag = TAG_GENERAL,
    operation_id="general_health",
    responses(
        (status = 200, description = "API is healthy")
    ),
)]
pub async fn health() -> ApiResponse<serde_json::Value> {
    ApiResponse::success_with_message(
        "OK", 
        json!({
            "timestamp": Utc::now().to_rfc3339()
        })
    )
}

pub async fn default() -> ApiResponse<()> {
    ApiResponse::not_found("Route not found!")
}