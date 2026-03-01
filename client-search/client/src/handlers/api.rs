use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;
use serde_json::json;

#[derive(Debug)]
pub struct ApiResponse<T: Serialize> {
    pub status_code: StatusCode,
    pub message: String,
    pub data: Option<T>,
}

impl<T: Serialize> ApiResponse<T> {
    // Success responses (2xx)
    pub fn success(data: T) -> Self {
        Self {
            status_code: StatusCode::OK,
            message: "Success".to_string(),
            data: Some(data),
        }
    }

    pub fn success_with_message(message: impl Into<String>, data: T) -> Self {
        Self {
            status_code: StatusCode::OK,
            message: message.into(),
            data: Some(data),
        }
    }

    pub fn created(data: T) -> Self {
        Self {
            status_code: StatusCode::CREATED,
            message: "Created successfully".to_string(),
            data: Some(data),
        }
    }

    pub fn accepted(data: T) -> Self {
        Self {
            status_code: StatusCode::ACCEPTED,
            message: "Accepted".to_string(),
            data: Some(data),
        }
    }

    pub fn no_content() -> ApiResponse<()> {
        ApiResponse {
            status_code: StatusCode::NO_CONTENT,
            message: "No content".to_string(),
            data: None,
        }
    }

    // Error responses (4xx)
    pub fn error(status_code: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status_code,
            message: message.into(),
            data: None,
        }
    }

    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::error(StatusCode::BAD_REQUEST, message)
    }

    pub fn unauthorized(message: impl Into<String>) -> Self {
        Self::error(StatusCode::UNAUTHORIZED, message)
    }

    pub fn forbidden(message: impl Into<String>) -> Self {
        Self::error(StatusCode::FORBIDDEN, message)
    }

    pub fn not_found(message: impl Into<String>) -> Self {
        Self::error(StatusCode::NOT_FOUND, message)
    }

    pub fn conflict(message: impl Into<String>) -> Self {
        Self::error(StatusCode::CONFLICT, message)
    }

    pub fn unprocessable_entity(message: impl Into<String>) -> Self {
        Self::error(StatusCode::UNPROCESSABLE_ENTITY, message)
    }

    pub fn too_many_requests(message: impl Into<String>) -> Self {
        Self::error(StatusCode::TOO_MANY_REQUESTS, message)
    }

    // Server error responses (5xx)
    pub fn internal_error(message: impl Into<String>) -> Self {
        Self::error(StatusCode::INTERNAL_SERVER_ERROR, message)
    }

    pub fn service_unavailable(message: impl Into<String>) -> Self {
        Self::error(StatusCode::SERVICE_UNAVAILABLE, message)
    }

    // Custom
    pub fn custom(status_code: StatusCode, message: impl Into<String>, data: T) -> Self {
        Self {
            status_code,
            message: message.into(),
            data: Some(data),
        }
    }
}

impl<T: Serialize> IntoResponse for ApiResponse<T> {
    fn into_response(self) -> Response {
        let body = json!({
            "message": self.message,
            "data": self.data,
        });

        (self.status_code, Json(body)).into_response()
    }
}

// New Error type wrapper
pub struct AppError(pub anyhow::Error);

// Implement IntoResponse for AppError
impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        ApiResponse::<()>::internal_error(self.0.to_string()).into_response()
    }
}

// Enable `?` operator for any error that can convert to anyhow::Error
impl<E> From<E> for AppError
where
    E: Into<anyhow::Error>,
{
    fn from(err: E) -> Self {
        Self(err.into())
    }
}

// Type alias for cleaner handler signatures
pub type ApiResult<T> = Result<ApiResponse<T>, AppError>;