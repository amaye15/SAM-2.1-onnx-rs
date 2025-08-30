use serde::{Deserialize, Serialize};
use uuid::Uuid;
use utoipa::ToSchema;

/// Model size selection for SAM2.
/// Larger models are slower but more accurate.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema, PartialEq, Eq, Hash)]
pub enum Sam2ModelSize {
    /// Fastest and smallest model.
    Tiny,
    /// Small model: balance of speed and accuracy.
    Small,
    /// Base+ model: higher accuracy.
    BasePlus,
    /// Largest and most accurate model.
    Large,
}

impl Sam2ModelSize {
    pub fn to_filename(&self) -> &'static str {
        match self {
            Sam2ModelSize::Tiny => "sam2_tiny.onnx",
            Sam2ModelSize::Small => "sam2_small.onnx",
            Sam2ModelSize::BasePlus => "sam2_base_plus.onnx",
            Sam2ModelSize::Large => "sam2_large.onnx",
        }
    }
}

/// Prompt point in model input space (1024x1024).
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct Point {
    /// X coordinate in model space [0, 1024).
    #[schema(minimum = 0, maximum = 1024, example = 512)]
    pub x: f32,
    /// Y coordinate in model space [0, 1024).
    #[schema(minimum = 0, maximum = 1024, example = 512)]
    pub y: f32,
    /// Point label: 1 = positive (foreground), 0 = negative (background).
    #[schema(value_type = i32, example = 1, minimum = 0, maximum = 1)]
    pub label: i32,
}

/// Request body for segmentation.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct SegmentRequest {
    /// Optional client-supplied correlation ID. If not provided the server will generate one.
    #[schema(example = "550e8400-e29b-41d4-a716-446655440000")]
    pub request_id: Option<Uuid>,
    /// Model size to use for inference.
    #[schema(example = "Large")]
    pub model: Sam2ModelSize,
    /// Base64-encoded PNG or JPEG image data at original resolution.
    #[schema(example = "<base64 PNG/JPEG data>")]
    pub image_b64: String,
    /// Positive/negative prompt points in 1024x1024 model space.
    #[schema(example = json!([{"x":512.0,"y":512.0,"label":1}]))]
    pub points: Vec<Point>,
    /// Optional probability threshold (0..1). Default is 0.5.
    #[schema(example = 0.5, minimum = 0.0, maximum = 1.0)]
    pub threshold: Option<f32>,
}

/// Segmentation result payload.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct SegmentResponse {
    /// Echoed or generated request ID for tracing.
    #[schema(example = "550e8400-e29b-41d4-a716-446655440000")]
    pub request_id: Uuid,
    /// Model actually used.
    #[schema(example = "Large")]
    pub model: Sam2ModelSize,
    /// IoU predictions for the 3 mask candidates.
    #[schema(example = json!([0.75, 0.81, 0.64]))]
    pub iou: [f32; 3],
    /// Index of the best mask candidate by IoU (0..2).
    #[schema(example = 1)]
    pub best_idx: usize,
    /// Wall-clock inference time in milliseconds.
    #[schema(example = 123)]
    pub inference_ms: u128,
    /// 1024x1024 RGBA overlay PNG (blue fill/outline, transparent elsewhere).
    #[schema(example = "<base64 PNG>")]
    pub mask_png_b64: String,
    /// Original-resolution RGBA PNG where pixels outside the mask are transparent.
    #[schema(example = "<base64 PNG>")]
    pub masked_region_png_b64: Option<String>,
}

/// Available models response.
#[derive(Debug, Clone, Serialize, Deserialize, ToSchema)]
pub struct ModelsResponse {
    /// List of available model sizes.
    #[schema(example = json!(["Tiny","Small","BasePlus","Large"]))]
    pub models: Vec<Sam2ModelSize>,
}

