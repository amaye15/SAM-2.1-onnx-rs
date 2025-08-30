use std::{collections::HashMap, sync::Arc, time::Instant};

use anyhow::Result;
use axum::{
    extract::{DefaultBodyLimit, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use axum_prometheus::PrometheusMetricLayer;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;
use image::{imageops::FilterType, DynamicImage, ImageBuffer, Rgba};
use ndarray::{s, Array, Array4, Ix4};
use ort::{session::{builder::GraphOptimizationLevel, Session}, value::Tensor};
use tokio::sync::RwLock;
use crate::shared::{ModelsResponse, Point, Sam2ModelSize, SegmentRequest, SegmentResponse};
use tower_http::{cors::{Any, CorsLayer}, services::ServeDir, trace::TraceLayer};
use tracing::{error, info, Level};
use utoipa::OpenApi;
use utoipa_swagger_ui::SwaggerUi;
use uuid::Uuid;

#[derive(Clone)]
struct AppState {
    models: Arc<RwLock<HashMap<Sam2ModelSize, Arc<RwLock<Session>>>>>,
}

#[derive(thiserror::Error, Debug)]
enum AppError {
    #[error("Bad request: {0}")]
    BadRequest(String),
    #[error("Internal error: {0}")]
    Internal(String),
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let (code, msg) = match self {
            AppError::BadRequest(m) => (StatusCode::BAD_REQUEST, m),
            AppError::Internal(m) => (StatusCode::INTERNAL_SERVER_ERROR, m),
        };
        (code, msg).into_response()
    }
}

#[utoipa::path(
    get,
    path = "/api/models",
    tag = "Models",
    summary = "List available model sizes",
    description = "Returns the list of supported SAM2 model variants you can use for segmentation.",
    responses(
        (status = 200, description = "Available model sizes", body = ModelsResponse)
    )
)]
async fn get_models() -> Json<ModelsResponse> {
    Json(ModelsResponse { models: vec![
        Sam2ModelSize::Tiny,
        Sam2ModelSize::Small,
        Sam2ModelSize::BasePlus,
        Sam2ModelSize::Large,
    ]})
}

#[derive(OpenApi)]
#[openapi(
    paths(get_models, segment),
    components(schemas(SegmentRequest, SegmentResponse, ModelsResponse, Sam2ModelSize, Point))
)]
struct ApiDoc;

#[utoipa::path(
    post,
    path = "/api/segment",
    tag = "Segmentation",
    summary = "Run SAM2 segmentation for an image with click prompts",
    description = r#"Request body fields:
- image_b64: Base64 PNG/JPEG of the input image (original resolution).
- model: Model size to use (Tiny | Small | BasePlus | Large).
- points: Array of click prompts in 1024x1024 model space. label=1 (positive), 0 (negative).
- request_id: Optional client correlation ID (UUID). Server generates one if omitted.
- threshold: Optional probability threshold (0..1). Default 0.5.
"#,
    request_body = SegmentRequest,
    responses(
        (status = 200, description = "Segmentation result including IoU, overlay mask PNG, and original-resolution cutout", body = SegmentResponse)
    )
)]
async fn segment(State(state): State<AppState>, Json(req): Json<SegmentRequest>) -> Result<Json<SegmentResponse>, AppError> {
    info!("segment: model={:?}, points={}", req.model, req.points.len());
    // 1) Decode image
    let img_bytes = B64.decode(req.image_b64.as_bytes()).map_err(|e| AppError::BadRequest(format!("invalid base64: {e}")))?;
    info!("segment: image bytes decoded: {} bytes", img_bytes.len());
    let dyn_img = image::load_from_memory(&img_bytes).map_err(|e| AppError::BadRequest(format!("invalid image: {e}")))?;

    // 2) Preprocess to 1024x1024 CHW f32
    let pre = preprocess_image(&dyn_img).map_err(|e| AppError::BadRequest(format!("preprocess error: {e}")))?;
    info!("segment: preprocessed image to [1,3,1024,1024]");

    // 3) Build points arrays
    if req.points.is_empty() { return Err(AppError::BadRequest("at least one point required".into())); }
    let n = req.points.len();
    let mut coords = Vec::with_capacity(n * 2);
    let mut labels = Vec::with_capacity(n);
    for p in &req.points {
        coords.push(p.x);
        coords.push(p.y);
        labels.push(p.label as f32);
    }
    let point_coords = ndarray::Array::from_shape_vec((1, n, 2), coords).map_err(|e| AppError::BadRequest(format!("bad point shape: {e}")))?.into_dyn();
    let point_labels = ndarray::Array::from_shape_vec((1, n), labels).map_err(|e| AppError::BadRequest(format!("bad label shape: {e}")))?.into_dyn();

    // 4) Get or load session
    let model_sel = req.model.clone();
    let session = get_or_load_session(&state, model_sel.clone()).await.map_err(|e| AppError::Internal(e.to_string()))?;

    // 5) Run inference
    let t0 = Instant::now();
    // Build inputs outside of the guard to avoid borrowing issues
    let t_image = Tensor::from_array(pre).map_err(|e| AppError::Internal(e.to_string()))?;
    let t_coords = Tensor::from_array(point_coords).map_err(|e| AppError::Internal(e.to_string()))?;
    let t_labels = Tensor::from_array(point_labels).map_err(|e| AppError::Internal(e.to_string()))?;

    // Run inference and compute outputs within the lock scope so borrows don't escape
    let (mask_png_b64, masked_region_png_b64, iou_arr, best_idx) = {
        let prob_thresh = req.threshold.unwrap_or(0.5);
        let mut guard = session.write().await;
        let outs = guard.run(ort::inputs![
            "image" => t_image,
            "point_coords" => t_coords,
            "point_labels" => t_labels,
        ]).map_err(|e| AppError::Internal(e.to_string()))?;

        let iou = outs["iou_predictions"].try_extract_array::<f32>().map_err(|e| AppError::Internal(e.to_string()))?; // [1,3]
        let flat = iou.iter().cloned().collect::<Vec<f32>>();
        let (mut best_idx, mut best_val) = (0usize, f32::MIN);
        for (i, v) in flat.iter().enumerate() { if *v > best_val { best_val = *v; best_idx = i % 3; } }

        // Prepare mask at 1024x1024
        let masks = outs["masks"].try_extract_array::<f32>().map_err(|e| AppError::Internal(e.to_string()))?; // [1,3,1024,1024]
        let masks4 = masks.into_dimensionality::<Ix4>().map_err(|e| AppError::Internal(e.to_string()))?; // [1,3,H,W]
        let chosen = masks4.slice(s![0, best_idx, .., ..]);
        let (h, w) = (chosen.shape()[0], chosen.shape()[1]);
        // Build probability map with a light Gaussian smoothing to reduce ragged edges
        let mut prob = vec![0f32; w*h];
        for y in 0..h { for x in 0..w {
            let logit = chosen[(y, x)];
            prob[y*w + x] = 1.0 / (1.0 + (-logit).exp());
        } }
        // Separable Gaussian blur: radius=3, sigma=1.6 (even smoother edges)
        let radius = 3usize;
        let sigma = 1.6f32;
        let ksize = 2*radius + 1;
        let mut kernel = vec![0f32; ksize];
        let s2 = 2.0*sigma*sigma;
        let mut ksum = 0f32;
        for i in 0..ksize { let x = i as i32 - radius as i32; let v = (-(x as f32 * x as f32)/s2).exp(); kernel[i]=v; ksum += v; }
        for i in 0..ksize { kernel[i] /= ksum; }
        let mut tmp = vec![0f32; w*h];
        // Horizontal
        for y in 0..h { for x in 0..w {
            let mut acc = 0f32;
            for i in 0..ksize {
                let dx = i as i32 - radius as i32;
                let xx = (x as i32 + dx).clamp(0, (w-1) as i32) as usize;
                acc += prob[y*w + xx] * kernel[i];
            }
            tmp[y*w + x] = acc;
        } }
        // Vertical
        for y in 0..h { for x in 0..w {
            let mut acc = 0f32;
            for i in 0..ksize {
                let dy = i as i32 - radius as i32;
                let yy = (y as i32 + dy).clamp(0, (h-1) as i32) as usize;
                acc += tmp[yy*w + x] * kernel[i];
            }
            prob[y*w + x] = acc;
        } }
        // Threshold to binary
        let mut mask_bin = vec![0u8; w*h];
        for i in 0..(w*h) { mask_bin[i] = if prob[i] > prob_thresh { 1 } else { 0 }; }
        // Morphological cleanup: closing then opening (3x3, 1 iter each)
        let mut work = vec![0u8; w*h];
        let mut dilate = |src: &Vec<u8>, dst: &mut Vec<u8>| {
            for y in 0..h { for x in 0..w {
                let mut v = 0u8;
                'outer: for dy in -1i32..=1 { let yy = y as i32 + dy; if yy<0 || yy>=h as i32 { continue; }
                    for dx in -1i32..=1 { let xx = x as i32 + dx; if xx<0 || xx>=w as i32 { continue; }
                        if src[yy as usize*w + xx as usize] != 0 { v = 1; break 'outer; }
                    }
                }
                dst[y*w + x] = v;
            } }
        };
        let mut erode = |src: &Vec<u8>, dst: &mut Vec<u8>| {
            for y in 0..h { for x in 0..w {
                let mut v = 1u8;
                'outer: for dy in -1i32..=1 { let yy = y as i32 + dy; if yy<0 || yy>=h as i32 { v=0; break; }
                    for dx in -1i32..=1 { let xx = x as i32 + dx; if xx<0 || xx>=w as i32 { v=0; break 'outer; }
                        if src[yy as usize*w + xx as usize] == 0 { v = 0; break 'outer; }
                    }
                }
                dst[y*w + x] = v;
            } }
        };
        // Closing
        dilate(&mask_bin, &mut work);
        erode(&work, &mut mask_bin);
        // Opening
        erode(&mask_bin, &mut work);
        dilate(&work, &mut mask_bin);

        // 1) Make a 1024x1024 PNG for overlay using distance-based anti-aliased outline + smooth fill
        // Compute boundary pixels (8-neighborhood)
        let mut is_boundary = vec![0u8; w*h];
        for y in 0..h { for x in 0..w {
            let idx = y*w + x; if mask_bin[idx]==0 { continue; }
            let mut boundary=false;
            for dy in -1i32..=1 { for dx in -1i32..=1 {
                if dx==0 && dy==0 { continue; }
                let ny = y as i32 + dy; let nx = x as i32 + dx;
                if ny<0 || nx<0 || ny>=h as i32 || nx>=w as i32 { boundary=true; break; }
                let nidx = (ny as usize)*w + (nx as usize);
                if mask_bin[nidx]==0 { boundary=true; break; }
            } if boundary { break; } }
            if boundary { is_boundary[idx]=1; }
        } }
        // Approximate Euclidean distance transform from boundary using 2-pass chamfer (1, sqrt2)
        let mut dist = vec![1e9f32; w*h];
        for i in 0..(w*h) { if is_boundary[i]!=0 { dist[i]=0.0; } }
        let s2 = 1.41421356f32; // sqrt(2)
        // Forward pass
        for y in 0..h { for x in 0..w {
            let idx = y*w + x; let mut d = dist[idx];
            if x>0 { d = d.min(dist[y*w + (x-1)] + 1.0); }
            if y>0 { d = d.min(dist[(y-1)*w + x] + 1.0); }
            if x>0 && y>0 { d = d.min(dist[(y-1)*w + (x-1)] + s2); }
            if x+1<w && y>0 { d = d.min(dist[(y-1)*w + (x+1)] + s2); }
            dist[idx]=d;
        } }
        // Backward pass
        for y in (0..h).rev() { for x in (0..w).rev() {
            let idx = y*w + x; let mut d = dist[idx];
            if x+1<w { d = d.min(dist[y*w + (x+1)] + 1.0); }
            if y+1<h { d = d.min(dist[(y+1)*w + x] + 1.0); }
            if x+1<w && y+1<h { d = d.min(dist[(y+1)*w + (x+1)] + s2); }
            if x>0 && y+1<h { d = d.min(dist[(y+1)*w + (x-1)] + s2); }
            dist[idx]=d;
        } }
        // Compose overlay per pixel from prob (smooth fill) and distance (anti-aliased line)
        let mut buf = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(w as u32, h as u32);
        let fill_rgb = (0u8, 140u8, 255u8);
        let line_rgb = (0u8, 64u8, 192u8);
        let base_fill_a = 160f32/255.0; // ~0.63
        let band = 0.25f32; // softness band around threshold
        let sigma_line = 1.2f32; // controls edge thickness and smoothness
        for y in 0..h { for x in 0..w {
            let i = y*w + x;
            let p = prob[i];
            // Fill alpha via smoothstep centered at prob_thresh
            let t = ((p - (prob_thresh - band)) / (2.0*band)).clamp(0.0, 1.0);
            let s = t*t*(3.0 - 2.0*t);
            let a_fill = base_fill_a * s;
            // Line alpha via Gaussian of distance to boundary
            let d = dist[i];
            let a_line = (-0.5 * (d/sigma_line)*(d/sigma_line)).exp().clamp(0.0, 1.0);
            // Premultiplied composite of fill and line
            let af = a_fill; let al = a_line; let a_out = af + al * (1.0 - af);
            if a_out <= 0.0 {
                buf.put_pixel(x as u32, y as u32, Rgba([0,0,0,0]));
            } else {
                let cf_r = fill_rgb.0 as f32 / 255.0 * af;
                let cf_g = fill_rgb.1 as f32 / 255.0 * af;
                let cf_b = fill_rgb.2 as f32 / 255.0 * af;
                let cl_r = line_rgb.0 as f32 / 255.0 * al;
                let cl_g = line_rgb.1 as f32 / 255.0 * al;
                let cl_b = line_rgb.2 as f32 / 255.0 * al;
                let c_r = cf_r + cl_r * (1.0 - af);
                let c_g = cf_g + cl_g * (1.0 - af);
                let c_b = cf_b + cl_b * (1.0 - af);
                let r = ((c_r / a_out)*255.0).round().clamp(0.0, 255.0) as u8;
                let g = ((c_g / a_out)*255.0).round().clamp(0.0, 255.0) as u8;
                let b = ((c_b / a_out)*255.0).round().clamp(0.0, 255.0) as u8;
                let a = (a_out*255.0).round().clamp(0.0, 255.0) as u8;
                buf.put_pixel(x as u32, y as u32, Rgba([r,g,b,a]));
            }
        } }
        let mut out_bytes: Vec<u8> = Vec::new();
        { let img_dyn = DynamicImage::ImageRgba8(buf); let mut cursor = std::io::Cursor::new(&mut out_bytes); img_dyn.write_to(&mut cursor, image::ImageFormat::Png).map_err(|e| AppError::Internal(e.to_string()))?; }
        let mask_png_b64 = B64.encode(out_bytes);

        // 2) Upsample mask back to original image dimensions and cut RGBA
        let orig = dyn_img.to_rgba8();
        let (ow, oh) = (orig.width() as usize, orig.height() as usize);
        let mut out_rgba = ImageBuffer::<Rgba<u8>, Vec<u8>>::new(ow as u32, oh as u32);
        for y in 0..oh {
            let v = (y as f32 + 0.5) / (oh as f32) * (h as f32) - 0.5;
            let y1 = v.floor().clamp(0.0, (h-1) as f32) as usize;
            let y2 = ((y1 + 1).min(h-1)) as usize;
            let fy = v - y1 as f32;
            for x in 0..ow {
                let u = (x as f32 + 0.5) / (ow as f32) * (w as f32) - 0.5;
                let x1 = u.floor().clamp(0.0, (w-1) as f32) as usize;
                let x2 = ((x1 + 1).min(w-1)) as usize;
                let fx = u - x1 as f32;
                let a11 = mask_bin[y1*w + x1] as f32;
                let a21 = mask_bin[y1*w + x2] as f32;
                let a12 = mask_bin[y2*w + x1] as f32;
                let a22 = mask_bin[y2*w + x2] as f32;
                let atop = a11*(1.0-fx) + a21*fx;
                let abot = a12*(1.0-fx) + a22*fx;
                let a = atop*(1.0-fy) + abot*fy; // 0..1
                let px = orig.get_pixel(x as u32, y as u32);
                let alpha = (a*255.0).round().clamp(0.0, 255.0) as u8;
                out_rgba.put_pixel(x as u32, y as u32, Rgba([px[0], px[1], px[2], alpha]));
            }
        }
        let mut cut_bytes: Vec<u8> = Vec::new();
        { let img_dyn = DynamicImage::ImageRgba8(out_rgba); let mut cursor = std::io::Cursor::new(&mut cut_bytes); img_dyn.write_to(&mut cursor, image::ImageFormat::Png).map_err(|e| AppError::Internal(e.to_string()))?; }
        let masked_region_png_b64 = Some(B64.encode(cut_bytes));

        let i0 = flat.get(0).copied().unwrap_or(0.0);
        let i1 = flat.get(1).copied().unwrap_or(0.0);
        let i2 = flat.get(2).copied().unwrap_or(0.0);
        (mask_png_b64, masked_region_png_b64, [i0, i1, i2], best_idx)
    };
    let dur_ms = t0.elapsed().as_millis();

    let resp = SegmentResponse {
        request_id: req.request_id.unwrap_or_else(Uuid::new_v4),
        model: model_sel,
        iou: iou_arr,
        best_idx,
        inference_ms: dur_ms,
        mask_png_b64,
        masked_region_png_b64,
    };

    Ok(Json(resp))
}

async fn get_or_load_session(state: &AppState, size: Sam2ModelSize) -> Result<Arc<RwLock<Session>>> {
    // Cache key is model size
    let mut map = state.models.write().await;
    if let Some(s) = map.get(&size) { return Ok(Arc::clone(s)); }

    let filename = size.to_filename();
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(filename);
    let path_str = path.to_string_lossy().to_string();

    // Tune ONNX Runtime session options for CPU inference
    let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let intra: usize = std::cmp::max(1, cpus / 2);
    let inter: usize = 1;

    // Build session with CPU-friendly options
    let builder = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(intra)?
        .with_inter_threads(inter)?
        .with_memory_pattern(true)?
        .with_config_entry("session.set_denormal_as_zero", "1")?;

    let session = builder.commit_from_file(path_str)?;

    let arc = Arc::new(RwLock::new(session));
    map.insert(size, Arc::clone(&arc));
    Ok(arc)
}

fn preprocess_image(img: &DynamicImage) -> anyhow::Result<Array4<f32>> {
    // Resize to 1024x1024, convert to RGB, CHW, and normalize with ImageNet stats
    let img = img.to_rgb8();
    let resized = image::imageops::resize(&img, 1024, 1024, FilterType::Lanczos3);
    let (w, h) = (resized.width() as usize, resized.height() as usize);
    let mut arr: Array4<f32> = Array::zeros((1, 3, h, w));

    // ImageNet mean/std as used by most SAM/SAM2 pipelines
    let mean = [0.485f32, 0.456, 0.406];
    let std = [0.229f32, 0.224, 0.225];

    for y in 0..h { for x in 0..w {
        let p = resized.get_pixel(x as u32, y as u32);
        let r = p[0] as f32 / 255.0;
        let g = p[1] as f32 / 255.0;
        let b = p[2] as f32 / 255.0;
        arr[[0, 0, y, x]] = (r - mean[0]) / std[0];
        arr[[0, 1, y, x]] = (g - mean[1]) / std[1];
        arr[[0, 2, y, x]] = (b - mean[2]) / std[2];
    }}
    Ok(arr)
}

mod shared;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let (prometheus_layer, metric_handle) = PrometheusMetricLayer::pair();

    let state = AppState { models: Arc::new(RwLock::new(HashMap::new())) };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let api_router = Router::new()
        .route("/models", get(get_models))
        .route("/segment", post(segment));

    let app = Router::new()
        .merge(SwaggerUi::new("/docs").url("/api-docs/openapi.json", ApiDoc::openapi()))
        .nest("/api", api_router)
        .fallback_service(ServeDir::new("sam2_server/static"))
        .layer(DefaultBodyLimit::max(50 * 1024 * 1024))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .layer(prometheus_layer)
        .with_state(state);

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], 8080));
    info!("server listening on http://{}", addr);
    axum::serve(tokio::net::TcpListener::bind(addr).await?, app).await?;
    Ok(())
}

