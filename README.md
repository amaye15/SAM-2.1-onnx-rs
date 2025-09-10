---
title: Steroid SAM 2.1
emoji: "👀"
colorFrom: purple
colorTo: green
sdk: docker
app_port: 8080
base_path: /
pinned: false
license: mit
short_description: A point prompt app using SAM 2.1 for generating image masks.
tags:
  - segmentation
  - computer-vision
  - onnx
  - rust
  - axum
  - docker
  - interactive
  - sam2
---

## Steroid SAM 2.1 (ONNX, Rust)
A lightweight point-prompting app powered by SAM 2.1 models exported to ONNX and served by a Rust/Axum web server. It provides a simple web UI and a clean JSON API to generate segmentation masks from positive/negative clicks.

## Features
- Interactive web UI for click-based segmentation (positive/negative points)
- Multiple SAM 2.1 model sizes: Tiny, Small, BasePlus, Large
- Fast ONNX Runtime CPU inference (no GPU required)
- Swagger UI at /docs for API exploration
- Docker image for Hugging Face Spaces or local deployment


## Quickstart (on Hugging Face Spaces)
This Space uses sdk: docker and builds from the Dockerfile at the repository root.
- After the container starts, open the Space
- Click “Load Sample” (or upload your own image)
- Choose Positive/Negative mode and click on the image to add points
- Select a model size (Large by default)
- Click “Run” to generate the mask overlay
- Optionally download the masked region PNG

Note: The app serves a static UI from sam2_server/static and exposes an API under /api. The OpenAPI/Swagger UI is available at /docs.


## Running locally
Option A — Cargo (development):
1) Ensure Rust is installed
2) From the repo root, run:
   - cargo run -p sam2_server
3) Open http://127.0.0.1:8080 (binds to 0.0.0.0:8080 in containers and Spaces)

Option B — Docker:
1) Build
   - docker build -t steroid-sam21 .
2) Run
   - docker run --rm -p 8080:8080 steroid-sam21
3) Open http://localhost:8080

Notes
- The server now binds to 0.0.0.0 and honors the PORT env var in containers/Spaces (defaults to 8080 locally). This makes it reachable behind the Spaces proxy.


## API
Base URL
- Local: http://127.0.0.1:8080
- In the Space: / (container root) — server binds 0.0.0.0 and will respect PORT
- Swagger: /docs

Endpoints
1) GET /api/models
   - Lists available model sizes
   - Response: { "models": ["Tiny", "Small", "BasePlus", "Large"] }

2) POST /api/segment
   - Runs segmentation for an image with click prompts
   - Request (JSON):
     {
       "image_b64": "<base64 PNG/JPEG>",
       "model": "Tiny|Small|BasePlus|Large",
       "points": [{"x": <float>, "y": <float>, "label": 1|0}, ...],
       "request_id": "<uuid, optional>",
       "threshold": <float 0..1, optional>
     }
   - Response (JSON):
     {
       "request_id": "<uuid>",
       "model": "<chosen model>",
       "iou": [f32, f32, f32],
       "best_idx": 0|1|2,
       "inference_ms": <u128>,
       "mask_png_b64": "<base64 PNG overlay (1024x1024)>",
       "masked_region_png_b64": "<base64 PNG of original-resolution cutout, optional>"
     }

Example curl
- Encode an image to base64 (PNG/JPEG) and send a few points (in 1024x1024 model space):

  curl -s -X POST \
       -H 'Content-Type: application/json' \
       -d '{
            "image_b64": "<BASE64>",
            "model": "Large",
            "points": [{"x": 512.0, "y": 512.0, "label": 1}],
            "threshold": 0.5
          }' \
       http://127.0.0.1:8080/api/segment | jq


## Models
- ONNX files available in the repository root. For the free Space deployment we include only sam2_tiny.onnx to stay under the 1 GB Space limit.
- The Dockerfile copies sam2_tiny.onnx into the image; the server loads it on first use and caches it.


## Tech stack
- Rust, Axum, Tokio, utoipa (OpenAPI), utoipa-swagger-ui
- ONNX Runtime (ort crate) with ndarray
- Simple static frontend (vanilla JS/Canvas)


## License
MIT


## Acknowledgements
- SAM 2.1 by Meta AI
- ONNX Runtime

