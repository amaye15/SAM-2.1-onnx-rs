#!/usr/bin/env python3
"""
Export all SAM 2.1 model sizes to ONNX format.
Supports: tiny, small, base-plus, and large models.
"""

import os
import sys
import subprocess
import shutil

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
from huggingface_hub import snapshot_download

# Add the sam2 directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sam2'))

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Model configurations
MODEL_CONFIGS = {
    'tiny': {
        'hf_id': 'facebook/sam2.1-hiera-tiny',
        'config_file': 'configs/sam2.1/sam2.1_hiera_t.yaml',
        'checkpoint_name': 'sam2.1_hiera_tiny.pt',
        'bb_feat_sizes': [(256, 256), (128, 128), (64, 64)]
    },
    'small': {
        'hf_id': 'facebook/sam2.1-hiera-small',
        'config_file': 'configs/sam2.1/sam2.1_hiera_s.yaml',
        'checkpoint_name': 'sam2.1_hiera_small.pt',
        'bb_feat_sizes': [(256, 256), (128, 128), (64, 64)]
    },
    'base_plus': {
        'hf_id': 'facebook/sam2.1-hiera-base-plus',
        'config_file': 'configs/sam2.1/sam2.1_hiera_b+.yaml',
        'checkpoint_name': 'sam2.1_hiera_base_plus.pt',
        'bb_feat_sizes': [(256, 256), (128, 128), (64, 64)]
    },
    'large': {
        'hf_id': 'facebook/sam2.1-hiera-large',
        'config_file': 'configs/sam2.1/sam2.1_hiera_l.yaml',
        'checkpoint_name': 'sam2.1_hiera_large.pt',
        'bb_feat_sizes': [(256, 256), (128, 128), (64, 64)]
    }
}
def model_local_dir_from_size(model_size: str) -> str:
    """Return the local download directory for a given model size."""
    return f"./sam2.1-hiera-{model_size.replace('_', '-')}-downloaded"


def cleanup_downloaded_files_for_model(model_size: str) -> None:
    """Delete the downloaded files for a model size after successful export/tests.

    Safety checks ensure we only remove the expected snapshot directory.
    """
    local_dir = model_local_dir_from_size(model_size)
    try:
        # Safety: ensure directory exists and name matches expected pattern
        base = os.path.basename(os.path.normpath(local_dir))
        if os.path.isdir(local_dir) and base.startswith("sam2.1-hiera-") and base.endswith("-downloaded"):
            shutil.rmtree(local_dir)
            print(f"🧹 Cleaned up downloaded files at: {local_dir}")
        else:
            print(f"⚠ Skipping cleanup; unexpected directory path: {local_dir}")
    except Exception as e:
        print(f"⚠ Failed to clean up {local_dir}: {e}")


class SAM2CompleteModel(nn.Module):
    """Complete SAM2 model wrapper for ONNX export."""

    def __init__(self, sam2_model, bb_feat_sizes):
        super().__init__()
        self.sam2_model = sam2_model
        self.image_encoder = sam2_model.image_encoder
        self.prompt_encoder = sam2_model.sam_prompt_encoder
        self.mask_decoder = sam2_model.sam_mask_decoder
        self.no_mem_embed = sam2_model.no_mem_embed
        self.directly_add_no_mem_embed = sam2_model.directly_add_no_mem_embed
        self.bb_feat_sizes = bb_feat_sizes

        # Precompute image_pe as a buffer for constant folding optimization
        with torch.no_grad():
            self.register_buffer(
                "image_pe_const",
                self.prompt_encoder.get_dense_pe()
            )

    def forward(self, image, point_coords, point_labels):
        """
        Complete SAM2 forward pass.

        Args:
            image: [1, 3, 1024, 1024] - Input image
            point_coords: [1, N, 2] - Point coordinates in pixels
            point_labels: [1, N] - Point labels (1=positive, 0=negative)

        Returns:
            masks: [1, 3, 1024, 1024] - Predicted masks
            iou_predictions: [1, 3] - IoU predictions
        """
        # 1. Image encoding
        backbone_out = self.sam2_model.forward_image(image)
        _, vision_feats, _, _ = self.sam2_model._prepare_backbone_features(backbone_out)

        # Add no_mem_embed if needed
        if self.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.no_mem_embed

        # Process features
        feats = []
        for feat, feat_size in zip(vision_feats[::-1], self.bb_feat_sizes[::-1]):
            feat_reshaped = feat.permute(1, 2, 0).reshape(1, -1, feat_size[0], feat_size[1])
            feats.append(feat_reshaped)
        feats = feats[::-1]

        image_embeddings = feats[-1]
        high_res_features = feats[:-1]

        # 2. Prompt encoding
        points = (point_coords, point_labels)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points, boxes=None, masks=None
        )

        # 3. Mask decoding
        low_res_masks, iou_predictions, _, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.image_pe_const,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        # 4. Upscale masks
        masks = torch.nn.functional.interpolate(
            low_res_masks, size=(1024, 1024), mode='bilinear', align_corners=False
        )

        return masks, iou_predictions

def download_model(model_size):
    """Download model from Hugging Face Hub."""
    config = MODEL_CONFIGS[model_size]
    local_dir = f"./sam2.1-hiera-{model_size.replace('_', '-')}-downloaded"

    print(f"Downloading {model_size} model from {config['hf_id']}...")

    if os.path.exists(local_dir):
        print(f"✓ Model directory already exists: {local_dir}")
        return local_dir

    try:
        snapshot_download(
            repo_id=config['hf_id'],
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"✓ Model downloaded to: {local_dir}")
        return local_dir
    except Exception as e:
        print(f"✗ Failed to download {model_size} model: {e}")
        return None

def load_sam2_model(model_size):
    """Load SAM2 model of specified size."""
    config = MODEL_CONFIGS[model_size]
    local_dir = download_model(model_size)

    if not local_dir:
        raise RuntimeError(f"Failed to download {model_size} model")

    config_file = config['config_file']
    ckpt_path = os.path.join(local_dir, config['checkpoint_name'])

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading {model_size} model...")
    sam2_model = build_sam2(
        config_file=config_file,
        ckpt_path=ckpt_path,
        device="cpu",
        mode="eval"
    )

    print(f"✓ {model_size} model loaded successfully")
    return sam2_model, config['bb_feat_sizes']

def create_test_inputs():
    """Create test inputs for the model."""
    image = torch.randn(1, 3, 1024, 1024)
    point_coords = torch.tensor([[[512.0, 512.0]]], dtype=torch.float32)
    point_labels = torch.tensor([[1]], dtype=torch.float32)
    return image, point_coords, point_labels

def test_model_wrapper(sam2_model, bb_feat_sizes, model_size):
    """Test the model wrapper before ONNX export."""
    print(f"\nTesting {model_size} model wrapper...")

    wrapper = SAM2CompleteModel(sam2_model, bb_feat_sizes)
    wrapper.eval()

    image, point_coords, point_labels = create_test_inputs()

    with torch.no_grad():
        masks, iou_predictions = wrapper(image, point_coords, point_labels)

    print(f"✓ {model_size} model wrapper test successful")
    print(f"  - Masks shape: {masks.shape}")
    print(f"  - IoU predictions shape: {iou_predictions.shape}")

    return wrapper

def slim_onnx_model_with_onnxslim(input_path: str, image_shape=(1,3,1024,1024), num_points=1) -> bool:
    """Slim an ONNX model in-place using onnxslim via uvx.

    Returns True if slimming succeeded and replaced the original file.
    """
    try:
        # Build command; include onnxruntime so model_check can run
        slim_path = input_path + ".slim.onnx"
        model_check_inputs = [
            f"image:{','.join(map(str, image_shape))}",
            f"point_coords:1,{num_points},2",
            f"point_labels:1,{num_points}",
        ]
        cmd = [
            "uvx", "--with", "onnxruntime", "onnxslim",
            input_path, slim_path,
            "--model-check",
            "--model-check-inputs",
            *model_check_inputs,
        ]
        print(f"Running ONNXSlim: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print("ONNXSlim failed; keeping original model.")
            if res.stderr:
                print(res.stderr[:1000])
            return False
        if not os.path.exists(slim_path):
            print("ONNXSlim did not produce output; keeping original model.")
            return False
        # Verify and replace original
        try:
            onnx_model = onnx.load(slim_path)
            onnx.checker.check_model(onnx_model)
        except Exception as e:
            print(f"Slimmed model failed ONNX checker: {e}; keeping original.")
            try:
                os.remove(slim_path)
            except Exception:
                pass
            return False
        # Replace original file atomically
        orig_size = os.path.getsize(input_path)
        slim_size = os.path.getsize(slim_path)
        os.replace(slim_path, input_path)
        print(f"✓ Replaced original ONNX with slimmed model. Size: {orig_size/(1024**2):.2f} MB -> {slim_size/(1024**2):.2f} MB")
        return True
    except FileNotFoundError as e:
        print(f"ONNXSlim or uvx not found: {e}. Skipping slimming.")
    except Exception as e:
        print(f"Unexpected error during ONNXSlim: {e}. Skipping slimming.")
    return False

def export_model_to_onnx(sam2_model, bb_feat_sizes, model_size):
    """Export SAM2 model to ONNX format."""
    output_path = f"sam2_{model_size}.onnx"
    print(f"\nExporting {model_size} model to ONNX...")

    wrapper = SAM2CompleteModel(sam2_model, bb_feat_sizes)
    wrapper.eval()

    image, point_coords, point_labels = create_test_inputs()

    try:
        torch.onnx.export(
            wrapper,
            (image, point_coords, point_labels),
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['image', 'point_coords', 'point_labels'],
            output_names=['masks', 'iou_predictions'],
            dynamic_axes={
                'image': {0: 'batch_size'},
                'point_coords': {0: 'batch_size', 1: 'num_points'},
                'point_labels': {0: 'batch_size', 1: 'num_points'},
                'masks': {0: 'batch_size'},
                'iou_predictions': {0: 'batch_size'}
            },
            training=torch.onnx.TrainingMode.EVAL,
            keep_initializers_as_inputs=False,
            verbose=False
        )

        print(f"✓ {model_size} model exported to: {output_path}")

        # Verify the exported model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"✓ ONNX model verification passed")

        # Get model info
        file_size = os.path.getsize(output_path)
        print(f"✓ ONNX model size: {file_size / (1024**2):.2f} MB")

        # Try to slim the ONNX model in-place with onnxslim
        slimmed = slim_onnx_model_with_onnxslim(output_path, image_shape=(1,3,1024,1024), num_points=1)
        if slimmed:
            # Recompute size after slimming
            file_size = os.path.getsize(output_path)
            print(f"✓ Slimmed ONNX model size: {file_size / (1024**2):.2f} MB")
        else:
            print("⚠ Skipping slimming or slimming failed; using original ONNX model.")

        return output_path, file_size

    except Exception as e:
        print(f"✗ Error exporting {model_size} to ONNX: {e}")
        raise

def test_onnx_model(onnx_path, original_model, bb_feat_sizes, model_size):
    """Test the ONNX model and compare with original."""
    print(f"\nTesting {model_size} ONNX model...")

    try:
        # Load ONNX model with CPU-optimized session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = True
        sess_options.enable_cpu_mem_arena = True
        try:
            import os as _os
            sess_options.intra_op_num_threads = max(1, (_os.cpu_count() or 1) // 2)
        except Exception:
            pass
        sess_options.inter_op_num_threads = 1

        providers = [("CPUExecutionProvider", {"use_arena": True})]
        ort_session = ort.InferenceSession(onnx_path, sess_options, providers=providers)

        image, point_coords, point_labels = create_test_inputs()

        # Run ONNX inference
        ort_inputs = {
            'image': image.numpy(),
            'point_coords': point_coords.numpy(),
            'point_labels': point_labels.numpy()
        }

        onnx_outputs = ort_session.run(None, ort_inputs)
        onnx_masks, onnx_iou = onnx_outputs

        # Compare with original model
        wrapper = SAM2CompleteModel(original_model, bb_feat_sizes)
        wrapper.eval()

        with torch.no_grad():
            torch_masks, torch_iou = wrapper(image, point_coords, point_labels)
            torch_masks = torch_masks.numpy()
            torch_iou = torch_iou.numpy()

        # Calculate differences
        mask_max_diff = abs(onnx_masks - torch_masks).max()
        iou_max_diff = abs(onnx_iou - torch_iou).max()

        print(f"✓ {model_size} ONNX inference successful")
        print(f"  - Masks max difference: {mask_max_diff:.6f}")
        print(f"  - IoU max difference: {iou_max_diff:.6f}")

        tolerance = 1e-3
        success = mask_max_diff < tolerance and iou_max_diff < tolerance

        if success:
            print(f"✓ Numerical accuracy within tolerance ({tolerance})")
        else:
            print(f"⚠ Some differences exceed tolerance ({tolerance})")

        return success

    except Exception as e:
        print(f"✗ Error testing {model_size} ONNX model: {e}")
        return False

def export_all_models():
    """Export all SAM2.1 model sizes to ONNX."""
    print("=== SAM 2.1 All Models ONNX Export ===\n")

    results = {}

    for model_size in MODEL_CONFIGS.keys():
        try:
            print(f"\n{'='*50}")
            print(f"Processing {model_size.upper()} model")
            print(f"{'='*50}")

            # Load model
            sam2_model, bb_feat_sizes = load_sam2_model(model_size)

            # Test wrapper
            wrapper = test_model_wrapper(sam2_model, bb_feat_sizes, model_size)

            # Export to ONNX
            onnx_path, file_size = export_model_to_onnx(sam2_model, bb_feat_sizes, model_size)

            # Test ONNX model
            success = test_onnx_model(onnx_path, sam2_model, bb_feat_sizes, model_size)

            # Cleanup downloaded files only if export + test succeeded
            if success:
                cleanup_downloaded_files_for_model(model_size)
            else:
                print(f"⚠ Skipping cleanup for {model_size}; export/test not fully successful.")

            results[model_size] = {
                'onnx_path': onnx_path,
                'file_size_mb': file_size / (1024**2),
                'success': success
            }

            print(f"✓ {model_size} model export completed!")

        except Exception as e:
            print(f"✗ Failed to export {model_size} model: {e}")
            results[model_size] = {
                'error': str(e),
                'success': False
            }

    # Print summary
    print(f"\n{'='*60}")
    print("EXPORT SUMMARY")
    print(f"{'='*60}")

    for model_size, result in results.items():
        if result['success']:
            print(f"✓ {model_size:12} - {result['onnx_path']:20} ({result['file_size_mb']:.1f} MB)")
        else:
            print(f"✗ {model_size:12} - FAILED")

    return results

if __name__ == "__main__":
    export_all_models()
