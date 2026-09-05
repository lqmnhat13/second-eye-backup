"""Optional local metric Depth Anything V2 backend (weights must be pre-downloaded)."""
import numpy as np


class MetricDepthBackend:
    def __init__(self, model_path: str, device: str = "cpu"):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.torch = torch
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_path, local_files_only=True)
        if getattr(self.model.config, "depth_estimation_type", None) != "metric":
            raise ValueError("Requires metric depth weights; relative depth is not meters")
        self.model.to(device).eval()

    def predict(self, frame):
        inputs = self.processor(images=frame[:, :, ::-1].copy(), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.inference_mode():
            output = self.model(**inputs)
            depth = self.torch.nn.functional.interpolate(
                output.predicted_depth.unsqueeze(1), size=frame.shape[:2],
                mode="bilinear", align_corners=False,
            )[0, 0]
        return depth.cpu().numpy()

    @staticmethod
    def object_distance(depth, bbox):
        """Median of central ROI; reject sparse/mixed depth instead of inventing confidence.

        A box is not a segmentation mask. Even uniform background may be wrong;
        the returned distance remains unvalidated, never a confidence probability.
        """
        x1, y1, x2, y2 = bbox
        dx, dy = (x2 - x1) // 4, (y2 - y1) // 4
        roi = depth[y1 + dy:y2 - dy, x1 + dx:x2 - dx]
        values = roi[np.isfinite(roi) & (roi > 0)]
        if values.size < 16 or values.size < roi.size * 0.8:
            return None
        q25, median, q75 = np.percentile(values, [25, 50, 75])
        if (q75 - q25) / median > 0.35:
            return None
        return float(median)
