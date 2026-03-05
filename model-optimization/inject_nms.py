#!/usr/bin/env python3
"""
inject_efficient_nms.py

Inject TensorRT EfficientNMS plugin (EfficientNMS_TRT) into a YOLO-style ONNX model.

Your model output: [B, 84, 8400]
Assumption:
- 84 = 4 + num_classes (default 80)
- Layout is [B, 84, N]. We transpose -> [B, N, 84]
- boxes  = [:, :, 0:4]      -> [B, N, 4]
- scores = [:, :, 4:4+C]    -> [B, N, C]

Flags:
- --boxes-are-xywh: convert xywh(center) to xyxy before NMS (common for YOLO exports)
- --apply-sigmoid: apply Sigmoid to class scores before NMS (use if scores are logits)

Usage example:
  python inject_efficient_nms.py --model-in model.onnx --model-out model_with_nms.onnx --boxes-are-xywh
"""

import argparse
from typing import Optional

import numpy as np
import onnx
import onnx_graphsurgeon as gs


def inspect_model(model_path: str) -> None:
    m = onnx.load(model_path)
    print("Inputs:")
    for i in m.graph.input:
        print(" ", i.name)
    print("Outputs:")
    for o in m.graph.output:
        print(" ", o.name)


def _xywh_to_xyxy(graph: gs.Graph, boxes_xywh: gs.Tensor, name_prefix: str = "boxes") -> gs.Tensor:
    """
    Convert [B, N, 4] boxes from xywh(center) -> xyxy.
    """
    # Slices along last dim
    x = gs.Variable(f"{name_prefix}_x", dtype=np.float32)
    y = gs.Variable(f"{name_prefix}_y", dtype=np.float32)
    w = gs.Variable(f"{name_prefix}_w", dtype=np.float32)
    h = gs.Variable(f"{name_prefix}_h", dtype=np.float32)

    graph.nodes.append(gs.Node("Slice", inputs=[boxes_xywh], outputs=[x],
                               attrs={"starts": [0], "ends": [1], "axes": [2]}))
    graph.nodes.append(gs.Node("Slice", inputs=[boxes_xywh], outputs=[y],
                               attrs={"starts": [1], "ends": [2], "axes": [2]}))
    graph.nodes.append(gs.Node("Slice", inputs=[boxes_xywh], outputs=[w],
                               attrs={"starts": [2], "ends": [3], "axes": [2]}))
    graph.nodes.append(gs.Node("Slice", inputs=[boxes_xywh], outputs=[h],
                               attrs={"starts": [3], "ends": [4], "axes": [2]}))

    half = gs.Constant(f"{name_prefix}_half", values=np.array([0.5], dtype=np.float32))

    w_half = gs.Variable(f"{name_prefix}_w_half", dtype=np.float32)
    h_half = gs.Variable(f"{name_prefix}_h_half", dtype=np.float32)
    graph.nodes.append(gs.Node("Mul", inputs=[w, half], outputs=[w_half]))
    graph.nodes.append(gs.Node("Mul", inputs=[h, half], outputs=[h_half]))

    x1 = gs.Variable(f"{name_prefix}_x1", dtype=np.float32)
    y1 = gs.Variable(f"{name_prefix}_y1", dtype=np.float32)
    x2 = gs.Variable(f"{name_prefix}_x2", dtype=np.float32)
    y2 = gs.Variable(f"{name_prefix}_y2", dtype=np.float32)

    graph.nodes.append(gs.Node("Sub", inputs=[x, w_half], outputs=[x1]))
    graph.nodes.append(gs.Node("Sub", inputs=[y, h_half], outputs=[y1]))
    graph.nodes.append(gs.Node("Add", inputs=[x, w_half], outputs=[x2]))
    graph.nodes.append(gs.Node("Add", inputs=[y, h_half], outputs=[y2]))

    boxes_xyxy = gs.Variable(f"{name_prefix}_xyxy", dtype=np.float32)
    graph.nodes.append(gs.Node("Concat", inputs=[x1, y1, x2, y2], outputs=[boxes_xyxy], attrs={"axis": 2}))
    return boxes_xyxy


def inject_efficient_nms(
    model_in: str,
    model_out: str,
    output_name: Optional[str],
    num_classes: int,
    max_output_boxes: int,
    score_threshold: float,
    iou_threshold: float,
    apply_sigmoid: bool,
    boxes_are_xywh: bool,
    class_agnostic: bool,
    keep_old_outputs: bool,
) -> None:
    model = onnx.load(model_in)
    graph = gs.import_onnx(model)

    # Pick which tensor to treat as the raw detection output
    if output_name is None:
        if not graph.outputs:
            raise RuntimeError("Model has no graph outputs. Provide --output-name explicitly.")
        raw_out = graph.outputs[0]
        output_name = raw_out.name
    else:
        tensors = graph.tensors()
        if output_name not in tensors:
            raise RuntimeError(
                f"Couldn't find tensor named '{output_name}'. "
                f"Try running with --inspect to see outputs."
            )
        raw_out = tensors[output_name]

    # Transpose [B, 84, N] -> [B, N, 84]
    out_t = gs.Variable(f"{output_name}_t", dtype=np.float32)
    graph.nodes.append(gs.Node("Transpose", inputs=[raw_out], outputs=[out_t], attrs={"perm": [0, 2, 1]}))

    # Slice boxes and scores from last dim
    boxes_raw = gs.Variable("boxes_raw", dtype=np.float32)
    graph.nodes.append(gs.Node("Slice", inputs=[out_t], outputs=[boxes_raw],
                               attrs={"starts": [0], "ends": [4], "axes": [2]}))

    scores_raw = gs.Variable("scores_raw", dtype=np.float32)
    graph.nodes.append(gs.Node("Slice", inputs=[out_t], outputs=[scores_raw],
                               attrs={"starts": [4], "ends": [4 + num_classes], "axes": [2]}))

    # Optional sigmoid for scores
    if apply_sigmoid:
        scores = gs.Variable("scores_sigmoid", dtype=np.float32)
        graph.nodes.append(gs.Node("Sigmoid", inputs=[scores_raw], outputs=[scores]))
        score_activation_attr = 0  # already applied
    else:
        scores = scores_raw
        score_activation_attr = 0  # don't ask plugin to apply activation

    # Optional xywh(center) -> xyxy conversion
    if boxes_are_xywh:
        boxes = _xywh_to_xyxy(graph, boxes_raw, "boxes")
    else:
        boxes = boxes_raw

    # EfficientNMS outputs
    B = "B"
    K = int(max_output_boxes)

    num_dets = gs.Variable("num_dets", dtype=np.int32, shape=[B, 1])
    det_boxes = gs.Variable("det_boxes", dtype=np.float32, shape=[B, K, 4])
    det_scores = gs.Variable("det_scores", dtype=np.float32, shape=[B, K])
    det_classes = gs.Variable("det_classes", dtype=np.int32, shape=[B, K])

    nms_node = gs.Node(
        op="EfficientNMS_TRT",
        name="EfficientNMS",
        inputs=[boxes, scores],
        outputs=[num_dets, det_boxes, det_scores, det_classes],
        attrs={
            "background_class": -1,
            "box_coding": 0,  # corner (xyxy)
            "iou_threshold": float(iou_threshold),
            "score_threshold": float(score_threshold),
            "max_output_boxes": int(max_output_boxes),
            "score_activation": int(score_activation_attr),
            "class_agnostic": int(1 if class_agnostic else 0),
        },
    )
    graph.nodes.append(nms_node)

    if keep_old_outputs:
        # Keep existing outputs and append new ones
        # (useful for debugging, but Triton configs may need updating)
        graph.outputs = list(graph.outputs) + [num_dets, det_boxes, det_scores, det_classes]
    else:
        # Replace outputs with NMS outputs
        graph.outputs = [num_dets, det_boxes, det_scores, det_classes]

    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), model_out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject TensorRT EfficientNMS into a YOLO ONNX graph.")
    parser.add_argument("--model-in", required=True, help="Path to input ONNX model")
    parser.add_argument("--model-out", required=True, help="Path to output ONNX model (with NMS)")
    parser.add_argument("--output-name", default=None,
                        help="Name of the raw detection output tensor. Default: first model output.")
    parser.add_argument("--num-classes", type=int, default=80, help="Number of classes (default: 80)")
    parser.add_argument("--max-output-boxes", type=int, default=100, help="Max detections per image (default: 100)")
    parser.add_argument("--score-threshold", type=float, default=0.25, help="Score threshold (default: 0.25)")
    parser.add_argument("--iou-threshold", type=float, default=0.45, help="IoU threshold (default: 0.45)")

    parser.add_argument("--apply-sigmoid", action="store_true",
                        help="Apply sigmoid to class scores before NMS (use if outputs are logits).")
    parser.add_argument("--boxes-are-xywh", action="store_true",
                        help="Convert boxes from xywh(center) to xyxy before NMS.")
    parser.add_argument("--class-agnostic", action="store_true", help="Class-agnostic NMS (default: off).")

    parser.add_argument("--keep-old-outputs", action="store_true",
                        help="Keep the original model outputs and append NMS outputs (debugging).")
    parser.add_argument("--inspect", action="store_true",
                        help="Print model inputs/outputs and exit.")

    args = parser.parse_args()

    if args.inspect:
        inspect_model(args.model_in)
        return

    inject_efficient_nms(
        model_in=args.model_in,
        model_out=args.model_out,
        output_name=args.output_name,
        num_classes=args.num_classes,
        max_output_boxes=args.max_output_boxes,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        apply_sigmoid=args.apply_sigmoid,
        boxes_are_xywh=args.boxes_are_xywh,
        class_agnostic=args.class_agnostic,
        keep_old_outputs=args.keep_old_outputs,
    )

    print("Done.")
    print(" Input :", args.model_in)
    print(" Output:", args.model_out)
    print(" If TensorRT complains about plugin not found, ensure you build with TensorRT that includes EfficientNMS_TRT.")


if __name__ == "__main__":
    main()