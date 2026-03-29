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
- --boxes-are-xywh: convert xywh(center) to corner boxes before NMS (common for YOLO exports)
- --apply-sigmoid: apply Sigmoid to class scores before NMS (use if scores are logits)
- --box-order: choose whether the plugin sees corner boxes as xyxy or yxyx

Usage example:
  python inject_efficient_nms.py --model-in model.onnx --model-out model_with_nms.onnx --boxes-are-xywh --box-order yxyx
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


def _make_slice(
    graph: gs.Graph,
    data: gs.Tensor,
    output_name: str,
    starts: list[int],
    ends: list[int],
    axes: list[int],
    *,
    steps: Optional[list[int]] = None,
    dtype=np.float32,
) -> gs.Tensor:
    """
    Build a Slice node using tensor inputs instead of legacy attributes.

    TensorRT is generally happier with the modern ONNX Slice form:
    data, starts, ends, axes, steps
    """
    output = gs.Variable(output_name, dtype=dtype)
    slice_inputs = [
        data,
        gs.Constant(f"{output_name}_starts", values=np.asarray(starts, dtype=np.int64)),
        gs.Constant(f"{output_name}_ends", values=np.asarray(ends, dtype=np.int64)),
        gs.Constant(f"{output_name}_axes", values=np.asarray(axes, dtype=np.int64)),
    ]

    if steps is not None:
        slice_inputs.append(
            gs.Constant(f"{output_name}_steps", values=np.asarray(steps, dtype=np.int64))
        )

    graph.nodes.append(gs.Node("Slice", inputs=slice_inputs, outputs=[output]))
    return output


def _reorder_boxes(graph: gs.Graph, boxes: gs.Tensor, box_order: str, name_prefix: str = "boxes") -> gs.Tensor:
    """
    Reorder [B, N, 4] corner boxes between xyxy and yxyx layouts.
    """
    if box_order == "xyxy":
        return boxes
    if box_order != "yxyx":
        raise ValueError(f"Unsupported box_order: {box_order}")

    # Use distinct tensor names here so GraphSurgeon does not see them as the
    # same tensors produced earlier by the xywh->corner conversion path.
    x1 = _make_slice(graph, boxes, f"{name_prefix}_reorder_x1", [0], [1], [2])
    y1 = _make_slice(graph, boxes, f"{name_prefix}_reorder_y1", [1], [2], [2])
    x2 = _make_slice(graph, boxes, f"{name_prefix}_reorder_x2", [2], [3], [2])
    y2 = _make_slice(graph, boxes, f"{name_prefix}_reorder_y2", [3], [4], [2])

    reordered = gs.Variable(f"{name_prefix}_{box_order}", dtype=np.float32)
    graph.nodes.append(
        gs.Node("Concat", inputs=[y1, x1, y2, x2], outputs=[reordered], attrs={"axis": 2})
    )
    return reordered


def _xywh_to_corners(
    graph: gs.Graph,
    boxes_xywh: gs.Tensor,
    box_order: str,
    name_prefix: str = "boxes",
) -> gs.Tensor:
    """
    Convert [B, N, 4] boxes from xywh(center) -> corner boxes.
    """
    # Slices along last dim
    x = _make_slice(graph, boxes_xywh, f"{name_prefix}_x", [0], [1], [2])
    y = _make_slice(graph, boxes_xywh, f"{name_prefix}_y", [1], [2], [2])
    w = _make_slice(graph, boxes_xywh, f"{name_prefix}_w", [2], [3], [2])
    h = _make_slice(graph, boxes_xywh, f"{name_prefix}_h", [3], [4], [2])

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
    return _reorder_boxes(graph, boxes_xyxy, box_order, name_prefix)


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
    box_order: str,
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
    boxes_raw = _make_slice(graph, out_t, "boxes_raw", [0], [4], [2])
    scores_raw = _make_slice(graph, out_t, "scores_raw", [4], [4 + num_classes], [2])

    # Optional sigmoid for scores
    if apply_sigmoid:
        scores = gs.Variable("scores_sigmoid", dtype=np.float32)
        graph.nodes.append(gs.Node("Sigmoid", inputs=[scores_raw], outputs=[scores]))
        score_activation_attr = 0  # already applied
    else:
        scores = scores_raw
        score_activation_attr = 0  # don't ask plugin to apply activation

    # Optional xywh(center) -> corner conversion
    if boxes_are_xywh:
        boxes = _xywh_to_corners(graph, boxes_raw, box_order, "boxes")
    else:
        boxes = _reorder_boxes(graph, boxes_raw, box_order, "boxes")

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
            "box_coding": 0,  # corner boxes
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
    parser.add_argument("--max-output-boxes", type=int, default=300, help="Max detections per image (default: 100)")
    parser.add_argument("--score-threshold", type=float, default=0.01, help="Score threshold (default: 0.25)")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold (default: 0.45)")

    parser.add_argument("--apply-sigmoid", action="store_true",
                        help="Apply sigmoid to class scores before NMS (use if outputs are logits).")
    parser.add_argument("--boxes-are-xywh", action="store_true",
                        help="Convert boxes from xywh(center) to corner boxes before NMS.")
    parser.add_argument(
        "--box-order",
        choices=["xyxy", "yxyx"],
        default="yxyx",
        help="Corner-box order to feed into EfficientNMS and expose on outputs (default: yxyx).",
    )
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
        box_order=args.box_order,
        class_agnostic=args.class_agnostic,
        keep_old_outputs=args.keep_old_outputs,
    )

    print("Done.")
    print(" Input :", args.model_in)
    print(" Output:", args.model_out)
    print(" If TensorRT complains about plugin not found, ensure you build with TensorRT that includes EfficientNMS_TRT.")


if __name__ == "__main__":
    main()
