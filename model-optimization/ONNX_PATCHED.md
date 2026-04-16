# Packed EfficientNMS Layout Spec (v1)

This document defines the **strict binary layout contract** for packed EfficientNMS
output produced by `model-optimization/inject_nms.py` when:

- `--nms-output-layout packed`
- `--nms-output-layout both`

Version: **1**  
Tensor name: **`det_packed`**  
DType: **FP32**  
Shape: **`[B, 1 + 6*K]`**

- `B` = batch size
- `K` = `max_output_boxes` (same value passed to EfficientNMS)

---

## Per-sample element order

For each batch item `b`, `det_packed[b, :]` is:

1. `num_dets` (1 element)
2. flattened `det_boxes` (4 * K elements)
3. `det_scores` (K elements)
4. `det_classes` (K elements; cast from INT32 to FP32 before packing)

Equivalent concatenation:

`[num_dets | boxes_flat | scores | classes]`

where:

- `boxes_flat` is flatten(`det_boxes[b, K, 4]`) in row-major order
- each box keeps plugin order (`--box-order` governs this upstream)

---

## Offset table (within one sample row)

- `num_dets`: start `0`, length `1`
- `boxes`: start `1`, length `4*K`
- `scores`: start `1 + 4*K`, length `K`
- `classes`: start `1 + 5*K`, length `K`

Total length:

`1 + 4*K + K + K = 1 + 6*K`

---

## Decoder requirements

Decoders should:

1. Assert dtype is FP32.
2. Assert row length is exactly `1 + 6*K`.
3. Read offsets exactly as specified above.
4. Treat `classes` values as integer class IDs encoded in FP32 (e.g. round or cast safely).
5. Gate behavior on layout version (`v1`) if future versions are introduced.

---