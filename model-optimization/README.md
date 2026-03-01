## Model Conversion
The following folder contains scripts that involve converting the raw `.pt` frame you got(whether its self made or pre-made - e.g. YOLOV9), to a format that is the most performant in a production system. We essentially want to get the fastest inference time, with minimal accuracy loss.

We first need to convert our model from `.pt` to `.onnx`, which is a universal format for machine learning models.
The conversion will result with model_**fp16** and model_**fp32** files, which represent the same model with different datatype formats. <br>
**FP16** Models are essentially equal in accuracy(for inference) and are much lighter and faster to run, and it is recommanded to use. 

Next, we would be converting the `.onnx` file we have to `.engine`, using NVIDIA's TensorRT tool.<br>
TensorRT compiles a model for your specific achitecture(one used at time of compilation), therefore making it very efficient when running on your machine.<br>

## Setting up the environment
```bash
uv sync
```

## Pytorch to Onnx Conversion
The following command is used for converting a model from Pytorch to Onnx:
```bash
# YOLOV9
uv run export_model.py \
  --model_type YOLOV9 \
  --model_path MODEL.pt \
  --model-source-code ./yolov9/

# DINOV3
uv run export_model.py \
  --model_type DINOV3 \
  --model_path MODEL.pt \
  --model-source-code ./dinov3/
  --dino-type dinov3_vitb16
```

## Onnx to TensorRT Conversion
The following command is used for converting a model (with support of batch inference).<br>
We would be doing the TensorRT conversion from within the docker image of Triton Server, to ensure its compatibility with the compiled model:
```bash
# YOLOV9
/usr/src/tensorrt/bin/trtexec \
    --onnx=MODEL.onnx \
    --saveEngine=CONVERTED.engine \
    --optShapes=images:8x3x640x640 \
    --minShapes=images:1x3x640x640 \
    --maxShapes=images:16x3x640x640 \
    --shapes=images:1x3x640x640 \
    --fp16  # omit this flag for FP32

# DinoV3
/usr/src/tensorrt/bin/trtexec \
    --onnx=MODEL.onnx \
    --saveEngine=CONVERTED.engine \
    --optShapes=images:8x3x224x224 \
    --minShapes=images:1x3x224x224 \
    --maxShapes=images:16x3x224x224 \
    --shapes=images:1x3x224x224 \
    --fp16  # omit this flag for FP32
```

## Extras - Get model best latency/throughput
Using a third party tool, `perf_analyzer`, we iterate over different batch sizes for one model instance. We find the sweet spot of when the model is giving the best latency for the max amount of batch size.
```bash
for b in 1 2 4 8 16 32; do
  perf_analyzer \
    -m <model_name> \
    -b $b \
    --concurrency-range 1:1 \
    --collect-metrics \
    --verbose-csv \
    -f results_batch_${b}.csv
done
```

To test performance of TRT model, use the following command:
```bash
/usr/src/tensorrt/bin/trtexec \
  --loadEngine=CONVERTED.engine \
  --shapes=images:8x3x640x640 \
  --exportTimes=inference_times.json

Dinov3 export:
```bash
# For TritonServer from 24.12+
/usr/src/tensorrt/bin/trtexec \
    --onnx=/yves/dinov3_vitb16-fp32-512.onnx \
    --saveEngine=/yves/dinov3_512.engine \
    --optShapes=images:8x3x512x512 \
    --minShapes=images:1x3x512x512 \
    --maxShapes=images:16x3x512x512 \
    --inputIOFormats=fp16:chw \
    --outputIOFormats=fp16:chw \
    --fp16 \
    --precisionConstraints=obey \
    --layerPrecisions='.*Softmax.*':fp32,'.*LayerNorm.*':fp32,'.*rope_embed.*':fp32 

# For TritonServer up to 24.12
/usr/src/tensorrt/bin/trtexec     
    --onnx=/yves/dinov3_vitb16-fp32-512.onnx     
    --saveEngine=/yves/dinov3_512_perfect.engine     
    --optShapes=images:8x3x512x512     
    --minShapes=images:1x3x512x512     
    --maxShapes=images:16x3x512x512     
    --inputIOFormats=fp16:chw     
    --outputIOFormats=fp16:chw     
    --fp16     
    --noTF32     
    --precisionConstraints=obey     
    --tacticSources=+cuBLAS,+cuBLAS_LT,+cuDNN     
    --layerPrecisions='.*Softmax.*':fp32,'.*LayerNorm.*':fp32,'.*rope_embed.*':fp32,'.*attn.*':fp16,'.*mlp.*':fp16,'.*patch_embed.*':fp16,'*':fp32
```