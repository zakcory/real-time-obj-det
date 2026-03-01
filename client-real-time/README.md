# Real Time Client
The following folder contains a real time AI analytics application that is build on top of NVIDIA Triton Inference Server.<br>
The application receives frames from a live video stream, and sends live AI analytics back to the source in extremely slim time constraints.<br>

## Architecture
An overview of the architecture is shown below:<br>
<img src="../assets/client-triton.png" alt="Architecture" width="700"/>

The architecture of the application consists of the following components:
- **Video Client**: A dynamic library that is loaded within the application, and receives live stream video from a dedicated distributor (see `video-player` folder). The video frames are then sent to the main application for processing, and the analytics results are sent back to the distributor.
- **Main Application**: The main Rust application that handles video frame processing, It receives video frames from the video client, and sends them to the Triton Server for inference. Once the inference results are received, it processes them and sends the analytics back to the video client and third party services (e.g. Kafka).
- **Triton Inference Server**: An NVIDIA Triton Inference Server that hosts the optimized Deep Learning models (Object detection, Embedding) (see `model-optimization` folder). It receives video frames from the main application, performs inference, and sends back the results.

## Getting Started
To get started with running the application, you must have the following prerequisites:
- A machine with a **compatible NVIDIA GPU** and the necessary drivers installed.
- **Docker** and Docker Compose installed on your machine.
- **Rust** toolchain installed for building the application.

Run the following command to start the application locally:
```bash
./run_local.sh
```