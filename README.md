# Real Time AI Analytics
The following folder contains a real time AI analytics application that is build on top of NVIDIA Triton Inference Server.<br>
The application receives frames from a live video stream, and sends live AI analytics back to the source in extremely slim time constraints.<br>

## Architecture
An overview of the architecture is shown below:<br>
<img src="assets/client-triton.png" alt="Architecture" width="700"/>

## Getting Started
To get started with running the application, you must have the following prerequisites:
- A machine with a **compatible NVIDIA GPU** and the necessary drivers installed.
- **Docker** and Docker Compose installed on your machine.
- **Rust** toolchain installed for building the application.

Run the following command to start the application locally:
```bash
./client/run_local.sh
```