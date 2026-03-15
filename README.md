# 2D to 3D Side-by-Side Video Converter (GPU Efficiency Optimized)

![3D SBS Converter Example](https://github.com/PointerSoftware/2D-to-3D-SBS-Converter/blob/main/2D-3D-SBS.png)

A powerful Python application that converts standard 2D videos into stereoscopic 3D Side-by-Side (SBS) format for VR viewing. This converter uses deep learning (MiDaS) to generate accurate depth maps and create high-quality stereoscopic 3D videos optimized for various VR headsets.

## Changes From Original [Repository](https://github.com/PointerSoftware/2D-to-3D-SBS-Converter)

- **Optimized pipeline**: Move Video Encode Process To GPU, Reduced latency by Over 50%.
- **Stand Alone Python Version**
- **No Longer Supports CPU-Only Converting**: The Program Cannot Run Without CUDA.
- **Local Friendly**: It Will No Longer Download MiDaS Model Everytime You Start, Only The Very First Time.
- **Adjust Output Canvas To 4K**: It Now Outputs 16:9 4K half SBS(1920x2160 per eye) Video now.

## Features

- **Deep Learning-Based Depth Estimation**: Uses the MiDaS neural network for accurate depth map generation
- **GPU Optimization**: Maximizes GPU utilization for faster processing with batch frame handling
- **Video Segment Processing**: Select specific portions of longer videos to convert
- **Adjustable 3D Parameters**:
  - Depth intensity control (0.0-1.0)
  - Convergence distance adjustment (1.0-10.0) 
  - Eye separation control (0.1-5.0)
- **Multiple Input Sources**:
  - Upload local video files (up to 500MB)
  - Process videos from URLs including YouTube
- **Format Support**: 
  - Input: MP4, AVI, MOV, WebM, MKV (up to 4K resolution)
  - Output: H.264 encoded MP4 in SBS format (1920x1080)
- **Real-time Preview**: Visualize parameter adjustments before processing
- **Clean User Interface**: Intuitive Gradio-based UI with tab-based workflow
- **Comprehensive Progress Tracking**: Detailed status updates during conversion

## Demo
- GUI preview and loading bar might not be working proberly in colab.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MaleWalk/VR_SBS_Video_Converter/blob/main/2D_to_3D_SBS_Converter_Colab.ipynb)

## How It Works

The converter uses a multi-stage pipeline to transform 2D videos into 3D SBS format:

1. **Frame Extraction**: Extracts frames from the source video
2. **Depth Estimation**: Processes each frame through the MiDaS neural network to generate depth maps
3. **Stereoscopic Synthesis**: Creates left and right eye views by applying a displacement algorithm based on the depth maps
4. **View Combination**: Combines the stereoscopic views into a side-by-side format
5. **Video Encoding**: Encodes the final frames into an H.264 MP4 video

## Installation

### Google Colab (Recommended)

The easiest way to run this converter is through Google Colab, which provides free GPU resources:

1. Open the [Colab Notebook](https://colab.research.google.com/github/PointerSoftware/2D-to-3D-SBS-Converter/blob/main/2D_to_3D_SBS_Converter.ipynb)
2. Select "Runtime" → "Change runtime type" → Set "Hardware accelerator" to "GPU"
3. Run all cells

### Local Installation

To run the converter locally (requires a CUDA-capable NVIDIA GPU):

1. Clone this repository:
   ```
   git clone https://github.com/PointerSoftware/2D-to-3D-SBS-Converter.git
   cd 2D-to-3D-SBS-Converter
   ```

2. Set up a Python virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the notebook:
   ```
   jupyter notebook 2D_to_3D_SBS_Converter.ipynb
   ```

## Usage Guide

### Basic Usage

1. Upload a video file or provide a URL to a video (supports YouTube)
2. Adjust the 3D effect parameters:
   - **Depth Intensity**: Controls the strength of the 3D effect (higher values create more pronounced depth)
   - **Convergence Distance**: Adjusts where objects appear relative to the screen plane
   - **Eye Separation**: Controls the virtual camera separation (higher values create more extreme 3D effects)
3. Click "Update Preview" to see how your settings affect the 3D output
4. Click "Process Video" to convert the entire video to 3D SBS format
5. Download the converted video for viewing in a VR headset or 3D display

### Video Segmentation

For longer videos, you can process specific segments:

1. Check the "Process a specific segment" box
2. Set the start and end times in seconds
3. Process only the selected portion of the video

This is useful for:
- Testing 3D settings on a clip before processing the entire video
- Processing very long videos in manageable chunks
- Creating 3D highlights from specific parts of longer videos

## Viewing the 3D Videos

The output video is in Side-by-Side (SBS) format, which can be viewed in:

- VR headsets using video players that support SBS format
- 3D TVs with SBS viewing mode
- Special 3D viewers like Google Cardboard with SBS-compatible apps

## Technical Details

- **Depth Map Generation**: Uses MiDaS DPT_Large model for depth estimation
- **Output Format**: 1920x1080 (16:9) with 960x720 (4:3) content for each eye, embedded with black bars
- **GPU Acceleration**: Optimized for NVIDIA GPUs with dynamic batch sizing based on available memory
- **Memory Management**: Periodic GPU memory cleanup to handle larger videos
- **Video Processing**: FFmpeg-based segment extraction and final encoding

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- OpenCV, NumPy, Gradio
- FFmpeg
- NVIDIA GPU with CUDA support (for GPU acceleration)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [MiDaS](https://github.com/isl-org/MiDaS) for the depth estimation model
- [Gradio](https://gradio.app/) for the user interface framework
- [FFmpeg](https://ffmpeg.org/) for video processing capabilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
