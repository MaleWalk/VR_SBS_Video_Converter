import os
import cv2
import numpy as np
import torch
import urllib.request
import gradio as gr
import tempfile
import time
import re
import shutil
import threading
import gc
from pathlib import Path
from tqdm.notebook import tqdm
import subprocess
import webbrowser    
#from google.colab import files

# Check if CUDA is available
if torch.cuda.is_available():
    print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Memory: {gpu_mem:.2f} GB")
    # Set CUDA device to GPU 0
    torch.cuda.set_device(0)
else:
    print("CUDA is not available. Using CPU.")

# Set default tensor type to cuda if available
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

# Optional: Set environment variable for PyTorch memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'

# Audio processing functions
def check_audio_stream(file_path):
    """Check if the video file has an audio stream"""
    try:
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
             '-show_entries', 'stream=codec_name', '-of', 'default=noprint_wrappers=1:nokey=1',
             file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # If there's output, an audio stream was found
        return bool(result.stdout.strip())
    except Exception as e:
        print(f"Error checking audio stream: {str(e)}")
        return False

def extract_audio(input_path, output_path):
    """Extract audio from a video file using ffmpeg"""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Use ffmpeg to extract audio
        cmd = [
            'ffmpeg',
            '-i', input_path,        # Input file
            '-vn',                   # Disable video
            '-acodec', 'copy',       # Copy audio codec without re-encoding
            '-y',                    # Overwrite output file if it exists
            output_path
        ]
        
        print(f"Extracting audio from {input_path} to {output_path}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error extracting audio: {result.stderr}")
            return None
        
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            print("Audio extraction failed - output file is empty or missing")
            return None
        
        print("Audio extracted successfully")
        return output_path
    except Exception as e:
        print(f"Error during audio extraction: {str(e)}")
        return None

def combine_video_audio(video_path, audio_path, output_path):
    """Combine video and audio files using ffmpeg"""
    try:
        # Use ffmpeg to merge video and audio
        cmd = [
            'ffmpeg',
            '-i', video_path,        # Video file
            '-i', audio_path,        # Audio file
            '-c:v', 'copy',          # Copy video without re-encoding
            '-c:a', 'aac',           # Use AAC for audio (better compatibility)
            '-b:a', '192k',          # Audio bitrate
            '-shortest',             # Match the duration of the shorter file
            '-y',                    # Overwrite output file if it exists
            output_path
        ]
        
        print(f"Combining video and audio into {output_path}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error combining video and audio: {result.stderr}")
            return False
        
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            print("Combination failed - output file is empty or missing")
            return False
        
        print("Video and audio combined successfully")
        return True
    except Exception as e:
        print(f"Error during combination: {str(e)}")
        return False

def setup_midas():
    """Initialize and return the MiDaS model for depth estimation using torch.hub
    with optimizations for GPU usage"""
    print("Loading MiDaS depth estimation model...")
    
    # Clean up any existing GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        # Print GPU info
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
        print(f"Memory Reserved: {torch.cuda.memory_reserved(0)/1e9:.2f} GB")
    # 1. 設定固定的本地快取路徑
    # 這會在你的程式資料夾下建立 'midas_cache' 檔案夾
    local_hub_dir = os.path.join(os.getcwd(), "midas_cache")
    os.makedirs(local_hub_dir, exist_ok=True)
    torch.hub.set_dir(local_hub_dir)
    # 3. 載入模型 (優先使用本地快取)
    # model_type 可選: "DPT_Large" (1.28GB), "DPT_Hybrid" (470MB), "MiDaS_small" (80MB)
    model_type = "DPT_Large" 
    print(f"正在從 {local_hub_dir} 載入模型 {model_type}...")
    
    try:
        # force_reload=False 確保如果本地已有檔案就不會重新下載
        midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True, force_reload=False)
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    except Exception as e:
        print(f"載入過程中發生錯誤: {e}，嘗試重新連網確認...")
        midas = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    
    midas.to(device)
    midas.eval()  # Set to evaluation mode
    
    # If using CUDA, optimize model for inference
    if device.type == 'cuda':
        midas = midas.half()
        # Enable cuDNN benchmark mode for best performance with fixed input sizes
        torch.backends.cudnn.benchmark = True
        
        # We'll skip TorchScript optimization as it's causing issues
        print("Skipping TorchScript optimization due to compatibility issues")
    
    # Load transforms
    try:
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    except Exception as e:
        print(f"Error loading transforms: {e}")
        # Fallback
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        
    transform = midas_transforms.dpt_transform
    
    # Report GPU memory usage after model loading
    if torch.cuda.is_available():
        print(f"GPU Memory After Model Load: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
    
    print("MiDaS model loaded successfully!")
    return midas, transform, device

def validate_video(file_path):
    """Validate if the input video file is supported"""
    # Check if file exists
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    # Check file extension
    valid_extensions = [".mp4", ".avi", ".mov", ".webm", ".mkv"]
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext not in valid_extensions:
        return False, f"Unsupported file format: {file_ext}. Supported formats: {', '.join(valid_extensions)}"
    
    # Check if OpenCV can open the file
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return False, "Cannot open video file with OpenCV"
    
    # Check resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width > 3840 or height > 2160:
        cap.release()
        return False, f"Video resolution ({width}x{height}) exceeds maximum supported resolution (3840x2160)"
    
    # Check file size (500MB limit)
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if file_size_mb > 1024:
        cap.release()
        return False, f"File size ({file_size_mb:.2f}MB) exceeds maximum supported size (500MB)"
    
    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if fps > 0 else 0
    cap.release()
    
    return True, {"width": width, "height": height, "fps": fps, "frame_count": frame_count, 
                  "size_mb": file_size_mb, "duration_sec": duration_sec}

def extract_video_segment(input_path, output_path, start_time, end_time):
    """Extract a segment from a video file using ffmpeg"""
    try:
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"Extracting segment from {start_time:.2f}s to {end_time:.2f}s...")
        
        # Use ffmpeg to extract the segment with stream copy
        cmd = [
            'ffmpeg',
            '-i', input_path,        # Input file
            '-ss', str(start_time),  # Start time in seconds
            '-to', str(end_time),    # End time in seconds
            '-c:v', 'copy',          # Copy video stream without re-encoding
            '-c:a', 'copy',          # Copy audio stream without re-encoding
            '-avoid_negative_ts', '1',  # Avoid negative timestamps
            '-y',                    # Overwrite output file if it exists
            output_path
        ]
        
        # Run the command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check if the command was successful
        if result.returncode != 0:
            print(f"Error extracting segment: {result.stderr}")
            raise Exception(f"ffmpeg error: {result.stderr}")
        
        # Verify the output file exists and has content
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise ValueError("Segment extraction failed - output file is empty or missing")
        
        print(f"Segment extracted successfully: {output_path}")
        return output_path
            
    except Exception as e:
        print(f"Error extracting video segment: {str(e)}")
        raise

def ensure_h264_mp4(input_path, temp_dir="temp_videos"):
    """Convert video to H.264 MP4 format if needed - optimized for speed"""
    # Generate a new filename for the converted video
    output_path = os.path.join(temp_dir, f"h264_{int(time.time())}.mp4")
    
    # Use ffprobe to check if the video is already H.264 encoded
    try:
        print(f"Checking encoding of {input_path}...")
        # Get video codec information with a short timeout
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=codec_name', '-of', 'default=noprint_wrappers=1:nokey=1',
             input_path],
            capture_output=True, text=True, check=True, timeout=10
        )
        codec = result.stdout.strip()
        
        if codec.lower() in ['h264', 'avc1']:
            print(f"Video is already H.264 encoded (codec: {codec})")
            return input_path
        else:
            print(f"Video is not H.264 encoded (detected codec: {codec}). Converting with fast settings...")
    except subprocess.TimeoutExpired:
        print("Codec detection timed out. Proceeding with conversion...")
    except Exception as e:
        print(f"Error checking video codec: {str(e)}. Converting with fast settings...")
    
    # Check if input file exists and has content
    if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
        raise ValueError(f"Input file {input_path} does not exist or is empty")
    
    # Convert to H.264 MP4 with hardware acceleration if available
    try:
        print("Starting fast H.264 conversion...")
        
        # Try using hardware acceleration if available
        # NVIDIA GPU acceleration
        hw_accel_commands = [
            # NVIDIA NVENC (if available)
            [
                'ffmpeg',
                '-i', input_path,
                '-c:v', 'h264_nvenc',  # NVIDIA GPU acceleration
                '-preset', 'p1',  # Fast encoding preset
                '-tune', 'hq',  # High quality tuning
                '-rc:v', 'vbr',  # Variable bitrate
                '-cq:v', '23',  # Quality level
                '-b:v', '5M',  # Target bitrate
                '-maxrate:v', '10M',  # Maximum bitrate
                '-bufsize:v', '10M',  # Buffer size
                '-c:a', 'aac',  # Audio codec
                '-b:a', '128k',  # Audio bitrate
                '-y',  # Overwrite output if exists
                output_path
            ],
            # Fallback to CPU with ultrafast preset
            [
                'ffmpeg',
                '-i', input_path,
                '-c:v', 'libx264',  # CPU encoding
                '-preset', 'ultrafast',  # Fastest encoding
                '-tune', 'fastdecode',  # Fast decoding optimization
                '-crf', '28',  # Lower quality for speed
                '-g', '30',  # Keyframe every 30 frames
                '-bf', '0',  # No B-frames (faster)
                '-c:a', 'aac',  # Audio codec
                '-b:a', '128k',  # Low audio bitrate
                '-ac', '2',  # Stereo audio
                '-ar', '44100',  # Standard audio sample rate
                '-strict', 'experimental',
                '-y',  # Overwrite output
                output_path
            ]
        ]
        
        # Try each acceleration method in order
        success = False
        for i, command in enumerate(hw_accel_commands):
            try:
                print(f"Trying encoding method {i+1}...")
                
                # Run the command
                print(f"Running conversion command: {' '.join(command)}")
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                
                # Set timeout for conversion (5 minutes)
                timeout = 300  # seconds
                start_time = time.time()
                
                # Monitor progress
                while process.poll() is None:
                    # Check if timeout has been reached
                    if time.time() - start_time > timeout:
                        process.terminate()
                        raise TimeoutError(f"Conversion timed out after {timeout} seconds")
                    
                    # Print progress indicator
                    print(".", end="", flush=True)
                    time.sleep(1)
                
                # Check if successful
                if process.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    print(f"\nSuccessfully converted to H.264 MP4 using method {i+1}: {output_path}")
                    success = True
                    break
                else:
                    print(f"\nMethod {i+1} failed with error code {process.returncode}")
            except Exception as e:
                print(f"Error with method {i+1}: {str(e)}")
        
        if success:
            return output_path
        else:
            # Fallback to simple copy method (no re-encoding)
            try:
                print("Attempting direct copy method as fallback...")
                subprocess.run([
                    'ffmpeg',
                    '-i', input_path,
                    '-c', 'copy',  # Just copy streams without re-encoding
                    '-y',
                    output_path
                ], check=True, timeout=300)
                
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    print(f"Successfully copied video to MP4 container: {output_path}")
                    return output_path
                else:
                    print("Copy method failed to produce a valid output file")
                    # If all conversion methods fail, return the original file path
                    return input_path
            except Exception as e2:
                print(f"All conversion methods failed: {str(e2)}")
                # If all conversion methods fail, return the original file path
                return input_path
            
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        # Return the original file path if all else fails
        return input_path

def get_video_duration(file_path):
    """Get the duration of a video file in seconds using ffprobe"""
    try:
        # Use ffprobe to get the duration
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=noprint_wrappers=1:nokey=1', file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=True
        )
        duration = float(result.stdout.strip())
        return duration
    except Exception as e:
        print(f"Error getting video duration: {str(e)}")
        # Fall back to OpenCV
        try:
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            return duration
        except Exception as e2:
            print(f"Error getting duration with OpenCV: {str(e2)}")
            return 0

def download_from_url(url):
    """Download video from URL and return local file path"""
    # Create temp directory if it doesn't exist
    temp_dir = "temp_videos"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate a temporary filename without extension
    timestamp = int(time.time())
    file_base = os.path.join(temp_dir, f"downloaded_video_{timestamp}")
    temp_file = f"{file_base}.mp4"
    
    # Check if it's a YouTube URL or regular URL
    if "youtube.com" in url or "youtu.be" in url:
        """try:
            print(f"Attempting to download YouTube video from: {url}")
            
            # Clean up YouTube URL - remove playlist parameters
            if "?list=" in url:
                url = url.split("?list=")[0]
            elif "&list=" in url:
                url = url.split("&list=")[0]
                
            # Remove any additional query parameters
            if "?si=" in url:
                url = url.split("?si=")[0]
            elif "&si=" in url:
                url = url.split("&si=")[0]
                
            print(f"Using cleaned URL: {url}")
            
            # Skip pytube and use yt-dlp directly with format 22 (usually 720p MP4)
            # This format tends to be more reliable with less need for conversion
            print("Downloading with yt-dlp using format 22 (720p MP4)...")
            try:
                # Ensure yt-dlp is updated
                !pip install -q --upgrade yt-dlp
                
                # Use format 22 which is typically 720p MP4 (most compatible)
                !yt-dlp -f 22 -o "{temp_file}" "{url}"
                
                if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                    print(f"Successfully downloaded video with format 22 to {temp_file}")
                    # This format should be H.264 compatible, but verify to be sure
                    return ensure_h264_mp4(temp_file, temp_dir)
                else:
                    # Try method 2 with best format
                    raise Exception("Format 22 download failed. Trying best format...")
                    
            except Exception as e:
                print(f"First download method failed: {str(e)}. Trying best format...")
                
                # Method 2: Use 'best' format
                try:
                    print("Downloading with yt-dlp using 'best' format...")
                    !yt-dlp -f "best" -o "{temp_file}" "{url}"
                    
                    if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                        print(f"Successfully downloaded video with 'best' format to {temp_file}")
                        return ensure_h264_mp4(temp_file, temp_dir)
                    else:
                        # Try method 3 with direct download
                        raise Exception("'Best' format download failed. Trying direct download...")
                        
                except Exception as e2:
                    print(f"Second download method failed: {str(e2)}. Trying direct link...")
                    
                    # Try using yt-dlp to get direct URL then download with urllib
                    try:
                        print("Getting direct video URL from YouTube...")
                        import json
                        # Get the info as JSON and extract direct URL
                        info_cmd = f"yt-dlp -f 18 -j {url}"
                        result = !{info_cmd}
                        if result:
                            info = json.loads(result[0])
                            direct_url = info.get('url')
                            if direct_url:
                                print(f"Got direct URL, downloading with urllib...")
                                urllib.request.urlretrieve(direct_url, temp_file)
                                if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                                    print(f"Successfully downloaded with direct URL to {temp_file}")
                                    return ensure_h264_mp4(temp_file, temp_dir)
                        
                        raise Exception("Could not get direct URL from YouTube")
                    except Exception as e3:
                        print(f"All download methods failed: {str(e3)}")
                        raise Exception("Could not download video from YouTube after multiple attempts")
                
        except Exception as e:
            raise Exception(f"YouTube download failed: {str(e)}")"""
    else:
        # Download regular URL
        try:
            print(f"Downloading from direct URL: {url}")
            urllib.request.urlretrieve(url, temp_file)
            print(f"Successfully downloaded to {temp_file}")
            
            # Ensure it's in H.264 format
            return ensure_h264_mp4(temp_file, temp_dir)
        except Exception as e:
            raise Exception(f"Error downloading video: {str(e)}")

# Function to estimate depth for a single frame
def estimate_depth(frame, model, transform, device):
    """Estimate depth for a single frame using MiDaS (GPU 輸出版)"""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)

    if device.type == 'cuda':
        input_batch = input_batch.half()  # <--- 加入這行，確保輸入和模型都是 Half
    
    with torch.no_grad():
        prediction = model(input_batch) 
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ) # 此時形狀為 (1, 1, H, W)
    
    # --- 核心改動：直接在 GPU 上做正規化，不轉 NumPy ---
    depth_min = prediction.min()
    depth_max = prediction.max()
    if depth_max - depth_min > 0:
        depth_tensor = (prediction - depth_min) / (depth_max - depth_min)
    else:
        depth_tensor = torch.zeros_like(prediction)
    
    return depth_tensor # 回傳的是 GPU 上的 Tensor
"""
def estimate_depth(frame, model, transform, device):
    #Estimate depth for a single frame using MiDaS
    # Preprocess image for MiDaS (using torch.hub transforms)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)
    
    # Compute depth prediction
    with torch.no_grad():
        prediction = model(input_batch) 
        # Resize prediction to original frame size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth = prediction.cpu().numpy()
    
    # Normalize depth map to 0-1 range
    depth_min = depth.min()
    depth_max = depth.max()
    if depth_max - depth_min > 0:
        depth = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth = np.zeros(depth.shape, dtype=depth.dtype)
    
    return depth
    """

    
def process_batch(frames, model, transform, device):
    """Process a batch of frames to get depth maps"""
    depth_maps = []
    
    # Process each frame in the batch separately
    # This is more compatible than trying to batch process
    for frame in frames:
        depth_map = estimate_depth(frame, model, transform, device)
        depth_maps.append(depth_map)
    
    return depth_maps

def create_depth_based_disparity(depth_map, depth_intensity, convergence, eye_separation):
    """Create disparity map from depth map using the control parameters"""
    # Invert depth map since closer objects should have larger disparity
    inverted_depth = 1.0 - depth_map
    
    # Apply intensity control
    disparity = inverted_depth * depth_intensity
    
    # Apply eye separation and convergence adjustment
    disparity = disparity * eye_separation / convergence
    
    return disparity



def generate_stereo_views(frame_tensor, depth_tensor, depth_intensity, convergence, eye_separation):
    """
    全 GPU 加速的視角合成
    frame_tensor: (1, 3, H, W) 位於 GPU 的影像 Tensor
    depth_tensor: (1, 1, H, W) 位於 GPU 的深度 Tensor
    """
    device = frame_tensor.device
    b, c, h, w = frame_tensor.shape
    
    # 1. 建立標準座標系 [-1, 1]
# 1. 建立標準座標系 [-1, 1] (明確指定 dtype，與輸入影像保持一致)
    dtype = frame_tensor.dtype
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device, dtype=dtype),
        torch.linspace(-1, 1, w, device=device, dtype=dtype),
        indexing='ij'
    )
    grid_x = grid_x.unsqueeze(0).expand(b, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(b, -1, -1)
    
    # 2. 計算位移 (Disparity)
    # 移除多餘維度，對齊形狀
    depth_squeeze = depth_tensor.squeeze(1) # 變成 (B, H, W)
    disparity = (1.0 - depth_squeeze) * depth_intensity * (eye_separation / convergence)
    max_shift = 0.05
    shift = disparity * max_shift
    
    # 3. 建立映射 Grid (對齊 cv2.BORDER_REPLICATE)
    left_grid = torch.stack([grid_x + shift, grid_y], dim=-1)
    right_grid = torch.stack([grid_x - shift, grid_y], dim=-1)
    
    # 4. GPU 硬體插值
    left_views = torch.nn.functional.grid_sample(
        frame_tensor, left_grid, mode='bilinear', padding_mode='border', align_corners=True
    )
    right_views = torch.nn.functional.grid_sample(
        frame_tensor, right_grid, mode='bilinear', padding_mode='border', align_corners=True
    )
    
    return left_views, right_views
    """
    向量化優化版：使用 cv2.remap 代替 Python 迴圈
    
    h, w = frame.shape[:2]
    
    # 1. 向量化計算位移量 (Disparity)
    # 直接對整個矩陣運算，不需要迴圈
    disparity = (1.0 - depth_map) * depth_intensity * (eye_separation / convergence)
    
    # 縮放位移量到像素單位 (最大寬度的 5%)
    max_shift = w * 0.05
    disparity_scaled = (disparity * max_shift).astype(np.float32)
    
    # 2. 建立坐標映射表 (Meshgrid)
    # 建立基礎的 x, y 坐標矩陣
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    x_coords = x_coords.astype(np.float32)
    y_coords = y_coords.astype(np.float32)
    
    # 3. 計算左右眼的映射坐標
    # 左眼視角：像素向右偏移 (讀取原本左側的像素)
    map_l_x = x_coords + disparity_scaled / 2
    # 右眼視角：像素向左偏移 (讀取原本右側的像素)
    map_r_x = x_coords - disparity_scaled / 2
    
    # 4. 使用 OpenCV 內建的高性能 remap 進行圖像重組
    # INTER_LINEAR 提供平滑的插值，BORDER_REPLICATE 則能自動處理邊緣空洞
    left_view = cv2.remap(frame, map_l_x, y_coords, 
                          interpolation=cv2.INTER_LINEAR, 
                          borderMode=cv2.BORDER_REPLICATE)
    
    right_view = cv2.remap(frame, map_r_x, y_coords, 
                           interpolation=cv2.INTER_LINEAR, 
                           borderMode=cv2.BORDER_REPLICATE)
    
    return left_view, right_view
    """

def create_side_by_side(left_view, right_view):
    """
    將左右眼視角合併為 1920x1080 滿版 Side-by-Side 格式 (無黑邊)。
    每個視角會被強制縮放至 960x1080。
    """
    # 目標單眼尺寸
    eye_w, eye_h = 1920, 2160
    
    # 1. 直接縮放至 960x1080，填滿所有空間
    # 使用 INTER_AREA 在縮小時品質較好，INTER_CUBIC 在放大時品質較好
    left_resized = cv2.resize(left_view, (eye_w, eye_h), interpolation=cv2.INTER_LINEAR)
    right_resized = cv2.resize(right_view, (eye_w, eye_h), interpolation=cv2.INTER_LINEAR)
    
    # 2. 使用 NumPy 水平拼接 (比手動 slice 畫布更快)
    sbs_frame = np.hstack((left_resized, right_resized))
    
    return sbs_frame


def process_video_to_3d_sbs(input_path, output_path, depth_intensity, convergence, eye_separation, 
                           progress=None, use_segment=False, segment_start=0, segment_end=None):
    """Convert a 2D video to 3D SBS using MiDaS depth estimation with GPU optimization"""

    try:
        # Validate input video
        valid, result = validate_video(input_path)
        if not valid:
            raise ValueError(result)
            
        video_info = result
        
        # Create temporary directory for intermediate files
        temp_dir = "temp_videos"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Base names for temp files
        timestamp = int(time.time())
        temp_base = os.path.join(temp_dir, f"temp_{timestamp}")
        temp_video_path = f"{temp_base}_video.mp4"  # For video without audio
        temp_audio_path = f"{temp_base}_audio.aac"  # For extracted audio
        
        # Track whether we're processing a segment
        is_segment = False
        original_input = input_path
        
        # Extract audio from the source video (original or segment)
        has_audio = check_audio_stream(input_path)
        if has_audio:
            print("Detected audio stream in the video")
            if extract_audio(input_path, temp_audio_path):
                print(f"Audio extracted to {temp_audio_path}")
            else:
                print("Could not extract audio. Output will have no sound.")
                has_audio = False
        else:
            print("No audio stream detected in the video")
            
        # If using a segment, extract it first
        segment_path = None
        if use_segment and segment_start is not None and segment_end is not None and segment_start < segment_end:
            try:
                # Create temporary segment file
                temp_dir = "temp_videos"
                os.makedirs(temp_dir, exist_ok=True)
                segment_path = os.path.join(temp_dir, f"segment_{int(time.time())}.mp4")
                
                # Extract the segment
                segment_path = extract_video_segment(input_path, segment_path, segment_start, segment_end)
                
                # Use the segment file for processing
                input_path = segment_path
                
                # Re-validate the segment
                valid, result = validate_video(input_path)
                if not valid:
                    raise ValueError(f"Segment validation failed: {result}")
                    
                video_info = result
                print(f"Using video segment from {segment_start}s to {segment_end}s")
                
            except Exception as e:
                print(f"Error extracting segment: {str(e)}. Processing entire video instead.")
                # Continue with the original file if segment extraction fails
        
        width, height = video_info["width"], video_info["height"]
        fps = video_info["fps"]
        frame_count = int(video_info["frame_count"])
        
        # Clear GPU memory before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Setup MiDaS model - note: only getting 3 return values now
        model, transform, device = setup_midas()
        
        
        # Determine optimal batch size based on available GPU memory and resolution
        batch_size = 1  # Default
        if torch.cuda.is_available():
            # Calculate available memory
            available_mem = torch.cuda.get_device_properties(0).total_memory
            current_mem = torch.cuda.memory_allocated()
            free_mem = available_mem - current_mem
            
            # Heuristic for batch size based on resolution
            pixel_count = width * height
            if pixel_count <= 640 * 480:  # SD video
                batch_size = 32
            elif pixel_count <= 1280 * 720:  # HD video
                batch_size = 16
            elif pixel_count <= 1920 * 1080:  # Full HD
                batch_size = 8
            else:  # 4K
                batch_size = 4
                
            print(f"Using batch size: {batch_size} for {width}x{height} video")
            print(f"Available GPU memory: {free_mem/1e9:.2f}GB")
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        
        # Create output video writer
        target_width, target_height = 3840, 2160  # 16*2:9 overall aspect ratio
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # H.264 codec
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (target_width, target_height))
        
        # Process frames
        frame_index = 0
        prev_depth_map = None
        
        # Report initial memory usage
        if torch.cuda.is_available():
            print(f"Initial GPU Memory: {torch.cuda.memory_allocated(0)/1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB")
        
        # Process video in batches
        while True:
            # Read batch of frames
            frames = []
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            
            if not frames:
                break  # End of video
            
            # 1. 取得深度圖 (這批深度圖現在是保留在 GPU 上的 Tensor)
            depth_maps = process_batch(frames, model, transform, device)
            
            # Process each frame with its depth map
            for i in range(len(frames)):
                frame = frames[i]
                depth_map = depth_maps[i]
                
                # Apply temporal smoothing (在 GPU 上直接進行 Tensor 運算)
                if prev_depth_map is not None:
                    depth_map = 0.8 * depth_map + 0.2 * prev_depth_map
                prev_depth_map = depth_map.clone() # 使用 clone() 避免報錯
                
                # 2. 將原始影像轉為 GPU Tensor (保留 0~255 的數值)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
                
                if device.type == 'cuda':
                    frame_tensor = frame_tensor.half()  # 確保輸入和模型都是 Half
                
                # 確保 depth_map 具有 (1, 1, H, W) 的維度以對接 grid_sample
                if depth_map.dim() == 2:
                    depth_input = depth_map.unsqueeze(0).unsqueeze(0)
                elif depth_map.dim() == 3:
                    depth_input = depth_map.unsqueeze(0)
                else:
                    depth_input = depth_map

                # 3. 呼叫 GPU 版合成 (這裡得到的 left_tensor, right_tensor 是 GPU 上的 Tensor)
                left_tensor, right_tensor = generate_stereo_views(
                    frame_tensor, depth_input, depth_intensity, convergence, eye_separation
                )

                # --- 【全新優化：在 GPU 上直接完成 4K 縮放與 SBS 拼接】 ---
                # target_size=(高度, 寬度)，單眼為 2160x1920
                left_resized = torch.nn.functional.interpolate(left_tensor, size=(2160, 1920), mode='bilinear', align_corners=False)
                right_resized = torch.nn.functional.interpolate(right_tensor, size=(2160, 1920), mode='bilinear', align_corners=False)
                
                # 在第 3 維度 (寬度維度) 進行拼接 (Concatenate)，變成 1 個 3840x2160 的 Tensor
                sbs_tensor = torch.cat((left_resized, right_resized), dim=3)

                # 4. 一次性將最終的 4K 畫面拿回 CPU 寫入
                # clamp(0, 255) 防溢位，.byte() 轉為 uint8
                sbs_np = sbs_tensor.squeeze(0).clamp(0, 255).permute(1, 2, 0).byte().cpu().numpy()
                
                # 轉回 BGR 給 OpenCV 寫入
                sbs_frame = cv2.cvtColor(sbs_np, cv2.COLOR_RGB2BGR)

                # Write frame to output (不需要再呼叫 create_side_by_side 了)
                out.write(sbs_frame)
                
                # Update progress
                frame_index += 1
                if progress is not None:
                    progress(min(1.0, frame_index / frame_count))
                
                # Report GPU memory periodically
                if frame_index % 100 == 0 and torch.cuda.is_available():
                    memory_used_gb = torch.cuda.memory_allocated(0)/1e9
                    total_mem_gb = torch.cuda.get_device_properties(0).total_memory/1e9
                    usage_percent = (memory_used_gb / total_mem_gb) * 100
                    print(f"Frame {frame_index}/{frame_count} - GPU Memory: {memory_used_gb:.2f}GB / {total_mem_gb:.2f}GB ({usage_percent:.1f}%)")
            
        # Ensure 100% progress at the end
        if progress is not None:
            progress(1.0)
        
        # Release resources
        cap.release()
        out.release()
        
        print("Processing complete!")
        
        # Now combine the processed video with the original audio
        if has_audio:
            print("Combining video with original audio...")
            if combine_video_audio(temp_video_path, temp_audio_path, output_path):
                print("Successfully combined video with audio")
            else:
                print("Audio combination failed. Using high quality encoding for video-only output...")
                # Fall back to just processing the video without audio
                if torch.cuda.is_available():
                    subprocess.run([
                        'ffmpeg',
                        '-i', temp_video_path,
                        '-c:v', 'h264_nvenc',  # NVIDIA hardware encoding
                        '-preset', 'p1',       # Medium quality/speed
                        '-b:v', '40M',          # Bitrate
                        '-y',                  # Overwrite output if exists
                        output_path
                    ], check=True, timeout=600)
                else:
                    subprocess.run([
                        'ffmpeg',
                        '-i', temp_video_path,
                        '-c:v', 'libx264',     # CPU encoding
                        '-preset', 'medium',   # Medium quality/speed
                        '-crf', '23',          # Quality level
                        '-y',                  # Overwrite output if exists
                        output_path
                    ], check=True, timeout=600)
        else:
            # No audio to add, just convert the video
            print("No audio to add. Finalizing video with high quality encoding...")
            if torch.cuda.is_available():
                subprocess.run([
                    'ffmpeg',
                    '-i', temp_video_path,
                    '-c:v', 'h264_nvenc',  # NVIDIA hardware encoding
                    '-preset', 'p1',       # Medium quality/speed
                    '-b:v', '40M',          # Bitrate
                    '-y',                  # Overwrite output if exists
                    output_path
                ], check=True, timeout=600)
            else:
                subprocess.run([
                    'ffmpeg',
                    '-i', temp_video_path,
                    '-c:v', 'libx264',     # CPU encoding
                    '-preset', 'medium',   # Medium quality/speed
                    '-crf', '23',          # Quality level
                    '-y',                  # Overwrite output if exists
                    output_path
                ], check=True, timeout=600)
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Clean up segment file if we created one
        if segment_path and os.path.exists(segment_path) and segment_path != input_path:
            try:
                os.remove(segment_path)
                print(f"Cleaned up temporary segment file: {segment_path}")
            except Exception as e:
                print(f"Warning: Could not remove temporary segment file: {str(e)}")
        
        return output_path
        
    except Exception as e:
        print(f"Error in process_video_to_3d_sbs: {str(e)}")
        raise

def generate_preview_frame(input_path, depth_intensity, convergence, eye_separation, frame_position=0.5):
    """Generate a preview frame for the given parameters"""
    try:
        # Validate input video
        valid, result = validate_video(input_path)
        if not valid:
            raise ValueError(result)
        
        # Setup MiDaS model
        model, transform, device = setup_midas()
        
        # Open input video
        cap = cv2.VideoCapture(input_path)
        
        # Get frame count and set position
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frame = int(frame_count * frame_position)
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        
        # Read frame
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise ValueError("Could not read frame from video")
        
        # Estimate depth
        depth_map = estimate_depth(frame, model, transform, device)
                # Generate stereo views
        #left_view, right_view = generate_stereo_views(frame, depth_map, depth_intensity, convergence, eye_separation)

        ##########
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
        if device.type == 'cuda':
            frame_tensor = frame_tensor.half()  # <--- 加入這行，確保輸入和模型都是 Half
                # 2. 呼叫 GPU 版合成
        left_tensor, right_tensor = generate_stereo_views(
            frame_tensor, depth_map, depth_intensity, convergence, eye_separation
        )

        # 3. 將合成結果從 GPU 拿回 CPU，轉回 OpenCV 格式 (NumPy BGR)
        # clamp 確保數值在 0-255 安全範圍內
        left_view_np = left_tensor.squeeze(0).clamp(0, 255).permute(1, 2, 0).byte().cpu().float().numpy()
        right_view_np = right_tensor.squeeze(0).clamp(0, 255).permute(1, 2, 0).byte().cpu().float().numpy()

        left_view = cv2.cvtColor(left_view_np, cv2.COLOR_RGB2BGR)
        right_view = cv2.cvtColor(right_view_np, cv2.COLOR_RGB2BGR)
        ##########
        # Generate stereo views
        #left_view, right_view = generate_stereo_views(frame, depth_map, depth_intensity, convergence, eye_separation)
        
        # Create side-by-side frame
        sbs_frame_full = create_side_by_side(left_view, right_view)
        
        # Create preview (smaller version)
        preview_height = 360
        preview_width = int(1920 * (preview_height / 1080))
        sbs_frame = cv2.resize(sbs_frame_full, (preview_width, preview_height))
        
        # Create comparison view with original frame
        h, w = frame.shape[:2]
        original_resized = cv2.resize(frame, (int(w * preview_height / h), preview_height))
        
        # Create final preview
        preview_width_total = original_resized.shape[1] + sbs_frame.shape[1] + 10
        preview = np.zeros((preview_height, preview_width_total, 3), dtype=np.uint8)
        
        # Add original frame
        preview[:, :original_resized.shape[1]] = original_resized
        # Add SBS frame
        preview[:, original_resized.shape[1]+10:] = sbs_frame
        
        # Add labels
        cv2.putText(preview, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(preview, "3D SBS (16:9)", (original_resized.shape[1]+20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Release resources
        cap.release()
        
        # Convert to RGB for display
        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        return preview_rgb
        
    except Exception as e:
        print(f"Error in generate_preview_frame: {str(e)}")
        raise

def create_gradio_interface():
    """Create and launch the Gradio interface for 3D SBS conversion with audio support"""
    # Global variables for state management
    input_video_path = None
    output_video_path = None
    video_duration = 0  # Store video duration for segment selection
    
    def save_uploaded_file(file_obj):
        """Helper function to save an uploaded file to disk"""
        temp_dir = "temp_videos"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate a filename with timestamp to avoid conflicts
        timestamp = int(time.time())
        file_name = f"uploaded_video_{timestamp}.mp4"
        file_path = os.path.join(temp_dir, file_name)
        
        print(f"Saving uploaded file to {file_path}")
        
        try:
            # Handle different file object types based on Gradio version
            if isinstance(file_obj, str):
                # It's a file path string, just copy the file
                shutil.copy(file_obj, file_path)
            elif hasattr(file_obj, 'name') and os.path.exists(file_obj.name):
                # It's an object with a name attribute that points to a real file
                shutil.copy(file_obj.name, file_path)
            else:
                # Try multiple approaches based on different versions of Gradio
                if hasattr(file_obj, 'read') and callable(file_obj.read):
                    # It's a file-like object, read and write it
                    with open(file_path, 'wb') as f:
                        f.write(file_obj.read())
                elif hasattr(file_obj, '_path') and os.path.exists(file_obj._path):
                    # Some versions of Gradio use a _path attribute
                    shutil.copy(file_obj._path, file_path)
                else:
                    # Fall back to trying to directly access file object (may not work in all cases)
                    with open(file_path, 'wb') as f:
                        if isinstance(file_obj, bytes):
                            f.write(file_obj)
                        else:
                            f.write(str(file_obj).encode('utf-8'))
        except Exception as e:
            print(f"Error saving file: {str(e)}")
            raise e
            
        # Ensure the video is in H.264 format
        return ensure_h264_mp4(file_path, temp_dir)
    
    def upload_video(video_file):
        """Handle video upload"""
        nonlocal input_video_path, video_duration
        
        if video_file is None:
            return None, "Please upload a video file", gr.Slider(minimum=0, maximum=1), gr.Slider(minimum=0, maximum=1), False
        
        try:
            # Save the uploaded file to disk and ensure H.264 encoding
            input_video_path = save_uploaded_file(video_file)
            
            # Validate video
            valid, result = validate_video(input_video_path)
            if not valid:
                return None, result, gr.Slider(minimum=0, maximum=1), gr.Slider(minimum=0, maximum=1), False
            
            # Get video duration for segment selection
            video_duration = result.get("duration_sec", 0)
            if video_duration <= 0:
                video_duration = get_video_duration(input_video_path)
            
            # Update segment sliders
            start_slider = gr.Slider(minimum=0, maximum=video_duration, value=0, step=0.1, label="Start Time (seconds)")
            end_slider = gr.Slider(minimum=0, maximum=video_duration, value=video_duration, step=0.1, label="End Time (seconds)")
            
            # Generate a preview frame
            preview = generate_preview_frame(input_video_path, 0.5, 5.0, 2.5)
            
            # Enable segment checkbox only if video is longer than 30 seconds
            enable_segment = video_duration > 30
            
            return preview, f"Video loaded successfully: {result['width']}x{result['height']}, {result['fps']:.2f} FPS, {result['frame_count']} frames, {result['size_mb']:.2f}MB, Duration: {video_duration:.2f}s", start_slider, end_slider, enable_segment
        except Exception as e:
            return None, f"Error processing video upload: {str(e)}", gr.Slider(minimum=0, maximum=1), gr.Slider(minimum=0, maximum=1), False
    
    def download_from_url_handler(url):
        """Handle video URL input"""
        nonlocal input_video_path, video_duration
        
        if not url or url.strip() == "":
            return None, "Please enter a valid URL", gr.Slider(minimum=0, maximum=1), gr.Slider(minimum=0, maximum=1), False
        
        try:
            # Download video and convert to H.264 if needed
            input_video_path = download_from_url(url)
            
            # Validate video
            valid, result = validate_video(input_video_path)
            if not valid:
                return None, result, gr.Slider(minimum=0, maximum=1), gr.Slider(minimum=0, maximum=1), False
            
            # Get video duration for segment selection
            video_duration = result.get("duration_sec", 0)
            if video_duration <= 0:
                video_duration = get_video_duration(input_video_path)
            
            # Update segment sliders
            start_slider = gr.Slider(minimum=0, maximum=video_duration, value=0, step=0.1, label="Start Time (seconds)")
            end_slider = gr.Slider(minimum=0, maximum=video_duration, value=video_duration, step=0.1, label="End Time (seconds)")
            
            # Generate a preview frame
            preview = generate_preview_frame(input_video_path, 0.5, 5.0, 2.5)
            
            # Enable segment checkbox only if video is longer than 30 seconds
            enable_segment = video_duration > 30
            
            return preview, f"Video downloaded and converted successfully: {result['width']}x{result['height']}, {result['fps']:.2f} FPS, {result['frame_count']} frames, {result['size_mb']:.2f}MB, Duration: {video_duration:.2f}s", start_slider, end_slider, enable_segment
            
        except Exception as e:
            return None, f"Error downloading video: {str(e)}", gr.Slider(minimum=0, maximum=1), gr.Slider(minimum=0, maximum=1), False
    
    def update_preview(depth_intensity, convergence, eye_separation):
        """Update preview based on parameter changes"""
        nonlocal input_video_path
        
        if input_video_path is None or not os.path.exists(input_video_path):
            return None, "No video loaded"
        
        try:
            # Generate new preview with current parameters
            preview = generate_preview_frame(input_video_path, depth_intensity, convergence, eye_separation)
            return preview, "Preview updated with new parameters"
        except Exception as e:
            return None, f"Error updating preview: {str(e)}"
    
    def update_end_time(start_time):
        """Update the end time slider to ensure it's always greater than start time"""
        return gr.Slider(minimum=start_time + 0.1, maximum=video_duration, value=max(start_time + 0.1, video_duration))
    
    def sync_segment_values(use_segment, segment_start, segment_end):
        """Synchronize segment values between tabs"""
        # For Gradio compatibility, return a tuple of values instead of a dictionary
        return use_segment, segment_start, segment_end
    
    def process_video(depth_intensity, convergence, eye_separation, use_segment, segment_start, segment_end, progress=gr.Progress()):
        """Process the video with the given parameters"""
        nonlocal input_video_path, output_video_path, video_duration
        
        if input_video_path is None or not os.path.exists(input_video_path):
            return None, "No video loaded"
        
        try:
            # Create output directory
            output_dir = "output_videos"
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate output filename
            base_name = os.path.basename(input_video_path)
            name, ext = os.path.splitext(base_name)
            
            # Add segment info to output filename if using segment
            if use_segment and segment_start is not None and segment_end is not None and segment_start < segment_end:
                output_video_path = os.path.join(output_dir, f"{name}_3D_SBS_{int(segment_start)}-{int(segment_end)}s.mp4")
            else:
                output_video_path = os.path.join(output_dir, f"{name}_3D_SBS.mp4")
            
            # Process video
            process_video_to_3d_sbs(
                input_path=input_video_path, 
                output_path=output_video_path, 
                depth_intensity=depth_intensity, 
                convergence=convergence, 
                eye_separation=eye_separation,
                progress=progress,
                use_segment=use_segment,
                segment_start=segment_start if use_segment else None,
                segment_end=segment_end if use_segment else None
            )
            
            segment_info = f" (segment {segment_start:.1f}s-{segment_end:.1f}s)" if use_segment else ""
            return output_video_path, f"Video processed successfully{segment_info}. Saved to {output_video_path} with 16:9 aspect ratio (1920x1080) as requested."
            
        except Exception as e:
            return None, f"Error processing video: {str(e)}"
    
    # Create the Gradio interface
    with gr.Blocks(title="2D to 3D SBS Video Converter (GPU Optimized)") as app:
        gr.Markdown("# 2D to 3D Side-by-Side Video Converter (GPU Optimized)")
        gr.Markdown("Convert standard 2D videos to stereoscopic 3D SBS format for VR viewing. Output has a 16:9 aspect ratio (1920x1080) with both eye views side by side. **Preserves original audio track** in the output video.")
        
        if torch.cuda.is_available():
            gpu_info = f"Using GPU: {torch.cuda.get_device_name(0)} with {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB memory"
            gr.Markdown(f"**{gpu_info}**")
        else:
            gr.Markdown("**Running in CPU mode. Processing will be slower without GPU acceleration.**")
        
        with gr.Tab("Upload Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    # Video upload widget
                    upload_input = gr.File(
                        label="Upload Video File (max 500MB)",
                        file_types=["video"],
                        file_count="single"
                    )
                    upload_button = gr.Button("Upload and Preview")
                
                with gr.Column(scale=2):
                    # Preview and status
                    preview = gr.Image(label="Preview")
                    status = gr.Textbox(label="Status", interactive=False)
                    
                    # Segment selection (initially hidden/disabled)
                    use_segment = gr.Checkbox(label="Process a specific segment of the video", value=False, interactive=False)
                    segment_start = gr.Slider(minimum=0, maximum=1, value=0, step=0.1, label="Start Time (seconds)")
                    segment_end = gr.Slider(minimum=0, maximum=1, value=1, step=0.1, label="End Time (seconds)")
            
            # Connect upload button
            upload_button.click(
                upload_video, 
                inputs=[upload_input], 
                outputs=[preview, status, segment_start, segment_end, use_segment]
            )
            
            # Update end time slider when start time changes to maintain valid range
            segment_start.change(update_end_time, inputs=[segment_start], outputs=[segment_end])
            
        with gr.Tab("Video URL"):
            with gr.Row():
                with gr.Column(scale=1):
                    # URL input widget
                    url_input = gr.Textbox(label="Video URL (YouTube or direct link)")
                    url_button = gr.Button("Download and Preview")
                
                with gr.Column(scale=2):
                    # Preview and status (shared with upload tab)
                    url_preview = gr.Image(label="Preview")
                    url_status = gr.Textbox(label="Status", interactive=False)
                    
                    # Segment selection (initially hidden/disabled)
                    url_use_segment = gr.Checkbox(label="Process a specific segment of the video", value=False, interactive=False)
                    url_segment_start = gr.Slider(minimum=0, maximum=1, value=0, step=0.1, label="Start Time (seconds)")
                    url_segment_end = gr.Slider(minimum=0, maximum=1, value=1, step=0.1, label="End Time (seconds)")
            
            # Connect URL button
            url_button.click(
                download_from_url_handler, 
                inputs=[url_input], 
                outputs=[url_preview, url_status, url_segment_start, url_segment_end, url_use_segment]
            )
            
            # Update end time slider when start time changes to maintain valid range
            url_segment_start.change(update_end_time, inputs=[url_segment_start], outputs=[url_segment_end])
        
        with gr.Tab("Convert to 3D"):
            with gr.Row():
                with gr.Column(scale=1):
                    # Depth control parameters
                    depth_intensity = gr.Slider(
                        minimum=0.0, maximum=1.0, value=0.5, step=0.01,
                        label="Depth Intensity",
                        info="Controls the strength of the 3D effect (0.0-1.0)"
                    )
                    
                    convergence = gr.Slider(
                        minimum=1.0, maximum=10.0, value=5.0, step=0.1,
                        label="Convergence Distance",
                        info="Adjusts the perceived distance of objects (1.0-10.0)"
                    )
                    
                    eye_separation = gr.Slider(
                        minimum=0.1, maximum=5.0, value=2.5, step=0.1,
                        label="Eye Separation",
                        info="Controls the distance between virtual cameras (0.1-5.0)"
                    )
                    
                    # Segment selection (duplicated for this tab for better UX)
                    conv_use_segment = gr.Checkbox(label="Process a specific segment of the video", value=False)
                    conv_segment_start = gr.Slider(minimum=0, maximum=video_duration, value=0, step=0.1, label="Start Time (seconds)")
                    conv_segment_end = gr.Slider(minimum=0, maximum=video_duration, value=video_duration, step=0.1, label="End Time (seconds)")
                    
                    # Update end time slider when start time changes
                    conv_segment_start.change(update_end_time, inputs=[conv_segment_start], outputs=[conv_segment_end])
                    
                    # Update preview button
                    update_button = gr.Button("Update Preview")
                    
                    # Process button
                    process_button = gr.Button("Process Video", variant="primary")
                
                with gr.Column(scale=2):
                    # Preview and status (shared)
                    convert_preview = gr.Image(label="Preview")
                    convert_status = gr.Textbox(label="Status", interactive=False)
                    
                    # Output video
                    output_video = gr.Video(label="Converted 3D SBS Video (16:9 aspect ratio with audio)")
            
            # Connect update preview button
            update_button.click(
                update_preview,
                inputs=[depth_intensity, convergence, eye_separation],
                outputs=[convert_preview, convert_status]
            )
            
            # Connect process button
            process_button.click(
                process_video,
                inputs=[depth_intensity, convergence, eye_separation, conv_use_segment, conv_segment_start, conv_segment_end],
                outputs=[output_video, convert_status]
            )
            
            # Synchronize segment values between tabs
            # Connect the segment controls to the sync function
            use_segment.change(
                sync_segment_values,
                inputs=[use_segment, segment_start, segment_end],
                outputs=[conv_use_segment, conv_segment_start, conv_segment_end]
            )
            
            url_use_segment.change(
                sync_segment_values,
                inputs=[url_use_segment, url_segment_start, url_segment_end],
                outputs=[conv_use_segment, conv_segment_start, conv_segment_end]
            )
            
            # Sync back from Convert tab to others
            conv_use_segment.change(
                sync_segment_values,
                inputs=[conv_use_segment, conv_segment_start, conv_segment_end],
                outputs=[use_segment, segment_start, segment_end]
            )
            
            conv_use_segment.change(
                sync_segment_values,
                inputs=[conv_use_segment, conv_segment_start, conv_segment_end],
                outputs=[url_use_segment, url_segment_start, url_segment_end]
            )
        
        # Help tab
        with gr.Tab("Help"):
            gr.Markdown("""
            ## How to Use This Tool
            
            1. Upload a video file or provide a URL to a video (supports YouTube).
            2. For longer videos, you can choose to process only a specific segment to save time and memory:
               - Check the "Process a specific segment" box
               - Set the start and end times in seconds
            3. Adjust the depth parameters to control the 3D effect:
               - **Depth Intensity**: Controls the strength of the 3D effect. Higher values create more pronounced depth.
               - **Convergence Distance**: Adjusts where objects appear to be in relation to the screen plane.
               - **Eye Separation**: Controls the virtual camera separation. Higher values create more extreme 3D effects.
            4. Click "Update Preview" to see how your settings affect the 3D output.
            5. Click "Process Video" to convert the entire video (or selected segment) to 3D SBS format.
            6. Download the converted video for viewing in a VR headset or 3D display.
            
            ## Video Segmentation
            
            The video segment feature allows you to process only a portion of a longer video. This is useful for:
            - Testing different 3D settings on a small clip before processing the entire video
            - Processing very long videos in manageable chunks to avoid memory issues or timeouts
            - Creating highlights in 3D from specific parts of a longer video
            
            ## Output Format
            
            - The final video will have a 16:9 aspect ratio (1920x1080)
            - Each eye view is positioned side by side with appropriate proportions
            - Black bars are added as needed to maintain the proper 16:9 aspect ratio
            - H.264 encoded MP4 format for maximum compatibility
            - Maintains the original video's frame rate
            
            ## Supported Formats
            
            - Input: MP4, AVI, MOV, WebM, MKV (up to 4K resolution, max 500MB)
            - Output: H.264 encoded MP4 in Side-by-Side format (1920x1080)
            
            ## Viewing the 3D Video
            
            The output video is in Side-by-Side (SBS) format, which can be viewed in:
            - VR headsets using video players that support SBS format
            - 3D TVs with SBS viewing mode
            - Special 3D viewers like Google Cardboard with SBS-compatible apps
            
            ## GPU Optimization
            
            This version of the converter is optimized to take advantage of NVIDIA GPUs for faster processing:
            
            - Batch processing of multiple frames at once to maximize GPU utilization
            - GPU-accelerated depth map generation
            - Optimized memory management to handle larger videos
            - Hardware-accelerated video encoding when available
            
            ## Troubleshooting
            
            - If processing fails, try using a smaller segment of the video.
            - For best results, use videos with good lighting and clear objects.
            - If the 3D effect is too strong or causes discomfort, lower the Depth Intensity and Eye Separation values.
            - If you experience issues with YouTube downloads, try using a direct video URL instead.
            - If you experience any issues, check the status messages for error details.
            """)
    
    # Launch the app
    app.launch(debug=True, share=True)

# Initialize and launch the Gradio application
urL='http://127.0.0.1:7860'
webbrowser.get('windows-default').open_new(urL)
create_gradio_interface()
