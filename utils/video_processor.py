import numpy as np
from PIL import Image
import os
import tempfile

class VideoProcessor:
    def __init__(self):
        """Initialize video processor"""
        self.supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    
    def is_supported_format(self, file_path):
        """Check if video format is supported"""
        _, ext = os.path.splitext(file_path.lower())
        return ext in self.supported_formats
    
    def get_video_info(self, video_path):
        """
        Get basic information about the video
        Note: Requires OpenCV which is currently not installed
        
        Args:
            video_path: Path to video file
            
        Returns:
            dict: Video information
        """
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise Exception("Cannot open video file")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'duration': duration,
                'format': os.path.splitext(video_path)[1]
            }
            
        except ImportError:
            raise Exception("OpenCV is not installed. Video processing requires opencv-python-headless.")
        except Exception as e:
            raise Exception(f"Error getting video info: {str(e)}")
    
    def extract_frames(self, video_path, sampling_rate=1, max_frames=None, start_time=0, end_time=None):
        """
        Extract frames from video
        Note: Requires OpenCV which is currently not installed
        
        Args:
            video_path: Path to video file
            sampling_rate: Extract every nth frame (1 = every frame)
            max_frames: Maximum number of frames to extract
            start_time: Start time in seconds
            end_time: End time in seconds
            
        Returns:
            List of numpy arrays (frames)
        """
        try:
            import cv2
            
            if not self.is_supported_format(video_path):
                raise Exception(f"Unsupported video format: {os.path.splitext(video_path)[1]}")
            
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise Exception("Cannot open video file")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Calculate frame range
            start_frame = int(start_time * fps) if start_time > 0 else 0
            end_frame = int(end_time * fps) if end_time else total_frames
            end_frame = min(end_frame, total_frames)
            
            frames = []
            frame_count = 0
            current_frame = 0
            
            # Set starting position
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                current_frame = start_frame
            
            while current_frame < end_frame:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Sample frames based on sampling rate
                if current_frame % sampling_rate == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    frame_count += 1
                    
                    # Check max frames limit
                    if max_frames and frame_count >= max_frames:
                        break
                
                current_frame += 1
            
            cap.release()
            
            if not frames:
                raise Exception("No frames extracted from video")
            
            return frames
            
        except ImportError:
            raise Exception("OpenCV is not installed. Video processing requires opencv-python-headless. Please use Image Detection mode instead.")
        except Exception as e:
            raise Exception(f"Error extracting frames: {str(e)}")
    
    def extract_frames_at_timestamps(self, video_path, timestamps):
        """
        Extract frames at specific timestamps
        Note: Requires OpenCV which is currently not installed
        
        Args:
            video_path: Path to video file
            timestamps: List of timestamps in seconds
            
        Returns:
            List of numpy arrays (frames)
        """
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise Exception("Cannot open video file")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = []
            
            for timestamp in timestamps:
                frame_number = int(timestamp * fps)
                
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
            
            cap.release()
            return frames
            
        except ImportError:
            raise Exception("OpenCV is not installed. Video processing requires opencv-python-headless.")
        except Exception as e:
            raise Exception(f"Error extracting frames at timestamps: {str(e)}")
    
    def create_frame_montage(self, frames, grid_size=(3, 3), frame_size=(224, 224)):
        """
        Create a montage of frames
        
        Args:
            frames: List of frame arrays
            grid_size: (rows, cols) for the grid
            frame_size: Size to resize each frame
            
        Returns:
            numpy array: Montage image
        """
        try:
            rows, cols = grid_size
            max_frames = rows * cols
            
            # Limit frames to grid size
            frames = frames[:max_frames]
            
            # Resize frames
            resized_frames = []
            for frame in frames:
                if isinstance(frame, np.ndarray):
                    frame_pil = Image.fromarray(frame.astype(np.uint8))
                    frame_resized = frame_pil.resize(frame_size)
                    resized_frames.append(np.array(frame_resized))
            
            # Pad with black frames if needed
            while len(resized_frames) < max_frames:
                black_frame = np.zeros((*frame_size, 3), dtype=np.uint8)
                resized_frames.append(black_frame)
            
            # Create montage
            montage_height = rows * frame_size[1]
            montage_width = cols * frame_size[0]
            montage = np.zeros((montage_height, montage_width, 3), dtype=np.uint8)
            
            for i, frame in enumerate(resized_frames):
                row = i // cols
                col = i % cols
                
                y_start = row * frame_size[1]
                y_end = y_start + frame_size[1]
                x_start = col * frame_size[0]
                x_end = x_start + frame_size[0]
                
                montage[y_start:y_end, x_start:x_end] = frame
            
            return montage
            
        except Exception as e:
            raise Exception(f"Error creating frame montage: {str(e)}")
    
    def save_frames(self, frames, output_dir, prefix="frame"):
        """
        Save frames as individual image files
        
        Args:
            frames: List of frame arrays
            output_dir: Directory to save frames
            prefix: Prefix for frame filenames
            
        Returns:
            List of saved file paths
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            saved_paths = []
            
            for i, frame in enumerate(frames):
                filename = f"{prefix}_{i:05d}.jpg"
                filepath = os.path.join(output_dir, filename)
                
                # Convert to PIL Image and save
                if isinstance(frame, np.ndarray):
                    frame_pil = Image.fromarray(frame.astype(np.uint8))
                    frame_pil.save(filepath, "JPEG", quality=95)
                    saved_paths.append(filepath)
            
            return saved_paths
            
        except Exception as e:
            raise Exception(f"Error saving frames: {str(e)}")
