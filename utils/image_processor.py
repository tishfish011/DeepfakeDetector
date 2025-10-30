import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import os

class ImageProcessor:
    def __init__(self):
        """Initialize image processor"""
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    def is_supported_format(self, file_path):
        """Check if image format is supported"""
        _, ext = os.path.splitext(file_path.lower())
        return ext in self.supported_formats
    
    def load_image(self, image_path):
        """
        Load image from file path
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image object
        """
        try:
            if not self.is_supported_format(image_path):
                raise Exception(f"Unsupported image format: {os.path.splitext(image_path)[1]}")
            
            image = Image.open(image_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            raise Exception(f"Error loading image: {str(e)}")
    
    def resize_image(self, image, size, maintain_aspect_ratio=True):
        """
        Resize image to specified size
        
        Args:
            image: PIL Image object
            size: Tuple (width, height) or int for square
            maintain_aspect_ratio: Whether to maintain aspect ratio
            
        Returns:
            PIL Image object
        """
        try:
            if isinstance(size, int):
                size = (size, size)
            
            if maintain_aspect_ratio:
                # Calculate new size maintaining aspect ratio
                original_size = image.size
                ratio = min(size[0] / original_size[0], size[1] / original_size[1])
                new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
                
                # Resize and center on canvas
                resized = image.resize(new_size, Image.Resampling.LANCZOS)
                
                # Create new image with desired size and paste resized image
                new_image = Image.new('RGB', size, (0, 0, 0))
                paste_x = (size[0] - new_size[0]) // 2
                paste_y = (size[1] - new_size[1]) // 2
                new_image.paste(resized, (paste_x, paste_y))
                
                return new_image
            else:
                return image.resize(size, Image.Resampling.LANCZOS)
            
        except Exception as e:
            raise Exception(f"Error resizing image: {str(e)}")
    
    def enhance_image(self, image, brightness=1.0, contrast=1.0, saturation=1.0, sharpness=1.0):
        """
        Enhance image with various filters
        
        Args:
            image: PIL Image object
            brightness: Brightness factor (1.0 = original)
            contrast: Contrast factor (1.0 = original)
            saturation: Saturation factor (1.0 = original)
            sharpness: Sharpness factor (1.0 = original)
            
        Returns:
            PIL Image object
        """
        try:
            enhanced = image
            
            if brightness != 1.0:
                enhancer = ImageEnhance.Brightness(enhanced)
                enhanced = enhancer.enhance(brightness)
            
            if contrast != 1.0:
                enhancer = ImageEnhance.Contrast(enhanced)
                enhanced = enhancer.enhance(contrast)
            
            if saturation != 1.0:
                enhancer = ImageEnhance.Color(enhanced)
                enhanced = enhancer.enhance(saturation)
            
            if sharpness != 1.0:
                enhancer = ImageEnhance.Sharpness(enhanced)
                enhanced = enhancer.enhance(sharpness)
            
            return enhanced
            
        except Exception as e:
            raise Exception(f"Error enhancing image: {str(e)}")
    
    def apply_filters(self, image, filter_type='none'):
        """
        Apply various filters to image
        
        Args:
            image: PIL Image object
            filter_type: Type of filter to apply
            
        Returns:
            PIL Image object
        """
        try:
            if filter_type == 'blur':
                return image.filter(ImageFilter.BLUR)
            elif filter_type == 'sharpen':
                return image.filter(ImageFilter.SHARPEN)
            elif filter_type == 'edge_enhance':
                return image.filter(ImageFilter.EDGE_ENHANCE)
            elif filter_type == 'emboss':
                return image.filter(ImageFilter.EMBOSS)
            elif filter_type == 'smooth':
                return image.filter(ImageFilter.SMOOTH)
            else:
                return image
            
        except Exception as e:
            raise Exception(f"Error applying filter: {str(e)}")
    
    def detect_faces(self, image):
        """
        Detect faces in image using OpenCV (requires opencv-python-headless)
        
        Args:
            image: PIL Image object or numpy array
            
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        try:
            import cv2
            # Convert PIL to numpy array if needed
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            
            # Load face cascade classifier
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            return faces.tolist() if len(faces) > 0 else []
            
        except ImportError:
            raise Exception("OpenCV is not installed. Face detection requires opencv-python-headless.")
        except Exception as e:
            raise Exception(f"Error detecting faces: {str(e)}")
    
    def crop_faces(self, image, padding=20):
        """
        Crop faces from image
        
        Args:
            image: PIL Image object
            padding: Padding around detected faces
            
        Returns:
            List of cropped face images
        """
        try:
            faces = self.detect_faces(image)
            
            if not faces:
                return []
            
            cropped_faces = []
            image_array = np.array(image)
            
            for (x, y, w, h) in faces:
                # Add padding
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(image_array.shape[1], x + w + padding)
                y2 = min(image_array.shape[0], y + h + padding)
                
                # Crop face
                face_crop = image_array[y1:y2, x1:x2]
                face_image = Image.fromarray(face_crop)
                cropped_faces.append(face_image)
            
            return cropped_faces
            
        except Exception as e:
            raise Exception(f"Error cropping faces: {str(e)}")
    
    def get_image_stats(self, image):
        """
        Get basic statistics about the image
        
        Args:
            image: PIL Image object
            
        Returns:
            dict: Image statistics
        """
        try:
            image_array = np.array(image)
            
            stats = {
                'size': image.size,
                'mode': image.mode,
                'format': getattr(image, 'format', 'Unknown'),
                'mean_rgb': np.mean(image_array, axis=(0, 1)).tolist(),
                'std_rgb': np.std(image_array, axis=(0, 1)).tolist(),
                'min_rgb': np.min(image_array, axis=(0, 1)).tolist(),
                'max_rgb': np.max(image_array, axis=(0, 1)).tolist(),
                'total_pixels': image.size[0] * image.size[1]
            }
            
            return stats
            
        except Exception as e:
            raise Exception(f"Error getting image stats: {str(e)}")
    
    def normalize_image(self, image, target_size=(224, 224)):
        """
        Normalize image for model input
        
        Args:
            image: PIL Image object
            target_size: Target size for normalization
            
        Returns:
            numpy array: Normalized image array
        """
        try:
            # Resize image
            resized = self.resize_image(image, target_size, maintain_aspect_ratio=False)
            
            # Convert to numpy array
            image_array = np.array(resized, dtype=np.float32)
            
            # Normalize to [0, 1]
            normalized = image_array / 255.0
            
            return normalized
            
        except Exception as e:
            raise Exception(f"Error normalizing image: {str(e)}")
