import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import tempfile
from models.deepfake_detector import DeepfakeDetector
from utils.video_processor import VideoProcessor
from utils.image_processor import ImageProcessor

# Page configuration
st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'current_model' not in st.session_state:
    st.session_state.current_model = 'simple_cnn'

# Available models
AVAILABLE_MODELS = {
    'simple_cnn': 'Simple CNN',
    'advanced_cnn': 'Advanced CNN with Attention',
    'vision_transformer': 'Vision Transformer (ViT)',
    'cnn_transformer': 'CNN-Transformer Hybrid',
    'cnn_transformer_cls': 'CNN-Transformer with CLS Token'
}

def get_checkpoint_path(model_name):
    """Get checkpoint path for a model if it exists"""
    # First, check for the main trained model file
    main_checkpoint = './deepfake_model.pth'
    if os.path.exists(main_checkpoint):
        return main_checkpoint
    
    # Fall back to model-specific checkpoint
    checkpoint_dir = './checkpoints'
    checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_best.pth')
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    
    return None

def load_model(model_name=None):
    """Load the deepfake detection model"""
    if model_name is None:
        model_name = st.session_state.current_model
    
    # Reload if model changed
    if st.session_state.detector is None or st.session_state.current_model != model_name:
        with st.spinner(f"Loading {AVAILABLE_MODELS.get(model_name, model_name)} model..."):
            checkpoint_path = get_checkpoint_path(model_name)
            st.session_state.detector = DeepfakeDetector(
                model_name=model_name,
                checkpoint_path=checkpoint_path
            )
            st.session_state.model_loaded = True
            st.session_state.current_model = model_name
            
            # Display checkpoint info
            if checkpoint_path:
                st.sidebar.success(f"Loaded trained model from checkpoint")
            else:
                st.sidebar.info(f"Using untrained model (no checkpoint found)")
    
    return st.session_state.detector


def display_confidence_score(confidence, prediction):
    """Display confidence score with visual indicators"""
    # Color coding based on confidence and prediction
    if prediction == "Real":
        color = "green" if confidence > 0.7 else "orange"
        emoji = "✅" if confidence > 0.7 else "⚠️"
    else:
        color = "red" if confidence > 0.7 else "orange"
        emoji = "❌" if confidence > 0.7 else "⚠️"
    
    # Display confidence with color coding
    st.markdown(f"""
    <div style='text-align: center; padding: 20px; border-radius: 10px; 
                background-color: {color}; color: white; margin: 10px 0;'>
        <h2>{emoji} {prediction}</h2>
        <h3>Confidence: {confidence:.2%}</h3>
    </div>
    """, unsafe_allow_html=True)

def process_image(image, detector):
    """Process a single image and return prediction"""
    try:
        # Convert PIL image to numpy array if needed
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image
            
        # Get prediction
        prediction, confidence = detector.predict_image(image_array)
        
        return prediction, confidence
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None

def main():
    st.title("🔍 Deepfake Detection System")
    st.markdown("Advanced AI-powered detection of manipulated images and videos using CNN and Transformer models trained on OpenFake dataset")
    
    # Sidebar for settings
    st.sidebar.header("Settings")
    
    # Model selection
    st.sidebar.subheader("Model Selection")
    selected_model = st.sidebar.selectbox(
        "Choose Model Architecture",
        options=list(AVAILABLE_MODELS.keys()),
        format_func=lambda x: AVAILABLE_MODELS[x],
        key='model_selector' 
    ) #ChatGTP Addition: hardcode resnet
    
    # Detection mode
    detection_mode = st.sidebar.selectbox(
        "Detection Mode",
        ["Image Detection", "Video Detection", "Sample Testing"]
    )
    
    # Load model
    detector = load_model(selected_model)
    
    # Display model info in sidebar
    # st.sidebar.subheader("Model Information")
    # model_info = detector.get_model_info()
    # st.sidebar.write(f"**Architecture**: {model_info.get('architecture', 'unknown')}")
    # st.sidebar.write(f"**Parameters**: {model_info.get('parameters', 0):,}")
    # st.sidebar.write(f"**Device**: {model_info.get('device', 'unknown')}")
    # if 'checkpoint_val_acc' in model_info and model_info['checkpoint_val_acc'] != 'unknown':
    #     st.sidebar.write(f"**Validation Accuracy**: {model_info['checkpoint_val_acc']:.2f}%")
    # st.sidebar.write(f"**Status**: {'Trained' if model_info.get('trained', False) else 'Untrained'}")
    
    if not st.session_state.model_loaded:
        st.error("Failed to load the detection model. Please refresh the page.")
        return
    
    if detection_mode == "Image Detection":
        st.header("📸 Image Detection")
        st.markdown("Upload an image to detect if it's real or a deepfake")
        
        # File uploader with drag and drop
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Original Image")
                image = Image.open(uploaded_file)
                st.image(image, use_column_width=True)
            
            with col2:
                st.subheader("Detection Results")
                
                with st.spinner("Analyzing image..."):
                    prediction, confidence = process_image(image, detector)
                
                if prediction is not None:
                    display_confidence_score(confidence, prediction)
                    
                    # Additional details
                    st.markdown("### Analysis Details")
                    st.info(f"**Model**: {detector.model_name}")
                    st.info(f"**Image Size**: {image.size[0]}x{image.size[1]} pixels")
                    st.info(f"**Prediction**: {prediction} with {confidence:.2%} confidence")
    
    elif detection_mode == "Video Detection":
        st.header("🎥 Video Detection")
        st.markdown("Upload a video to analyze frames for deepfake detection")
        
        # Video settings
        col1, col2 = st.columns([1, 1])
        with col1:
            frame_sampling_rate = st.slider(
                "Frame Sampling Rate (frames per second)",
                min_value=1,
                max_value=10,
                value=2,
                help="Higher values analyze more frames but take longer"
            )
        
        with col2:
            max_frames = st.slider(
                "Maximum Frames to Analyze",
                min_value=10,
                max_value=100,
                value=30,
                help="Maximum number of frames to process"
            )
        
        # Video file uploader
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=["mp4", "avi", "mov", "mkv"],
            help="Supported formats: MP4, AVI, MOV, MKV"
        )
        
        if uploaded_video is not None:
            # Save uploaded video temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_video.read())
                tmp_video_path = tmp_file.name
            
            try:
                # Process video
                video_processor = VideoProcessor()
                
                with st.spinner("Extracting frames from video..."):
                    frames = video_processor.extract_frames(
                        tmp_video_path, 
                        sampling_rate=frame_sampling_rate,
                        max_frames=max_frames
                    )
                
                if frames:
                    st.success(f"Extracted {len(frames)} frames for analysis")
                    
                    # Analyze frames
                    progress_bar = st.progress(0)
                    predictions = []
                    confidences = []
                    
                    for i, frame in enumerate(frames):
                        prediction, confidence = process_image(frame, detector)
                        if prediction is not None:
                            predictions.append(prediction)
                            confidences.append(confidence)
                        progress_bar.progress((i + 1) / len(frames))
                    
                    if predictions:
                        # Calculate overall statistics
                        fake_count = predictions.count("Deepfake")
                        real_count = predictions.count("Real")
                        avg_confidence = np.mean(confidences)
                        
                        # Display results
                        col1, col2, col3 = st.columns([1, 1, 1])
                        
                        with col1:
                            st.metric("Frames Analyzed", len(predictions))
                        
                        with col2:
                            st.metric("Real Frames", real_count)
                        
                        with col3:
                            st.metric("Deepfake Frames", fake_count)
                        
                        # Overall assessment
                        fake_ratio = fake_count / len(predictions)
                        if fake_ratio > 0.5:
                            overall_prediction = "Deepfake"
                        else:
                            overall_prediction = "Real"
                        
                        display_confidence_score(avg_confidence, overall_prediction)
                        
                        # Frame-by-frame results
                        with st.expander("View Frame-by-Frame Results"):
                            for i, (frame, pred, conf) in enumerate(zip(frames[:10], predictions[:10], confidences[:10])):
                                col1, col2 = st.columns([1, 2])
                                with col1:
                                    st.image(frame, caption=f"Frame {i+1}", use_column_width=True)
                                with col2:
                                    st.write(f"**Prediction**: {pred}")
                                    st.write(f"**Confidence**: {conf:.2%}")
                                st.markdown("---")
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_video_path)
    
    elif detection_mode == "Sample Testing":
        st.header("🧪 Sample Testing")
        st.markdown("Test the system with pre-loaded sample images")
        
        # Sample images (using placeholder URLs since we can't include actual images)
        sample_images = {
            "Sample Real Image": "https://via.placeholder.com/400x300/4CAF50/FFFFFF?text=Sample+Real+Image",
            "Sample Deepfake Image": "https://via.placeholder.com/400x300/F44336/FFFFFF?text=Sample+Deepfake+Image"
        }
        
        selected_sample = st.selectbox("Choose a sample image", list(sample_images.keys()))
        
        if st.button("Analyze Sample"):
            try:
                # For demonstration, we'll create a placeholder image
                # In a real implementation, you would have actual sample images
                import requests
                from io import BytesIO
                
                response = requests.get(sample_images[selected_sample])
                image = Image.open(BytesIO(response.content))
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Sample Image")
                    st.image(image, use_column_width=True)
                
                with col2:
                    st.subheader("Detection Results")
                    
                    with st.spinner("Analyzing sample..."):
                        prediction, confidence = process_image(image, detector)
                    
                    if prediction is not None:
                        display_confidence_score(confidence, prediction)
                        
                        st.markdown("### Sample Analysis")
                        st.info(f"This is a {selected_sample.lower()} used for testing the detection system.")
                        
            except Exception as e:
                st.error(f"Error loading sample image: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
            <p>Deepfake Detection System - Powered by Advanced CNN and Transformer Models</p>
            <p>Training Data: OpenFake Dataset (~3M real + ~963k synthetic images)</p>
            <p>⚠️ This tool is for educational and research purposes</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
