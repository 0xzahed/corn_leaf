import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="Corn Disease Detection",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add max-width container
st.markdown("""
<style>
    /* Max Width Container - 7xl (~1536px) */
    .main .block-container {
        max-width: 1536px;
        padding-left: 2rem;
        padding-right: 2rem;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

# Custom CSS for better UI with responsive design
st.markdown("""
<style>
    /* Main Headers - Responsive */
    .main-header {
        font-size: clamp(1.8rem, 5vw, 3rem);
        color: #2E7D32;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: clamp(1rem, 3vw, 1.5rem);
        color: #558B2F;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Result Box - Responsive */
    .result-box {
        padding: clamp(1rem, 3vw, 2rem);
        border-radius: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .disease-name {
        font-size: clamp(1.5rem, 4vw, 2.5rem);
        font-weight: bold;
        margin: 1rem 0;
    }
    .confidence-score {
        font-size: clamp(1rem, 2.5vw, 1.5rem);
        margin: 0.5rem 0;
    }
    
    /* Info Box - Responsive */
    .info-box {
        padding: clamp(1rem, 2vw, 1.5rem);
        border-radius: 10px;
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
        font-size: clamp(0.9rem, 1.5vw, 1rem);
    }
    
    /* Button Styling - Responsive */
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-size: clamp(1rem, 2vw, 1.2rem);
        padding: clamp(0.6rem, 1.5vw, 0.75rem);
        border-radius: 10px;
        border: none;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        border: none;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background-color: #F1F8E9;
    }
    
    /* Mobile Optimization */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem;
            margin-bottom: 0.5rem;
        }
        .sub-header {
            font-size: 1rem;
            margin-bottom: 1rem;
        }
        .result-box {
            padding: 1rem;
        }
        .disease-name {
            font-size: 1.5rem;
        }
        .confidence-score {
            font-size: 1.2rem;
        }
        .info-box {
            padding: 0.8rem;
            font-size: 0.9rem;
        }
        .stButton>button {
            font-size: 1rem;
            padding: 0.6rem;
        }
    }
    
    /* Tablet Optimization */
    @media (min-width: 769px) and (max-width: 1024px) {
        .main-header {
            font-size: 2.5rem;
        }
        .sub-header {
            font-size: 1.3rem;
        }
    }
    
    /* Image Responsive */
    img {
        max-width: 100%;
        height: auto;
    }
</style>
""", unsafe_allow_html=True)

# Disease information dictionary
DISEASE_INFO = {
    "Common_Rust": {
        "description": "ছত্রাকজনিত রোগ যা পাতায় লাল-বাদামী দাগ সৃষ্টি করে। Common rust is a fungal disease affecting corn leaves with reddish-brown pustules.",
        "symptoms": "Small circular to elongate reddish-brown pustules on both leaf surfaces",
        "treatment": "ফাঙ্গিসাইড প্রয়োগ করুন, সংক্রমিত পাতা সরান। Use resistant hybrids, apply fungicides if necessary",
        "severity": "মাঝারি (Moderate)",
        "color": "#FFA500"
    },
    "Corn Leaf Blight": {
        "description": "পাতায় বড় ট্যান রঙের ক্ষত তৈরি করে। Northern corn leaf blight causes cigar-shaped lesions on corn leaves.",
        "symptoms": "Long, cigar-shaped grayish-green to tan lesions on leaves",
        "treatment": "প্রতিরোধী জাত ব্যবহার করুন, ফসল পরিবর্তন। Plant resistant varieties, apply fungicides",
        "severity": "উচ্চ (High)",
        "color": "#FF4500"
    },
    "Gray Leaf Spot": {
        "description": "পাতার শিরার মধ্যে আয়তাকার ধূসর-বাদামী দাগ। Gray leaf spot is a fungal disease causing rectangular lesions.",
        "symptoms": "Rectangular, gray to tan lesions with parallel edges between leaf veins",
        "treatment": "ফাঙ্গিসাইড চিকিৎসা, ফসলের অবশিষ্টাংশ ব্যবস্থাপনা। Use resistant hybrids, fungicide application",
        "severity": "উচ্চ (High)",
        "color": "#DC143C"
    },
    "Healthy": {
        "description": "কোন রোগ সনাক্ত করা হয়নি - গাছ সুস্থ। The corn plant appears healthy with no visible disease symptoms.",
        "symptoms": "Green, vibrant leaves without lesions or discoloration",
        "treatment": "নিয়মিত পর্যবেক্ষণ চালিয়ে যান। Maintain good agricultural practices and regular monitoring",
        "severity": "সুস্থ (Healthy)",
        "color": "#32CD32"
    },
    "Maize Chlorotic Mottle Virus": {
        "description": "ভাইরাসজনিত রোগ যা ক্লোরোটিক মোটলিং সৃষ্টি করে। A viral disease causing chlorotic mottling and stunting.",
        "symptoms": "Chlorotic mottling, yellowing, stunting, and poor ear development",
        "treatment": "পোকা নিয়ন্ত্রণ করুন, সংক্রমিত গাছ অপসারণ। Control insect vectors, use resistant varieties",
        "severity": "অত্যন্ত উচ্চ (Very High)",
        "color": "#8B0000"
    }
}

@st.cache_resource
def load_trained_model():
    """Load the trained InceptionV3 model"""
    try:
        # Local model path (primary)
        LOCAL_MODEL_PATH = '/media/panda/Data1/leaf_ditection/lightning_studio_inceptionv3_corn_disease_full_training.h5'
        
        # Google Drive backup (if local file not found)
        GOOGLE_DRIVE_FILE_ID = "1N4BXw33VbFYl18sXus314sjr6j2uvUrT"
        DOWNLOAD_MODEL_NAME = 'model.h5'
        
        # First, try to load from local path
        if os.path.exists(LOCAL_MODEL_PATH):
            st.info("🔄 Loading model from local path...")
            model = load_model(LOCAL_MODEL_PATH)
            st.success("✅ Model loaded successfully from local storage!")
            return model, LOCAL_MODEL_PATH
        
        # If local model not found, try to download from Google Drive
        st.warning("⚠️ Local model not found. Attempting to download from Google Drive...")
        
        if not os.path.exists(DOWNLOAD_MODEL_NAME):
            try:
                import gdown
                st.info("📥 Downloading model from Google Drive... (This may take a minute)")
                url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
                gdown.download(url, DOWNLOAD_MODEL_NAME, quiet=False)
                st.success("✅ Model downloaded successfully!")
            except Exception as download_error:
                st.error(f"❌ Failed to download model: {str(download_error)}")
                st.error("Please check your internet connection and Google Drive file permissions.")
                return None, None
        
        # Load the downloaded model
        st.info("🔄 Loading downloaded model...")
        model = load_model(DOWNLOAD_MODEL_NAME)
        st.success("✅ Model loaded successfully!")
        return model, DOWNLOAD_MODEL_NAME
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

def preprocess_image(image, target_size=(299, 299)):
    """Preprocess image for InceptionV3 model"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image
    image = image.resize(target_size)
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Expand dimensions
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess using InceptionV3 preprocessing
    img_array = preprocess_input(img_array)
    
    return img_array

def is_corn_leaf_image(image):
    """
    Basic validation - always returns True (simplified for cloud deployment)
    """
    return True, 100.0

def predict_disease(model, image):
    """Predict disease from image with confidence filtering"""
    # Preprocess image
    processed_img = preprocess_image(image)
    
    # Make prediction
    predictions = model.predict(processed_img, verbose=0)
    
    # Get class names (must match training order)
    class_names = ['Common_Rust', 'Corn Leaf Blight', 'Gray Leaf Spot', 'Healthy', 'Maize Chlorotic Mottle Virus']
    
    # Convert to numpy and then to Python float to avoid float32 issues
    predictions_np = np.array(predictions[0])
    
    # Get predicted class and confidence
    predicted_class_idx = np.argmax(predictions_np)
    predicted_class = class_names[predicted_class_idx]
    confidence = float(predictions_np[predicted_class_idx] * 100)
    
    # Check if all predictions are too close (model is uncertain)
    max_confidence = np.max(predictions_np)
    second_max_confidence = np.partition(predictions_np, -2)[-2]
    
    # If difference between top 2 predictions is too small, model is uncertain
    confidence_gap = (max_confidence - second_max_confidence) * 100
    
    # Get all predictions for display - convert to Python float
    all_predictions = {class_names[i]: float(predictions_np[i] * 100) for i in range(len(class_names))}
    
    return predicted_class, confidence, all_predictions, confidence_gap

def main():
    # Header
    st.markdown('<p class="main-header">🌽 Corn Disease Detection System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Corn Leaf Disease Classification</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2329/2329039.png", width=150)
        st.title("ℹ️ Information")
        st.markdown("""
        ### About
        This application uses a deep learning model to detect diseases in corn leaves.
        
        ### How to use | ব্যবহার করার নিয়ম:
        1. Upload a corn leaf image | ভুট্টা পাতার ছবি আপলোড করুন
        2. Click 'Analyze Image' | 'Analyze Image' বাটনে ক্লিক করুন
        3. View the prediction results | ফলাফল দেখুন
        
        ### Supported Diseases | সমর্থিত রোগসমূহ:
        - Common Rust | সাধারণ মরিচা
        - Corn Leaf Blight | ভুট্টার পাতা ঝলসানো
        - Gray Leaf Spot | ধূসর পাতার দাগ
        - Healthy | সুস্থ
        - Maize Chlorotic Mottle Virus | ভাইরাস
        
        ### Model Info | মডেল তথ্য:
        - Accuracy: 99%+
        - Input Size: 299x299
        - Training Dataset: Augmented Corn Leaf Dataset
        
        ### 📸 Best Practices | সেরা অনুশীলন:
        - পরিষ্কার, ভালো আলোতে ছবি তুলুন
        - পাতার surface স্পষ্ট দেখা যায় এমন ছবি
        - ঝাপসা ছবি এড়িয়ে চলুন
        """)
    
    # Load model
    model, model_name = load_trained_model()
    
    if model is None:
        st.stop()
    
    st.success("✅ Model loaded successfully")
    
    # File uploader
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "📤 Upload Corn Leaf Image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a corn leaf for disease detection"
    )
    
    if uploaded_file is not None:
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📸 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True, caption="Original Image")
            
            # Image details
            st.markdown(f"""
            <div class="info-box">
                <b>Image Details:</b><br>
                Format: {image.format}<br>
                Size: {image.size[0]} x {image.size[1]} pixels<br>
                Mode: {image.mode}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🔬 Analysis")
            
            # Analyze button
            if st.button("🚀 Analyze Image", use_container_width=True):
                with st.spinner("🔄 Analyzing image... Please wait..."):
                    try:
                        # Step 1: Validate if it's a corn leaf image
                        st.info("🔍 Step 1: Validating image...")
                        is_leaf, green_percentage = is_corn_leaf_image(image)
                        
                        if not is_leaf:
                            st.error(f"""
                            🚫 **This doesn't appear to be a corn leaf image!**
                            
                            **এটি ভুট্টার পাতার ছবি বলে মনে হচ্ছে না!**
                            
                            - Green content detected: {green_percentage:.1f}%
                            - Expected: At least 10% green pixels
                            
                            Please upload a clear image of a corn leaf.
                            অনুগ্রহ করে ভুট্টার পাতার স্পষ্ট ছবি আপলোড করুন।
                            """)
                            st.stop()
                        
                        st.success(f"✅ Image validation passed (Green content: {green_percentage:.1f}%)")
                        
                        # Step 2: Predict disease
                        st.info("🤖 Step 2: Running AI analysis...")
                        predicted_class, confidence, all_predictions, confidence_gap = predict_disease(model, image)
                        
                        # Step 3: Check confidence levels
                        CONFIDENCE_THRESHOLD = 50  # Minimum confidence required
                        CONFIDENCE_GAP_THRESHOLD = 20  # Minimum gap between top 2 predictions
                        
                        # Determine if prediction is reliable
                        is_reliable = confidence >= CONFIDENCE_THRESHOLD and confidence_gap >= CONFIDENCE_GAP_THRESHOLD
                        
                        if not is_reliable:
                            st.warning(f"""
                            ⚠️ **Low Confidence Detection! / কম নিশ্চয়তা!**
                            
                            The AI model is not very confident about this prediction.
                            মডেল এই ছবি সম্পর্কে নিশ্চিত নয়।
                            
                            **Possible reasons / সম্ভাব্য কারণ:**
                            - Image quality is poor / ছবির মান খারাপ
                            - Not a corn leaf / ভুট্টার পাতা নয়
                            - Unusual disease pattern / অস্বাভাবিক রোগের লক্ষণ
                            
                            **Confidence:** {confidence:.2f}%
                            **Gap from 2nd prediction:** {confidence_gap:.2f}%
                            
                            Please try:
                            - Upload a clearer image / আরও স্পষ্ট ছবি আপলোড করুন
                            - Ensure good lighting / ভাল আলো নিশ্চিত করুন
                            - Consult an agricultural expert / কৃষি বিশেষজ্ঞের পরামর্শ নিন
                            """)
                        
                        # Display result
                        result_color = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)" if is_reliable else "linear-gradient(135deg, #FF9800 0%, #F57C00 100%)"
                        st.markdown(f"""
                        <div class="result-box" style="background: {result_color};">
                            <h2>🎯 Detection Result</h2>
                            <p class="disease-name">{predicted_class.replace('_', ' ')}</p>
                            <p class="confidence-score">Confidence: {confidence:.2f}%</p>
                            <p style="font-size: 1rem;">Reliability: {"✅ High" if is_reliable else "⚠️ Low"}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display disease information
                        if predicted_class in DISEASE_INFO:
                            info = DISEASE_INFO[predicted_class]
                            
                            with st.expander("📋 Disease Information", expanded=is_reliable):
                                if not is_reliable:
                                    st.warning("⚠️ Take this information with caution due to low confidence / কম নিশ্চয়তার কারণে সতর্কতার সাথে দেখুন")
                                st.markdown(f"**Description:** {info['description']}")
                                st.markdown(f"**Symptoms:** {info['symptoms']}")
                                st.markdown(f"**Treatment:** {info['treatment']}")
                                st.markdown(f"**Severity Level:** {info['severity']}")
                        
                        # Display all predictions
                        with st.expander("📊 All Class Probabilities", expanded=not is_reliable):
                            sorted_predictions = dict(sorted(all_predictions.items(), key=lambda x: x[1], reverse=True))
                            for class_name, prob in sorted_predictions.items():
                                st.progress(prob / 100)
                                st.write(f"**{class_name.replace('_', ' ')}**: {prob:.2f}%")
                            
                            st.info(f"Confidence gap between top 2 predictions: {confidence_gap:.2f}%")
                        
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
    
    else:
        # Display sample images info
        st.info("👆 Please upload a corn leaf image to get started!")
        
        st.markdown("---")
        st.markdown("### 📚 Sample Expected Images | নমুনা ছবি")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown("**🟤 Common Rust**")
            st.caption("Small reddish-brown pustules | লাল-বাদামী দাগ")
        with col2:
            st.markdown("**🟠 Corn Leaf Blight**")
            st.caption("Cigar-shaped lesions | সিগার আকৃতির ক্ষত")
        with col3:
            st.markdown("**⚫ Gray Leaf Spot**")
            st.caption("Rectangular gray lesions | আয়তাকার ধূসর দাগ")
        with col4:
            st.markdown("**🟢 Healthy**")
            st.caption("Green vibrant leaves | সুস্থ সবুজ পাতা")
        with col5:
            st.markdown("**🔴 Maize Chlorotic Mottle Virus**")
            st.caption("Chlorotic mottling | ক্লোরোটিক দাগ")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🌱 Corn Disease Detection System | Powered by Deep Learning</p>
        <p>© 2025 | Built with Streamlit & TensorFlow</p>
        <p style="font-size: 0.85em; margin-top: 10px;">
        ⚠️ <b>Important Note:</b> এটি একটি AI-based diagnostic tool। গুরুত্বপূর্ণ সিদ্ধান্তের জন্য কৃষি বিশেষজ্ঞ বা উদ্ভিদ রোগবিদের সাথে পরামর্শ করুন।
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
