import streamlit as st
import cv2
import numpy as np
from PIL import Image

def analyze_face(frame):
    # Initialize face detector
    face_cascade = cv2.CascadeClassifier()
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if not face_cascade.load(face_cascade_path):
        st.error(f"Error: Could not load face cascade classifier from {face_cascade_path}")
        return frame, {}
    
    # Initialize eye detector for more precise measurements
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    # Enhanced image processing
    frame = cv2.GaussianBlur(frame, (3, 3), 0)  # Reduced blur for more detail
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,  # Reduced for more precision
        minNeighbors=6,
        minSize=(150, 150),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    overlay = frame.copy()
    
    # Define facial regions
    regions = {
        'upper_forehead': (0, 0.15),
        'lower_forehead': (0.15, 0.33),
        'upper_eyes': (0.33, 0.39),
        'eyes': (0.39, 0.45),
        'upper_nose': (0.45, 0.55),
        'lower_nose': (0.55, 0.65),
        'upper_lip': (0.65, 0.73),
        'lower_lip': (0.73, 0.80),
        'upper_chin': (0.80, 0.90),
        'lower_chin': (0.90, 1.0)
    }
    
    all_scores = {}
    
    for (x, y, w, h) in faces:
        face_region = gray[y:y+h, x:x+w]
        face_center_x = x + w//2
        
        # Detect eyes for more precise alignment
        eyes = eye_cascade.detectMultiScale(face_region)
        if len(eyes) >= 2:
            # Sort eyes by x-coordinate
            eyes = sorted(eyes, key=lambda e: e[0])
            left_eye, right_eye = eyes[:2]
            
            # Adjust face center based on eyes
            left_eye_center = (x + left_eye[0] + left_eye[2]//2, y + left_eye[1] + left_eye[3]//2)
            right_eye_center = (x + right_eye[0] + right_eye[2]//2, y + right_eye[1] + right_eye[3]//2)
            face_center_x = (left_eye_center[0] + right_eye_center[0]) // 2
        
        region_scores = {}
        detailed_metrics = {}
        
        # Analyze each region with multiple metrics
        for region_name, (start_pct, end_pct) in regions.items():
            start_y = int(y + h * start_pct)
            end_y = int(y + h * end_pct)
            
            # Get region halves with minimal padding
            pad = 3
            left_half = gray[start_y-pad:end_y+pad, x-pad:face_center_x+pad]
            right_half = gray[start_y-pad:end_y+pad, face_center_x-pad:x+w+pad]
            
            # Skip this region if shape is invalid
            if left_half.size == 0 or right_half.size == 0:
                continue
                
            right_half_flipped = cv2.flip(right_half, 1)
            
            # Ensure same size
            min_width = min(left_half.shape[1], right_half_flipped.shape[1])
            min_height = min(left_half.shape[0], right_half_flipped.shape[0])
            
            if min_width <= 0 or min_height <= 0:
                continue
                
            left_half = left_half[:min_height, :min_width]
            right_half_flipped = right_half_flipped[:min_height, :min_width]
            
            # Multiple detailed metrics
            # Structural similarity
            diff = cv2.absdiff(left_half, right_half_flipped)
            mse = np.mean(diff ** 2)
            ssim = 1 - (mse / (255 * 255))
            
            # Edge analysis with multiple directions
            left_edges_x = cv2.Sobel(left_half, cv2.CV_64F, 1, 0)
            right_edges_x = cv2.Sobel(right_half_flipped, cv2.CV_64F, 1, 0)
            left_edges_y = cv2.Sobel(left_half, cv2.CV_64F, 0, 1)
            right_edges_y = cv2.Sobel(right_half_flipped, cv2.CV_64F, 0, 1)
            
            edge_similarity_x = 1 - np.mean(np.abs(left_edges_x - right_edges_x)) / 255
            edge_similarity_y = 1 - np.mean(np.abs(left_edges_y - right_edges_y)) / 255
            
            # Texture analysis
            left_texture = cv2.Laplacian(left_half, cv2.CV_64F)
            right_texture = cv2.Laplacian(right_half_flipped, cv2.CV_64F)
            texture_similarity = 1 - np.mean(np.abs(left_texture - right_texture)) / 255
            
            # Gradient magnitude similarity
            left_grad_mag = np.sqrt(left_edges_x**2 + left_edges_y**2)
            right_grad_mag = np.sqrt(right_edges_x**2 + right_edges_y**2)
            gradient_similarity = 1 - np.mean(np.abs(left_grad_mag - right_grad_mag)) / 255
            
            # Store detailed metrics
            detailed_metrics[region_name] = {
                'structural': float(ssim),
                'edge_x': float(edge_similarity_x),
                'edge_y': float(edge_similarity_y),
                'texture': float(texture_similarity),
                'gradient': float(gradient_similarity)
            }
            
            # Combined weighted score
            region_score = (
                0.25 * ssim +
                0.2 * edge_similarity_x +
                0.2 * edge_similarity_y +
                0.15 * texture_similarity +
                0.2 * gradient_similarity
            )
            region_scores[region_name] = float(region_score)
            
            # Draw detailed analysis lines
            # Horizontal region boundaries
            cv2.line(overlay, (x, start_y), (x+w, start_y), (255, 255, 255), 1)
            
            # Vertical analysis lines with varying density
            num_verticals = int(12 * (end_pct - start_pct))  # More lines in important regions
            for i in range(num_verticals):
                vert_x = x + (w * i // (num_verticals-1))
                color = (
                    int(255 * (1 - region_score)),
                    int(255 * region_score),
                    0
                )
                cv2.line(overlay, (vert_x, start_y), (vert_x, end_y), color, 1)
        
        # Draw rectangle around face and symmetry line
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 255, 255), 1)
        cv2.line(overlay, (face_center_x, y), (face_center_x, y+h), (0, 255, 0), 1)
        
        # Calculate overall symmetry score
        if region_scores:
            symmetry_score = np.mean(list(region_scores.values()))
            
            # Display overall score
            score_text = f"Symmetry: {symmetry_score:.3f}"
            text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = frame.shape[1] - text_size[0] - 10
            cv2.putText(overlay, 
                      score_text,
                      (text_x, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 
                      0.7,
                      (0, 0, 0),
                      2)
            
            # Store all scores for display in UI
            all_scores = {
                'overall': float(symmetry_score),
                'regions': region_scores,
                'detailed': detailed_metrics
            }
    
    # Smooth overlay blending
    alpha = 0.7
    result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    return result, all_scores

def main():
    st.set_page_config(page_title="Face Symmetry Analyzer", layout="wide")
    
    st.title("Face Symmetry Analyzer")
    st.write("This app analyzes facial symmetry using computer vision techniques.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Camera Input")
        run_analysis = st.checkbox("Start Analysis")
        
        if run_analysis:
            stframe = st.empty()
            vid_cap = cv2.VideoCapture(0)
            
            # Set camera properties
            vid_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            vid_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            vid_cap.set(cv2.CAP_PROP_FPS, 30)
            
            metrics_placeholder = st.empty()
            
            while run_analysis:
                success, img = vid_cap.read()
                if success:
                    img = cv2.flip(img, 1)
                    result_img, scores = analyze_face(img)
                    stframe.image(result_img, channels="BGR", use_column_width=True)
                    
                    if scores:
                        with metrics_placeholder.container():
                            st.subheader(f"Overall Symmetry: {scores['overall']:.3f}")
                            
                            # Create color for the score (red to green)
                            color = f"rgb({int(255*(1-scores['overall']))}, {int(255*scores['overall'])}, 0)"
                            st.markdown(f"<div style='background-color: {color}; padding: 10px; border-radius: 5px;'>"
                                      f"<h3 style='color: white; text-align: center;'>{scores['overall']:.3f}</h3>"
                                      f"</div>", unsafe_allow_html=True)
                else:
                    st.error("Failed to access camera")
                    break
                
            vid_cap.release()
        else:
            st.info("Check the box above to start the camera analysis")
            
            # Upload option as an alternative
            st.subheader("Or upload an image")
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                img_array = np.array(image)
                
                # Convert RGB to BGR (OpenCV format)
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                result_img, scores = analyze_face(img_array)
                
                st.image(result_img, channels="BGR", use_column_width=True)
                
                if scores:
                    st.subheader(f"Overall Symmetry: {scores['overall']:.3f}")
                    
                    # Create color for the score (red to green)
                    color = f"rgb({int(255*(1-scores['overall']))}, {int(255*scores['overall'])}, 0)"
                    st.markdown(f"<div style='background-color: {color}; padding: 10px; border-radius: 5px;'>"
                              f"<h3 style='color: white; text-align: center;'>{scores['overall']:.3f}</h3>"
                              f"</div>", unsafe_allow_html=True)
    
    with col2:
        st.subheader("About Face Symmetry")
        st.write("""
        This application analyzes facial symmetry using multiple computer vision techniques:
        
        - **Structural Similarity**: Comparing pixel values directly
        - **Edge Detection**: Comparing edge patterns horizontally and vertically
        - **Texture Analysis**: Comparing texture patterns
        - **Gradient Analysis**: Comparing directional changes
        
        The analysis divides the face into 10 regions and measures symmetry in each region. An overall score from 0 (asymmetrical) to 1 (perfectly symmetrical) is calculated.
        
        For best results:
        - Ensure good, even lighting
        - Face the camera directly
        - Keep your expression neutral
        - Remove glasses if possible
        """)
        
        st.subheader("Technical Details")
        with st.expander("How it works"):
            st.write("""
            The application uses OpenCV's Haar Cascade classifiers to detect faces and eyes. 
            For each detected face, it:
            
            1. Finds the vertical symmetry line
            2. Divides the face into regions (forehead, eyes, nose, lips, chin)
            3. For each region, mirrors the right side and compares with the left
            4. Calculates symmetry using multiple metrics
            
            The weighted combination of these metrics produces region scores, which are averaged for an overall symmetry score.
            """)

if __name__ == "__main__":
    main()
