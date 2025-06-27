import streamlit as st
from PIL import Image, ImageDraw
import pytesseract
import cv2
import numpy as np
import io
import csv
import re # For regex in MRZ parsing
from collections import defaultdict # For grouping detected words

try:
    # Attempt to get Tesseract version to confirm executable path is correct
    pytesseract.get_tesseract_version()
except pytesseract.TesseractNotFoundError:
    st.error("Tesseract OCR engine not found. Please ensure:")
    st.markdown("1. Tesseract is installed on your system.")
    st.markdown(f"2. (Local Dev) The path to `tesseract.exe` (or `tesseract` on Linux/macOS) in `app.py` is correct if specified.")
    st.markdown("3. (Local Dev - Windows) Tesseract's installation directory (e.g., `C:\\Program Files\\Tesseract-OCR\\`) is added to your system's PATH environmental variable.")
    st.markdown("4. (Streamlit Cloud) Add `tesseract-ocr` to your `packages.txt` file.")
    st.info("Download Tesseract from: https://tesseract-ocr.github.io/tessdoc/Downloads.html")
    st.stop() # Stop the app if Tesseract is not found

# --- Define Target Image Size for Consistent Regions ---
# All uploaded images will be resized to these exact dimensions (width x height)
# to ensure that coordinates remain consistent for auto-detection logic.
REFERENCE_WIDTH = 1280
REFERENCE_HEIGHT = 818


# --- Utility Function for Image Preprocessing ---
def preprocess_image(image_bytes,
                     target_dims=None,
                     grayscale=True,
                     brightness_factor=1.0,
                     contrast_factor=1.0,
                     sharpen_intensity=0,
                     bilateral_filter=False,
                     deskew=False,
                     threshold_type='adaptive_gaussian',
                     noise_reduction_type='median_blur',
                     morph_open=False,
                     morph_close=False,
                     scale_factor=1.0,
                     contrast_enhance=False):
    """
    Applies various image preprocessing techniques to enhance text for OCR.

    Args:
        image_bytes (bytes): The raw bytes of the image file.
        target_dims (tuple, optional): A tuple (width, height) to which the image will be
                                       resized first, exactly. This is crucial for consistent processing.
        grayscale (bool): Convert image to grayscale.
        brightness_factor (float): Factor to adjust brightness (1.0 is original).
        contrast_factor (float): Factor to adjust contrast (1.0 is original).
        sharpen_intensity (int): Intensity of sharpening (0 for no sharpen, higher for more).
        bilateral_filter (bool): Apply bilateral filter for edge-preserving noise reduction.
        deskew (bool): Flag to indicate if deskewing should be applied using the angle from session state.
        threshold_type (str): Type of thresholding to apply ('none', 'simple', 'adaptive_mean', 'adaptive_gaussian').
        noise_reduction_type (str): Type of general noise reduction to apply ('none', 'median_blur', 'fast_n_means').
        morph_open (bool): Apply morphological opening to remove small noise.
        morph_close (bool): Apply morphological closing to fill small holes/gaps in text.
        scale_factor (float): Factor to scale the image up/down (e.g., 2.0 for 2x upscaling).
                               This scaling happens *after* the initial fixed `target_dims` resize.
        contrast_enhance (bool): Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]:
            (fully_processed_image_cv2_for_ocr, visually_processed_image_cv2_for_display)
            The first is optimized for OCR (possibly binarized), the second for human viewing.
    """
    image_np = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Could not load image. Please check the file format or if the file is corrupted.")
        return None, None

    if target_dims and (img.shape[1] != target_dims[0] or img.shape[0] != target_dims[1]):
        img = cv2.resize(img, target_dims, interpolation=cv2.INTER_AREA)

    if grayscale and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if scale_factor != 1.0:
        img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    processed_image_for_ocr_cv2 = img.copy()
    processed_image_for_display_cv2 = img.copy()

    if processed_image_for_ocr_cv2.dtype != np.uint8:
        processed_image_for_ocr_cv2 = processed_image_for_ocr_cv2.astype(np.uint8)
    if processed_image_for_display_cv2.dtype != np.uint8:
        processed_image_for_display_cv2 = processed_image_for_display_cv2.astype(np.uint8)

    # Apply preprocessing for display and OCR images
    for temp_img in [processed_image_for_display_cv2, processed_image_for_ocr_cv2]:
        temp_img = cv2.convertScaleAbs(temp_img, alpha=contrast_factor, beta=(brightness_factor - 1) * 127)
        if bilateral_filter and len(temp_img.shape) == 2:
            temp_img = cv2.bilateralFilter(temp_img, 9, 75, 75)
        elif noise_reduction_type == 'median_blur':
            temp_img = cv2.medianBlur(temp_img, 5)
        elif noise_reduction_type == 'fast_n_means':
            if len(temp_img.shape) == 2:
                temp_img = cv2.fastNlMeansDenoising(temp_img, None, h=30, templateWindowSize=7, searchWindowSize=21)
            else:
                temp_img = cv2.fastNlMeansDenoisingColored(temp_img, None, hColor=30, h=30, templateWindowSize=7, searchWindowSize=21)
        if contrast_enhance and len(temp_img.shape) == 2:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            temp_img = clahe.apply(temp_img)
        if sharpen_intensity > 0:
            K = sharpen_intensity
            kernel = np.array([[0, -K, 0], [-K, 1 + 4 * K, -K], [0, -K, 0]], dtype=np.float32)
            temp_img = cv2.filter2D(temp_img, -1, kernel)
        
        # Deskewing: apply the stored angle
        if deskew and 'angle_detected_for_display' in st.session_state and st.session_state.angle_detected_for_display != 0.0:
            angle = st.session_state.angle_detected_for_display
            (h, w) = temp_img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            temp_img = cv2.warpAffine(temp_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        # Assign back to the correct variable
        if temp_img is processed_image_for_display_cv2:
            processed_image_for_display_cv2 = temp_img
        else:
            processed_image_for_ocr_cv2 = temp_img

    # OCR-Specific Thresholding and Morphological Operations (only on processed_image_for_ocr_cv2)
    if len(processed_image_for_ocr_cv2.shape) == 2 and threshold_type != 'none':
        if threshold_type == 'simple':
            _, processed_image_for_ocr_cv2 = cv2.threshold(processed_image_for_ocr_cv2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif threshold_type == 'adaptive_mean':
            processed_image_for_ocr_cv2 = cv2.adaptiveThreshold(processed_image_for_ocr_cv2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        elif threshold_type == 'adaptive_gaussian':
            processed_image_for_ocr_cv2 = cv2.adaptiveThreshold(processed_image_for_ocr_cv2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        if morph_open:
            kernel = np.ones((2,2), np.uint8)
            processed_image_for_ocr_cv2 = cv2.morphologyEx(processed_image_for_ocr_cv2, cv2.MORPH_OPEN, kernel)
        if morph_close:
            kernel = np.ones((2,2), np.uint8)
            processed_image_for_ocr_cv2 = cv2.morphologyEx(processed_image_for_ocr_cv2, cv2.MORPH_CLOSE, kernel)

    return processed_image_for_ocr_cv2, processed_image_for_display_cv2


# --- Function for Auto-Refining a Specific Field Region (Improved for multi-word/full detection) ---
def auto_refine_field_region(image_for_detection_cv2, base_bbox, padding=60):
    """
    Refines a given base bounding box by finding and merging text contours
    within an expanded search area around it. Designed to capture full words/phrases.
    """
    img_h, img_w = image_for_detection_cv2.shape[:2]
    bx, by, bw, bh = base_bbox

    sx1 = max(0, bx - padding)
    sy1 = max(0, by - padding)
    sx2 = min(img_w, bx + bw + padding)
    sy2 = min(img_h, by + bh + padding)

    search_crop = image_for_detection_cv2[sy1:sy2, sx1:sx2].copy()

    if search_crop.size == 0 or search_crop.max() == search_crop.min():
        fallback_padding = 30
        fb_x1 = max(0, bx - fallback_padding)
        fb_y1 = max(0, by - fallback_padding)
        fb_x2 = min(img_w, bx + bw + fallback_padding)
        fb_y2 = min(img_h, by + bh + fallback_padding)
        return (fb_x1, fb_y1, fb_x2 - fb_x1, fb_y2 - fb_y1)

    if len(search_crop.shape) == 3:
        search_crop = cv2.cvtColor(search_crop, cv2.COLOR_BGR2GRAY)
    
    if search_crop.max() > 1 and search_crop.min() < 255:
        search_crop_binary = cv2.adaptiveThreshold(search_crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    else:
        search_crop_binary = search_crop.copy()

    if np.mean(search_crop_binary) > 127:
           search_crop_binary = cv2.bitwise_not(search_crop_binary)

    dilation_kernel_width = max(5, int(bh * 0.7))
    dilation_kernel_height = max(3, int(bh * 0.25))
    dilation_kernel = np.ones((dilation_kernel_height, dilation_kernel_width), np.uint8) 
    dilated_search_crop = cv2.dilate(search_crop_binary, dilation_kernel, iterations=4)

    contours, _ = cv2.findContours(dilated_search_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidate_bboxes = []
    min_text_height = max(8, int(bh * 0.4))
    max_text_height = int(bh * 2.0)

    for cnt in contours:
        x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
        area = w_c * h_c

        if area < 100:
            continue
        if h_c < min_text_height or h_c > max_text_height:
            continue
        
        aspect_ratio = w_c / h_c if h_c > 0 else 0
        if not (0.01 < aspect_ratio < 100.0):
            continue
        
        global_x = sx1 + x_c
        global_y = sy1 + y_c
        candidate_bboxes.append((global_x, global_y, w_c, h_c))

    if not candidate_bboxes:
        fallback_padding = 30
        fb_x1 = max(0, bx - fallback_padding)
        fb_y1 = max(0, by - fallback_padding)
        fb_x2 = min(img_w, bx + bw + fallback_padding)
        fb_y2 = min(img_h, by + bh + fallback_padding)
        return (fb_x1, fb_y1, fb_x2 - fb_x1, fb_y2 - fb_y1)

    candidate_bboxes.sort(key=lambda b: (b[1], b[0]))

    merged_bboxes = []
    if candidate_bboxes:
        current_group = list(candidate_bboxes[0])
        for i in range(1, len(candidate_bboxes)):
            next_box = candidate_bboxes[i]
            current_x1, current_y1, current_w, current_h = current_group
            next_x1, next_y1, next_w, next_h = next_box

            horizontal_gap_threshold = max(35, int(current_h * 1.5))
            vertical_overlap_min_ratio = 0.85
            
            dist_x = next_x1 - (current_x1 + current_w)
            overlap_y1 = max(current_y1, next_y1)
            overlap_y2 = min(current_y1 + current_h, next_y1 + next_h)
            vertical_overlap_height = max(0, overlap_y2 - overlap_y1)

            if (dist_x < horizontal_gap_threshold and dist_x > - (next_w * 0.7) and
                vertical_overlap_height / min(current_h, next_h) >= vertical_overlap_min_ratio):
                current_group[0] = min(current_x1, next_x1)
                current_group[1] = min(current_y1, next_y1)
                current_group[2] = max(current_x1 + current_w, next_x1 + next_w) - current_group[0]
                current_group[3] = max(current_y1 + current_h, next_y1 + next_h) - current_group[1]
            else:
                merged_bboxes.append(tuple(current_group))
                current_group = list(next_box)
        merged_bboxes.append(tuple(current_group))

    best_merged_bbox = None
    max_score = -1

    base_center_y = by + bh / 2
    base_center_x = bx + bw / 2

    for m_x, m_y, m_w, m_h in merged_bboxes:
        area = m_w * m_h
        m_center_y = m_y + m_h / 2
        m_center_x = m_x + m_w / 2
        
        vertical_dist_penalty = abs(m_center_y - base_center_y)
        horizontal_dist_penalty = abs(m_center_x - base_center_x) * 0.02

        score = area - (vertical_dist_penalty * 20) - (horizontal_dist_penalty * 0.1)

        overlap_x1 = max(bx, m_x)
        overlap_x2 = min(bx + bw, m_x + m_w)
        overlap_width = max(0, overlap_x2 - overlap_x1)
        
        if overlap_width < bw * 0.65:
            continue
        
        width_overage_penalty = max(0, m_w - bw) * 0.9
        height_overage_penalty = max(0, m_h - bh) * 0.7
        score -= (width_overage_penalty + height_overage_penalty)

        if m_w > bw * 1.5 or m_h > bh * 2.0:
            score -= area * 0.5

        if score > max_score and m_w > 0 and m_h > 0:
            best_merged_bbox = (m_x, m_y, m_w, m_h)
            max_score = score
            
    if best_merged_bbox:
        buffer_x = 20
        buffer_y = 20
        rx, ry, rw, rh = best_merged_bbox
        
        final_x1 = max(0, rx - buffer_x)
        final_y1 = max(0, ry - buffer_y)
        final_x2 = min(img_w, rx + rw + buffer_x)
        final_y2 = min(img_h, ry + rh + buffer_y)

        final_w = max(0, final_x2 - final_x1)
        final_h = max(0, final_y2 - final_y1)

        if final_w < 20 or final_h < 10:
            fallback_padding = 30
            final_x1 = max(0, bx - fallback_padding)
            final_y1 = max(0, by - fallback_padding)
            final_x2 = min(img_w, bx + bw + fallback_padding)
            final_y2 = min(img_h, by + bh + fallback_padding)
            final_w = max(0, final_x2 - final_x1)
            final_h = max(0, final_y2 - final_y1)

        return (final_x1, final_y1, final_w, final_h)
    else:
        fallback_padding = 30
        fb_x1 = max(0, bx - fallback_padding)
        fb_y1 = max(0, by - fallback_padding)
        fb_x2 = min(img_w, bx + bw + fallback_padding)
        fb_y2 = min(img_h, by + bh + fallback_padding)
        return (fb_x1, fb_y1, fb_x2 - fb_x1, fb_y2 - fb_y1)


# --- Automatic Region Detection based on Labels ---

# Define keywords for each field. Removed fixed size/offset hints as they are now dynamically calculated.
EXPECTED_LABELS_AND_VALUE_HINTS = {
    "Passport No.": ["Passport No.", "Passport Number", "No."],
    "Surname": ["Surname"],
    "Given Names": ["Given Names", "Given Name", "Name"],
    "Nationality": ["Nationality"],
    "Date of Birth": ["Date of Birth", "DOB", "Birth"],
    "Sex": ["Sex", "Gender"],
    "Place of Birth": ["Place of Birth", "Place of Issuance"],
    "Date of Issue": ["Date of Issue", "Issue Date"],
    "Date of Expiry": ["Date of Expiry", "Expiry Date", "Exp.", "Expiration"],
}

def auto_detect_field_regions(image_pil_ocr):
    """
    Automatically detects regions for specified fields based on label keywords.
    
    Args:
        image_pil_ocr (PIL.Image): The preprocessed grayscale PIL image for OCR.

    Returns:
        dict: A dictionary where keys are field names and values are (x, y, w, h) bounding boxes.
              Returns an empty dictionary if no fields can be detected.
    """
    detected_regions = {}
    
    # Perform full image OCR to get all word data
    # PSM 3 is suitable for general text layout analysis to find labels.
    data = pytesseract.image_to_data(image_pil_ocr, output_type=pytesseract.Output.DICT, lang='eng', config='--oem 3 --psm 3')

    img_width, img_height = image_pil_ocr.size

    for field_name, labels_keywords in EXPECTED_LABELS_AND_VALUE_HINTS.items():
        found_label_bbox = None
        
        # Iterate through detected words to find the label
        for i in range(len(data['text'])):
            word = data['text'][i].strip()
            # Ensure confidence is high enough for the label itself
            if not word or data['conf'][i] < 60: # Added a confidence threshold for label detection
                continue
            
            # Check if any of the label keywords are present in the detected word/phrase
            for label_keyword in labels_keywords:
                # Using simple 'in' for flexibility, case-insensitive match
                if label_keyword.lower() in word.lower():
                    # Get the bounding box of the found label
                    x = data['left'][i]
                    y = data['top'][i]
                    w = data['width'][i]
                    h = data['height'][i]
                    found_label_bbox = (x, y, w, h)
                    break # Found the label for this field, move to calculate value region
            if found_label_bbox:
                break # Stop searching for labels for this field

        if found_label_bbox:
            lx, ly, lw, lh = found_label_bbox
            
            
            initial_guess_x = lx + lw + 10 # Start 10px to the right of the label
            initial_guess_y = ly - 5 # A little above the label for robustness
            initial_guess_width = img_width - initial_guess_x - 20 # Almost to the right edge, with some margin
            initial_guess_height = lh + 20 # A little taller than the label for robustness

            # Clamp the initial guess to image boundaries
            initial_guess_x = max(0, initial_guess_x)
            initial_guess_y = max(0, initial_guess_y)
            initial_guess_width = max(10, min(initial_guess_width, img_width - initial_guess_x))
            initial_guess_height = max(10, min(initial_guess_height, img_height - initial_guess_y))

            guessed_value_bbox = (initial_guess_x, initial_guess_y, initial_guess_width, initial_guess_height)
            
            # Refine this guessed bounding box using text contours
            # Pass the original CV2 image optimized for OCR for contour detection
            refined_bbox = auto_refine_field_region(np.array(image_pil_ocr), guessed_value_bbox, padding=30)
            
            if refined_bbox:
                detected_regions[field_name] = refined_bbox
            else:
                st.warning(f"Label '{field_name}' found, but text content could not be refined. Using broader initial guess as fallback.")
                detected_regions[field_name] = guessed_value_bbox
        else:
            st.warning(f"Label for '{field_name}' not found. Skipping auto-detection for this field.")

    return detected_regions


# --- Streamlit UI ---
st.set_page_config(
    page_title="Document Scanner with Automatic Region Detection",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄 Document Scanner with Advanced OCR and Automatic Region Detection")
st.markdown("""
Upload an image of a document (e.g., a passport) to extract text using OCR.
This tool includes advanced preprocessing options to improve readability for unclear and noisy images.
It now **automatically detects and refines field regions** by identifying labels on the document.
**The OCR engine used is Tesseract's Deep Learning-based LSTM model (Tesseract 4+).**
""")

# --- Sidebar for Preprocessing Options ---
st.sidebar.header("Image Preprocessing Options")
st.sidebar.markdown("Adjust these settings to significantly improve OCR accuracy, especially for low-quality or noisy images.")

# Arrange options in a logical flow corresponding to preprocessing steps
st.sidebar.subheader("1. Basic Adjustments")
apply_grayscale = st.sidebar.checkbox("Convert to Grayscale", value=True,
                                      help="Converts the image to black and white tones. Highly recommended as most OCR is optimized for grayscale or binary images.")
scale_factor = st.sidebar.slider("Additional Image Scale Factor (for OCR)", 0.5, 3.0, 1.0, 0.1,
                                  help=f"Scales the image up or down *after* it's been resized to {REFERENCE_WIDTH}x{REFERENCE_HEIGHT}. This can further boost resolution for OCR.")

# Sliders for Brightness, Contrast, and Sharpening
st.sidebar.subheader("2. Brightness, Contrast & Sharpening")
brightness_factor = st.sidebar.slider("Brightness", 0.5, 2.0, 1.20, 0.05,
                                       help="Adjust the overall brightness of the image. A value of 1.0 is original brightness.")
contrast_factor = st.sidebar.slider("Contrast", 0.5, 2.0, 1.40, 0.05,
                                     help="Adjust the contrast of the image. A value of 1.0 is original contrast.")
sharpen_intensity = st.sidebar.slider("Sharpening Intensity", 0, 3, 3, 1,
                                       help="Controls the sharpness of edges in the image. Higher values make text crisper. Use cautiously as too much can introduce noise.")


st.sidebar.subheader("3. Denoising & Enhancement")
bilateral_filter_option = st.sidebar.checkbox("Apply Bilateral Filter (Smooth Edges)", value=True,
                                               help="Applies a non-linear filter that smooths images while preserving edges. Excellent for reducing noise without blurring text characters.")
noise_reduction_option = st.sidebar.selectbox(
    "General Noise Reduction Type",
    ('None', 'Median Blur', 'Fast N-Means Denoising'),
    index=1,
    help="Removes unwanted noise (speckles, grain) from the image. 'Median Blur' is generally fast and effective. 'Fast N-Means' is more advanced for complex noise. Use in conjunction with or instead of Bilateral Filter."
)
apply_contrast_enhance = st.sidebar.checkbox("Enhance Contrast (CLAHE)", value=False,
                                               help="Applies Contrast Limited Adaptive Histogram Equalization. Excellent for images with uneven lighting or low contrast areas, making text stand out.")


st.sidebar.subheader("4. Layout & Binarization")
apply_deskew = st.sidebar.checkbox("Auto-Deskew Image", value=False,
                                     help="Automatically detects and corrects slight rotations in the image. Highly recommended for scanned or photographed documents.")

# Option to toggle final thresholding for visual preview vs OCR input
apply_final_thresholding_for_ocr = st.sidebar.checkbox("Apply Final Thresholding (for OCR)", value=True,
                                                       help="Converts the image to pure black and white. Essential for optimal OCR. Uncheck to see the image before binarization.")
selected_threshold_type = 'none'
if apply_final_thresholding_for_ocr:
    threshold_option = st.sidebar.selectbox(
        "Thresholding Type",
        ('Simple Binary (Otsu)', 'Adaptive Gaussian', 'Adaptive Mean'),
        index=0,
        help="Converts the image to pure black and white. 'Adaptive Gaussian' and 'Adaptive Mean' are better for images with uneven lighting. 'Simple Binary (Otsu)' tries to find an optimal global threshold."
    )
    threshold_map_current = {
        'Simple Binary (Otsu)': 'simple',
        'Adaptive Mean': 'adaptive_mean',
        'Adaptive Gaussian': 'adaptive_gaussian'
    }
    selected_threshold_type = threshold_map_current[threshold_option]


st.sidebar.subheader("5. Post-Binarization Cleaning")
apply_morph_open = False
apply_morph_close = False
if apply_final_thresholding_for_ocr:
    apply_morph_open = st.sidebar.checkbox("Apply Morphological Opening", value=True,
                                           help="Removes small white noise (speckles) from the background, and breaks apart small connections. Best used after thresholding.")
    apply_morph_close = st.sidebar.checkbox("Apply Morphological Closing", value=False,
                                           help="Fills small holes inside text characters and closes small gaps between text components. Best used after thresholding.")


noise_reduction_map = {
    'None': 'none',
    'Median Blur': 'median_blur',
    'Fast N-Means Denoising': 'fast_n_means'
}

# Session state to store deskew angle if detected
if 'angle_detected_for_display' not in st.session_state:
    st.session_state.angle_detected_for_display = 0.0

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload an image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read image as bytes
    image_bytes = uploaded_file.getvalue()
    original_image_pil = Image.open(io.BytesIO(image_bytes))

    # Display Original Image with a sensible maximum width
    st.columns(1)[0].subheader("Original Image")
    st.columns(1)[0].image(original_image_pil, caption="Uploaded Image", use_column_width=True)


    # --- Preprocess Image ---
    with st.spinner(f"Applying preprocessing steps and resizing to {REFERENCE_WIDTH}x{REFERENCE_HEIGHT}px..."):
        temp_img_for_angle_detection = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if temp_img_for_angle_detection is not None:
            temp_img_for_angle_detection = cv2.resize(temp_img_for_angle_detection, (REFERENCE_WIDTH, REFERENCE_HEIGHT), interpolation=cv2.INTER_AREA)
            
            if apply_deskew:
                try:
                    _, temp_threshold_img = cv2.threshold(temp_img_for_angle_detection, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    coords = np.column_stack(np.where(temp_threshold_img > 0))
                    if len(coords) > 50:
                        angle = cv2.minAreaRect(coords)[-1]
                        if angle < -45: angle = -(90 + angle)
                        else: angle = -angle
                        st.session_state.angle_detected_for_display = angle
                        st.sidebar.info(f"Detected skew angle: {angle:.2f} degrees.")
                    else:
                        st.session_state.angle_detected_for_display = 0.0
                        st.sidebar.warning("Not enough text features found for effective deskewing. Skipping angle detection.")
                except Exception as e:
                    st.session_state.angle_detected_for_display = 0.0
                    st.sidebar.error(f"Error during deskew angle detection: {e}. Please ensure the image has clear text.")
            else:
                st.session_state.angle_detected_for_display = 0.0
        else:
            st.session_state.angle_detected_for_display = 0.0

        processed_image_for_ocr_cv2, processed_image_for_display_cv2 = preprocess_image(
            image_bytes,
            target_dims=(REFERENCE_WIDTH, REFERENCE_HEIGHT),
            grayscale=apply_grayscale,
            brightness_factor=brightness_factor,
            contrast_factor=contrast_factor,
            sharpen_intensity=sharpen_intensity,
            bilateral_filter=bilateral_filter_option,
            deskew=apply_deskew,
            threshold_type=selected_threshold_type,
            noise_reduction_type=noise_reduction_map[noise_reduction_option],
            morph_open=apply_morph_open,
            morph_close=apply_morph_close,
            scale_factor=scale_factor,
            contrast_enhance=apply_contrast_enhance
        )

    if processed_image_for_ocr_cv2 is None or processed_image_for_display_cv2 is None:
        st.error("Image preprocessing failed. Please try another image or adjust settings.")
        st.stop()

    if len(processed_image_for_display_cv2.shape) == 2:
        processed_image_pil_display = Image.fromarray(processed_image_for_display_cv2).convert('RGB')
    else:
        processed_image_pil_display = Image.fromarray(cv2.cvtColor(processed_image_for_display_cv2, cv2.COLOR_BGR2RGB))

    if len(processed_image_for_ocr_cv2.shape) == 3:
        processed_image_pil_ocr = Image.fromarray(cv2.cvtColor(processed_image_for_ocr_cv2, cv2.COLOR_BGR2GRAY))
    else:
        processed_image_pil_ocr = Image.fromarray(processed_image_for_ocr_cv2)
    
    if processed_image_pil_ocr.mode != 'L':
          processed_image_pil_ocr = processed_image_pil_ocr.convert('L')


    st.markdown("---")
    st.subheader(f"Preprocessed Image & Auto-Detected Field Regions ({processed_image_pil_display.width}x{processed_image_pil_display.height})")
    st.info(f"All uploaded images are automatically resized to **{REFERENCE_WIDTH}x{REFERENCE_HEIGHT} pixels** for consistent processing. Red boxes show the **auto-detected and refined text regions** based on identified labels.")
    st.warning("Automatic detection relies on finding specific labels (e.g., 'Surname', 'Passport No.'). If labels are unclear or missing, detection might be inaccurate.")


    # --- Automatic Region Detection ---
    with st.spinner("Automatically detecting field regions..."):
        final_ocr_regions = auto_detect_field_regions(processed_image_pil_ocr)

    # --- Draw Detected Regions ---
    visual_processed_image_pil_with_regions = processed_image_pil_display.copy()
    draw = ImageDraw.Draw(visual_processed_image_pil_with_regions)

    if not final_ocr_regions:
        st.warning("No regions could be automatically detected. Please ensure the document is clear.")
    else:
        for field_name, bbox in final_ocr_regions.items():
            rx, ry, rw, rh = bbox
            draw.rectangle([rx, ry, rx + rw, ry + rh], outline="red", width=2)
            try:
                draw.text((rx + 5, ry + 5), field_name, fill="red")
            except Exception: # Fallback if font is not found
                draw.text((rx + 5, ry + 5), field_name, fill="red")

    st.image(visual_processed_image_pil_with_regions, caption="Preprocessed Image with Auto-Detected Regions", use_column_width=True)

    st.markdown("---")
    st.subheader("OCR Result")

    # Button to trigger OCR
    if st.button("Perform OCR on All Auto-Detected Regions"):
        extracted_data = {}
        st.write("---")
        st.markdown("### Extracted Data by Field")
        progress_bar = st.progress(0)
        total_fields = len(final_ocr_regions)
        processed_fields_count = 0

        if not final_ocr_regions:
            st.info("No regions were detected, so no OCR can be performed.")
        else:
            # Loop through the final_ocr_regions to perform OCR
            for i, (field_name, bbox) in enumerate(final_ocr_regions.items()):
                st.subheader(f"**Field:** {field_name}")
                
                x, y, w, h = bbox
                # Ensure crop box coordinates are valid within the OCR image dimensions
                crop_x1 = max(0, x)
                crop_y1 = max(0, y)
                crop_x2 = min(processed_image_pil_ocr.width, x + w)
                crop_y2 = min(processed_image_pil_ocr.height, y + h)
                
                crop_box = (crop_x1, crop_y1, crop_x2, crop_y2)

                try:
                    # Ensure minimum crop dimensions for valid OCR
                    if (crop_x2 > crop_x1 + 5) and (crop_y2 > crop_y1 + 5): 
                        cropped_image_for_ocr = processed_image_pil_ocr.crop(crop_box)
                        st.image(cropped_image_for_ocr, caption=f"Cropped Image for {field_name} (OCR Input)", width=200)

                        # Get the specific OCR config for this field. Using PSM 6 for individual fields.
                        ocr_config = '--oem 3 --psm 6' # Default for all detected fields
                        
                        text = pytesseract.image_to_string(cropped_image_for_ocr, lang='eng', config=ocr_config).strip()

                        if text:
                            st.success(f"**Extracted Text for {field_name}:** `{text}`")
                            extracted_data[field_name] = text
                        else:
                            st.warning(f"No text extracted for {field_name}.")
                            extracted_data[field_name] = ""
                    else:
                        st.warning(f"Region for {field_name} is too small or invalid ({crop_x2-crop_x1}x{crop_y2-crop_y1}). Skipping OCR.")
                        extracted_data[field_name] = "Invalid Region (Too Small)"

                except Exception as e:
                    st.error(f"Error processing field '{field_name}': {e}")
                    extracted_data[field_name] = f"Error: {e}"
                
                processed_fields_count += 1
                progress_bar.progress(processed_fields_count / total_fields)
                st.markdown("---")

            st.success("Region-based OCR complete!")

            # Display all extracted data in a final summary
            st.markdown("### Summary of All Extracted Data")
            for field, value in extracted_data.items():
                st.write(f"**{field}:** {value if value else '*No data extracted*'} ")

            # Option to download the extracted data as CSV
            csv_data = io.StringIO()
            csv_writer = csv.writer(csv_data)
            csv_writer.writerow(["Field Name", "Extracted Value"])
            for field, value in extracted_data.items():
                csv_writer.writerow([field, value])
            
            st.download_button(
                label="Download Extracted Data (CSV)",
                data=csv_data.getvalue(),
                file_name="extracted_document_data.csv",
                mime="text/csv"
            )

else:
    st.info("Please upload an image to start scanning.")


st.markdown("---")
st.markdown("""
<small>Built with Streamlit and OpenCV. OCR powered by Tesseract.</small>
""", unsafe_allow_html=True)
