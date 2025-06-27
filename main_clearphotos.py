import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import pytesseract
import cv2
import numpy as np
import io
import csv
import re
from datetime import datetime


tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

try:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    pytesseract.get_tesseract_version()
except pytesseract.TesseractNotFoundError:
    st.error(f"Tesseract OCR engine not found at '{tesseract_path}'.")
    st.error("Please ensure it is installed and the path is correct, or added to your system's PATH environment variables.")
    st.stop()

# --- Constants ---
REFERENCE_WIDTH = 1280
REFERENCE_HEIGHT = 818 # Standard height for good aspect ratio with 1280 width

# --- Image Preprocessing Functions ---
def preprocess_image(image_bytes,
                     target_dims=None,
                     grayscale=True,
                     brightness_factor=1.10,
                     contrast_factor=1.40,
                     sharpen_intensity=0,
                     bilateral_filter=False,
                     deskew_angle=0.0, # Pass detected angle here
                     threshold_type='adaptive_gaussian',
                     noise_reduction_type='median_blur',
                     morph_open=False,
                     morph_close=False,
                     scale_factor=1.0,
                     contrast_enhance=False):
    """
    Applies a series of image preprocessing techniques based on provided parameters.
    Returns two versions of the processed image: one optimized for OCR and one for display.
    """
    image_np = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Could not load image. Please check the file format or if the file is corrupted.")
        return None, None

    # Apply overall target dimensions resizing if specified
    if target_dims and (img.shape[1] != target_dims[0] or img.shape[0] != target_dims[1]):
        img = cv2.resize(img, target_dims, interpolation=cv2.INTER_AREA)

    # Apply additional scale factor (for OCR resolution boost)
    if scale_factor != 1.0:
        img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale early if requested for OCR, affects subsequent operations
    # Create copies for OCR and display paths
    if grayscale:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed_image_for_ocr_cv2 = img_gray.copy()
        processed_image_for_display_cv2 = img_gray.copy() # Display also grayscale if selected
    else:
        processed_image_for_ocr_cv2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # OCR always gets grayscale
        processed_image_for_display_cv2 = img.copy() # Display gets color if selected

    # Ensure images are uint8 type
    processed_image_for_ocr_cv2 = processed_image_for_ocr_cv2.astype(np.uint8)
    processed_image_for_display_cv2 = processed_image_for_display_cv2.astype(np.uint8)

    # Brightness and Contrast (applied to both)
    # Convert display image to BGR for correct contrast/brightness if it's currently grayscale and 'grayscale' is false
    if not grayscale and len(processed_image_for_display_cv2.shape) == 2:
        processed_image_for_display_cv2 = cv2.cvtColor(processed_image_for_display_cv2, cv2.COLOR_GRAY2BGR)

    processed_image_for_ocr_cv2 = cv2.convertScaleAbs(processed_image_for_ocr_cv2, alpha=contrast_factor, beta=(brightness_factor - 1) * 127)
    processed_image_for_display_cv2 = cv2.convertScaleAbs(processed_image_for_display_cv2, alpha=contrast_factor, beta=(brightness_factor - 1) * 127)


    # Noise Reduction (applied to both)
    if noise_reduction_type == 'median_blur':
        processed_image_for_ocr_cv2 = cv2.medianBlur(processed_image_for_ocr_cv2, 5)
        # For display, apply median blur to each channel if color
        if len(processed_image_for_display_cv2.shape) == 3:
            processed_image_for_display_cv2 = cv2.medianBlur(processed_image_for_display_cv2, 5)
        else:
            processed_image_for_display_cv2 = cv2.medianBlur(processed_image_for_display_cv2, 5)
    elif noise_reduction_type == 'fast_n_means':
        if len(processed_image_for_ocr_cv2.shape) == 2: # Grayscale
            processed_image_for_ocr_cv2 = cv2.fastNlMeansDenoising(processed_image_for_ocr_cv2, None, h=30, templateWindowSize=7, searchWindowSize=21)
        else: # Color - should not happen for OCR image at this stage
            st.warning("OCR image was not grayscale for Fast N-Means. Denoising skipped for OCR.")

        if len(processed_image_for_display_cv2.shape) == 2: # Grayscale display
            processed_image_for_display_cv2 = cv2.fastNlMeansDenoising(processed_image_for_display_cv2, None, h=30, templateWindowSize=7, searchWindowSize=21)
        else: # Color display
            processed_image_for_display_cv2 = cv2.fastNlMeansDenoisingColored(processed_image_for_display_cv2, None, hColor=30, h=30, templateWindowSize=7, searchWindowSize=21)


    # Bilateral Filter (applied to both, only if grayscale)
    if bilateral_filter:
        if len(processed_image_for_ocr_cv2.shape) == 2:
            processed_image_for_ocr_cv2 = cv2.bilateralFilter(processed_image_for_ocr_cv2, 9, 75, 75)
        else:
            st.warning("OCR image was not grayscale for Bilateral Filter. Filtering skipped for OCR.")

        if len(processed_image_for_display_cv2.shape) == 2:
            processed_image_for_display_cv2 = cv2.bilateralFilter(processed_image_for_display_cv2, 9, 75, 75)
        else: # For color image, apply to each channel or convert to LAB and apply to L channel
            lab = cv2.cvtColor(processed_image_for_display_cv2, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_filtered = cv2.bilateralFilter(l, 9, 75, 75)
            processed_image_for_display_cv2 = cv2.cvtColor(cv2.merge([l_filtered, a, b]), cv2.COLOR_LAB2BGR)


    # Contrast Enhancement (CLAHE) - FIX FOR CV2.ERROR
    if contrast_enhance:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        # Apply CLAHE to the OCR image (which should always be grayscale at this point)
        if len(processed_image_for_ocr_cv2.shape) == 2: # Ensure it's grayscale
            processed_image_for_ocr_cv2 = clahe.apply(processed_image_for_ocr_cv2)
        else:
            # Fallback if OCR image is unexpectedly not grayscale, convert temporarily
            st.warning("OCR image was not grayscale for CLAHE. Converting temporarily for OCR.")
            temp_gray_ocr = cv2.cvtColor(processed_image_for_ocr_cv2, cv2.COLOR_BGR2GRAY)
            processed_image_for_ocr_cv2 = clahe.apply(temp_gray_ocr)

        # Apply CLAHE to the display image
        if len(processed_image_for_display_cv2.shape) == 2: # If it's already grayscale
            processed_image_for_display_cv2 = clahe.apply(processed_image_for_display_cv2)
        else: # If it's a color image (BGR), convert to LAB and apply CLAHE to L channel
            lab = cv2.cvtColor(processed_image_for_display_cv2, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_clahe = clahe.apply(l)
            merged_lab = cv2.merge([l_clahe, a, b])
            processed_image_for_display_cv2 = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)


    # Sharpening (applied to both)
    if sharpen_intensity > 0:
        K = sharpen_intensity
        kernel = np.array([[0, -K, 0],
                           [-K, 1 + 4 * K, -K],
                           [0, -K, 0]], dtype=np.float32)
        processed_image_for_ocr_cv2 = cv2.filter2D(processed_image_for_ocr_cv2, -1, kernel)
        
        if len(processed_image_for_display_cv2.shape) == 3: # Color image
            processed_image_for_display_cv2 = cv2.filter2D(processed_image_for_display_cv2, -1, kernel)
        else: # Grayscale image
            processed_image_for_display_cv2 = cv2.filter2D(processed_image_for_display_cv2, -1, kernel)


    # Deskew (applied to both if angle is non-zero)
    if deskew_angle != 0.0:
        (h, w) = processed_image_for_ocr_cv2.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, deskew_angle, 1.0)
        processed_image_for_ocr_cv2 = cv2.warpAffine(processed_image_for_ocr_cv2, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        if len(processed_image_for_display_cv2.shape) == 3: # If display image is color
            processed_image_for_display_cv2 = cv2.warpAffine(processed_image_for_display_cv2, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        else: # If display image is grayscale
            processed_image_for_display_cv2 = cv2.warpAffine(processed_image_for_display_cv2, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Final Thresholding for OCR (only applied to the OCR specific image, and only if grayscale)
    if len(processed_image_for_ocr_cv2.shape) == 2 and threshold_type != 'none':
        if threshold_type == 'simple':
            _, processed_image_for_ocr_cv2 = cv2.threshold(processed_image_for_ocr_cv2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif threshold_type == 'adaptive_mean':
            processed_image_for_ocr_cv2 = cv2.adaptiveThreshold(processed_image_for_ocr_cv2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        elif threshold_type == 'adaptive_gaussian':
            processed_image_for_ocr_cv2 = cv2.adaptiveThreshold(processed_image_for_ocr_cv2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Post-binarization morphological operations
        if morph_open:
            kernel = np.ones((2,2), np.uint8)
            processed_image_for_ocr_cv2 = cv2.morphologyEx(processed_image_for_ocr_cv2, cv2.MORPH_OPEN, kernel)
        if morph_close:
            kernel = np.ones((2,2), np.uint8)
            processed_image_for_ocr_cv2 = cv2.morphologyEx(processed_image_for_ocr_cv2, cv2.MORPH_CLOSE, kernel)
    
    # Ensure processed_image_for_display_cv2 matches 'grayscale' setting for final output
    if grayscale and len(processed_image_for_display_cv2.shape) == 3:
        processed_image_for_display_cv2 = cv2.cvtColor(processed_image_for_display_cv2, cv2.COLOR_BGR2GRAY)
    elif not grayscale and len(processed_image_for_display_cv2.shape) == 2:
        # If display was grayscale earlier but 'grayscale' is now false, convert back to BGR
        processed_image_for_display_cv2 = cv2.cvtColor(processed_image_for_display_cv2, cv2.COLOR_GRAY2BGR)


    return processed_image_for_ocr_cv2, processed_image_for_display_cv2


# --- MRZ Parsing and Cleaning Functions ---
def clean_mrz_line_basic(line):
    """
    Cleans an MRZ line by keeping only allowed characters and handling common OCR errors.
    This version is less aggressive in character replacement to avoid introducing errors
    if Tesseract's initial read was correct.
    """
    # Allowed characters in MRZ (uppercase letters, digits, and '<')
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
    
    # Common OCR confusions that are relatively safe to correct
    replacements = {
        'O': '0', 'D': '0', 'I': '1', 'L': '1',
        'Z': '2', 'S': '5', 'G': '6', 'B': '8',
        ' ': '<', '_': '<', '-': '<', # Replace spaces/underscores/hyphens with filler if in MRZ context
    }
    
    cleaned_chars = []
    for c in line.strip().upper():
        if c in allowed:
            cleaned_chars.append(c)
        elif c in replacements:
            cleaned_chars.append(replacements[c])
        # Discard characters not in allowed or common replacements
    
    cleaned = "".join(cleaned_chars)
    return cleaned

def parse_mrz_td3(mrz_full_raw_text):
    """
    Parses a TD3 (Passport) MRZ string into its constituent fields.
    This function expects a two-line MRZ format (44 characters per line).
    It includes more robust cleaning and validation.
    """
    parsed_data = {
        "MRZ Raw": mrz_full_raw_text,
        "Parse Status": "Failed",
        "Document Type": "N/A",
        "Surname": "N/A",
        "Given Names": "N/A",
        "Passport Number": "N/A",
    }

    # Clean and split lines
    lines = [clean_mrz_line_basic(line) for line in mrz_full_raw_text.strip().split('\n') if line.strip()]

    line1 = ""
    line2 = ""

    # Attempt to reconstruct two 44-character lines
    if len(lines) == 1 and len(lines[0]) >= 88:
        # If Tesseract merged lines, split it
        line1, line2 = lines[0][:44], lines[0][44:88]
    elif len(lines) >= 2:
        # Take the first two non-empty, cleaned lines
        line1, line2 = lines[0][:44], lines[1][:44]
    else:
        parsed_data["Parse Status"] = "Failed: Not enough MRZ lines or invalid format detected."
        return parsed_data
    
    # Pad to exactly 44 characters (crucial for precise slicing according to TD3 standard)
    line1 = line1.ljust(44, '<')[:44]
    line2 = line2.ljust(44, '<')[:44]

    parsed_data["MRZ Raw"] = line1 + "\n" + line2 # Update MRZ Raw with cleaned and re-formatted lines

    try:
        # Line 1 Parsing
        parsed_data["Document Type"] = line1[0]


        name_segment = line1[5:44]
        if '<<' in name_segment:
            surname_raw, given_names_raw = name_segment.split('<<', 1)
            parsed_data["Surname"] = re.sub(r'<+', ' ', surname_raw).strip()
            parsed_data["Given Names"] = re.sub(r'<+', ' ', given_names_raw).strip()
        else:
            parsed_data["Surname"] = re.sub(r'<+', ' ', name_segment).strip()
            parsed_data["Given Names"] = "N/A"

        # Line 2 Parsing
        parsed_data["Passport Number"] = line2[0:9].replace('<', '').strip()
        
        # Check digit for Passport Number (char 9) - not stored as a field, but used for validation
        passport_num_check_digit = line2[9]

        parsed_data["Nationality"] = line2[10:13]
        
        # Date of Birth (YYMMDD) and its check digit
        dob_check_digit = line2[19]

        
        
        # Determine parse status
        essential_fields = [
        
            parsed_data["Surname"], parsed_data["Passport Number"],
        ]
        
        if all(field not in ["N/A", "", "<", "<<", "<<<"] for field in essential_fields):
            parsed_data["Parse Status"] = "Success"
        else:
            missing_fields = [k for k, v in parsed_data.items() if k in ["Document Type", "Surname", "Passport Number"] and v in ["N/A", "", "<", "<<", "<<<"]]
            if missing_fields:
                parsed_data["Parse Status"] = f"Partial Success: Missing/Empty core fields: {', '.join(missing_fields)}"
            else:
                parsed_data["Parse Status"] = "Partial Success: Some fields might need review."

    except IndexError as e:
        parsed_data["Parse Status"] = f"Failed: MRZ structure mismatch (IndexError: {e}). Raw: '{mrz_full_raw_text}'"
    except Exception as e:
        parsed_data["Parse Status"] = f"Failed: General parsing error: {e}. Raw: '{mrz_full_raw_text}'"
    
    return parsed_data

def parse_dob(dob_text):
    """
    Attempts to parse various date formats into a standardized 'DD MONYYYY' format.
    Prioritizes YYMMDD (MRZ format), then common free-form formats.
    """
    dob_text = dob_text.strip().replace(' ', '').upper()

    # Try YYMMDD format first (e.g., 850117) - common in MRZ
    if len(dob_text) == 6 and dob_text.isdigit():
        try:
            # Heuristic for 2-digit years: if > current year % 100, assume 19xx, else 20xx
            current_year_last_two_digits = datetime.now().year % 100
            year_prefix = '19' if int(dob_text[0:2]) > current_year_last_two_digits else '20'
            full_year_str = year_prefix + dob_text[0:2]
            return datetime.strptime(full_year_str + dob_text[2:6], "%Y%m%d").strftime("%d %b %Y").upper()
        except ValueError:
            pass # Continue to next format

    # Try DD MONYYYY (e.g., 17 JAN 1985, 23 APR 1984)
    # Using regex to capture parts of the date string
    match = re.search(r'(\d{1,2})\s*(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s*(\d{2,4})', dob_text, re.IGNORECASE)
    if match:
        day = int(match.group(1))
        month_abbr = match.group(2).upper()
        year = int(match.group(3))
        
        # Handle 2-digit years
        if year < 100:
            year = 1900 + year if year > 50 else 2000 + year # Heuristic for 2-digit years
        
        try:
            date_obj = datetime(year, datetime.strptime(month_abbr, "%b").month, day)
            return date_obj.strftime("%d %b %Y").upper()
        except ValueError:
            pass

    # Try YYYY-MM-DD or similar numeric only (e.g., 1985-01-17, 19840423)
    match_numeric_date = re.search(r'(\d{4})[/-]?(\d{2})[/-]?(\d{2})', dob_text)
    if match_numeric_date:
        try:
            return datetime.strptime(match_numeric_date.group(0).replace('-', ''), "%Y%m%d").strftime("%d %b %Y").upper()
        except ValueError:
            pass

    # Final fallback if parsing fails
    return dob_text if dob_text else "N/A"


def auto_refine_field_region(image_for_detection_cv2, base_bbox, padding=60):
    """
    Automatically refines a bounding box to tightly fit text contours within a search area.
    This function is robust, incorporating dilation and intelligent merging of text contours.
    """
    img_h, img_w = image_for_detection_cv2.shape[:2]
    bx, by, bw, bh = base_bbox

    # Define a generous search crop area based on the initial bbox
    sx1 = max(0, bx - padding)
    sy1 = max(0, by - padding)
    sx2 = min(img_w, bx + bw + padding)
    sy2 = min(img_h, by + bh + padding)

    search_crop = image_for_detection_cv2[sy1:sy2, sx1:sx2].copy()

    # Handle cases where search_crop might be empty or uniformly black/white
    if search_crop.size == 0 or (search_crop.max() == search_crop.min() and search_crop.max() in [0, 255]):
        fallback_padding = 30
        fb_x1 = max(0, bx - fallback_padding)
        fb_y1 = max(0, by - fallback_padding)
        fb_x2 = min(img_w, bx + bw + fallback_padding)
        fb_y2 = min(img_h, by + bh + fallback_padding)
        return (fb_x1, fb_y1, fb_x2 - fb_x1, fb_y2 - fb_y1)

    # Convert to grayscale if it's a color image (though it should already be grayscale from preprocess_image)
    if len(search_crop.shape) == 3:
        search_crop = cv2.cvtColor(search_crop, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding to convert to binary (black text on white background)
    if search_crop.max() > 1 and search_crop.min() < 255: # Only threshold if not already binary
        search_crop_binary = cv2.adaptiveThreshold(search_crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 15)
        # THRESH_BINARY_INV means text becomes white on black background, which is often better for findContours
    else:
        search_crop_binary = search_crop.copy() # Assume it's already binary

    # Ensure black text on white background (invert if necessary for contour detection)
    if np.mean(search_crop_binary) < 127: # If mean is low, it's mostly black (white text on black)
        search_crop_binary = cv2.bitwise_not(search_crop_binary) # Invert to black text on white

    # Dilate characters to connect broken text segments and get more robust contours
    dilation_kernel_width = max(5, int(bh * 0.7)) # Horizontal dilation for connecting words
    dilation_kernel_height = max(3, int(bh * 0.25)) # Vertical dilation for connecting lines/parts of characters
    dilation_kernel = np.ones((dilation_kernel_height, dilation_kernel_width), np.uint8)
    dilated_search_crop = cv2.dilate(search_crop_binary, dilation_kernel, iterations=4)

    # Find contours (RETR_EXTERNAL retrieves only the outermost contours)
    contours, _ = cv2.findContours(dilated_search_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidate_bboxes = []
    min_text_height = max(8, int(bh * 0.4)) # Minimum height for a text line
    max_text_height = int(bh * 2.0) # Maximum height to avoid large non-text contours

    for cnt in contours:
        x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
        area = w_c * h_c
        
        # Filter contours based on size and aspect ratio typical for text
        if area < 50: # Small areas are usually noise
            continue
        if h_c < min_text_height or h_c > max_text_height:
            continue
        aspect_ratio = w_c / h_c if h_c > 0 else 0
        if not (0.01 < aspect_ratio < 100.0): # Very thin or very wide (e.g., vertical lines)
            continue

        # Convert contour coordinates back to global image coordinates
        global_x = sx1 + x_c
        global_y = sy1 + y_c
        candidate_bboxes.append((global_x, global_y, w_c, h_c))

    if not candidate_bboxes:
        # If no valid text contours found, return a slightly padded original bbox as a fallback
        fallback_padding = 30
        fb_x1 = max(0, bx - fallback_padding)
        fb_y1 = max(0, by - fallback_padding)
        fb_x2 = min(img_w, bx + bw + fallback_padding)
        fb_y2 = min(img_h, by + bh + fallback_padding)
        return (fb_x1, fb_y1, fb_x2 - fb_x1, fb_y2 - fb_y1)

    # Sort bounding boxes (primarily by Y, then by X) to process lines of text
    candidate_bboxes.sort(key=lambda b: (b[1], b[0]))

    # Merge overlapping or horizontally close bounding boxes that are vertically aligned
    merged_bboxes = []
    if candidate_bboxes:
        current_group = list(candidate_bboxes[0]) # Initialize with the first box
        for i in range(1, len(candidate_bboxes)):
            next_box = candidate_bboxes[i]
            current_x1, current_y1, current_w, current_h = current_group
            next_x1, next_y1, next_w, next_h = next_box

            # Define thresholds for merging (consider horizontal gap and vertical overlap)
            horizontal_gap_threshold = max(35, int(current_h * 1.5)) # Allows for spaces between words/segments
            vertical_overlap_min_ratio = 0.85 # Minimum vertical overlap to consider same line

            dist_x = next_x1 - (current_x1 + current_w)
            
            overlap_y1 = max(current_y1, next_y1)
            overlap_y2 = min(current_y1 + current_h, next_y1 + next_h)
            vertical_overlap_height = max(0, overlap_y2 - overlap_y1)

            # Check if boxes are close horizontally AND overlap significantly vertically
            if (dist_x < horizontal_gap_threshold and dist_x > - (next_w * 0.7) and # Allows for some overlap if very close
                (vertical_overlap_height / min(current_h, next_h)) >= vertical_overlap_min_ratio):
                
                # Merge: extend current_group to include next_box
                current_group[0] = min(current_x1, next_x1)
                current_group[1] = min(current_y1, next_y1)
                current_group[2] = max(current_x1 + current_w, next_x1 + next_w) - current_group[0]
                current_group[3] = max(current_y1 + current_h, next_y1 + next_h) - current_group[1]
            else:
                # If not merging, add current group to merged list and start a new group
                merged_bboxes.append(tuple(current_group))
                current_group = list(next_box)
        merged_bboxes.append(tuple(current_group)) # Add the last group

    # Select the "best" merged bounding box (e.g., the one closest to the original center, or largest)
    best_merged_bbox = None
    max_score = -1
    base_center_y = by + bh / 2
    base_center_x = bx + bw / 2

    for m_x, m_y, m_w, m_h in merged_bboxes:
        area = m_w * m_h
        m_center_y = m_y + m_h / 2
        m_center_x = m_x + m_w / 2

        # Score based on area, proximity to original center, and deviation from original size
        vertical_dist_penalty = abs(m_center_y - base_center_y)
        horizontal_dist_penalty = abs(m_center_x - base_center_x) * 0.02 
        
        score = area - (vertical_dist_penalty * 20) - (horizontal_dist_penalty * 0.1)

        # Penalize if the merged box doesn't overlap enough with the original base box horizontally
        overlap_x1 = max(bx, m_x)
        overlap_x2 = min(bx + bw, m_x + m_w)
        overlap_width = max(0, overlap_x2 - overlap_x1)
        if overlap_width < bw * 0.65: # Require at least 65% horizontal overlap
            continue

        # Penalize if the merged box is significantly larger than the base box
        width_overage_penalty = max(0, m_w - bw) * 0.9
        height_overage_penalty = max(0, m_h - bh) * 0.7
        score -= (width_overage_penalty + height_overage_penalty)

        # Strong penalty for extremely large (likely incorrect) merged boxes
        if m_w > bw * 1.5 or m_h > bh * 2.0:
            score -= area * 0.5

        if score > max_score and m_w > 0 and m_h > 0: # Ensure valid dimensions
            best_merged_bbox = (m_x, m_y, m_w, m_h)
            max_score = score
    
    # Final adjustment: add a small buffer around the best merged bbox
    if best_merged_bbox:
        buffer_x = 10 
        buffer_y = 10
        rx, ry, rw, rh = best_merged_bbox
        
        final_x1 = max(0, rx - buffer_x)
        final_y1 = max(0, ry - buffer_y)
        final_x2 = min(img_w, rx + rw + buffer_x)
        final_y2 = min(img_h, ry + rh + buffer_y)
        
        final_w = max(0, final_x2 - final_x1)
        final_h = max(0, final_y2 - final_y1)

        # Fallback if the final bbox calculation results in an invalid size
        if final_w <= 0 or final_h <= 0 or final_w < 20 or final_h < 10:
            fallback_padding = 30
            final_x1 = max(0, bx - fallback_padding)
            final_y1 = max(0, by - fallback_padding)
            final_x2 = min(img_w, bx + bw + fallback_padding)
            final_y2 = min(img_h, by + bh + fallback_padding)
            final_w = max(0, final_x2 - final_x1)
            final_h = max(0, final_y2 - final_y1)
            if final_w <= 0 or final_h <= 0: # Last resort: return original base_bbox
                return base_bbox
        return (final_x1, final_y1, final_w, final_h)
    else:
        # If no best merged bbox found, return a slightly padded base bbox
        fallback_padding = 30
        fb_x1 = max(0, bx - fallback_padding)
        fb_y1 = max(0, by - fallback_padding)
        fb_x2 = min(img_w, bx + bw + fallback_padding)
        fb_y2 = min(img_h, by + bh + fallback_padding)
        return (fb_x1, fb_y1, fb_x2 - fb_x1, fb_y2 - fb_y1)


# --- Streamlit UI and Logic ---
# Page configuration MUST be the first Streamlit command
st.set_page_config(
    page_title="Passport MRZ & Date of Birth OCR",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄 Passport MRZ & Date of Birth OCR")
st.markdown("""
Upload a passport image to extract the MRZ and Date of Birth using OCR.
This tool uses advanced preprocessing and MRZ-specific cleaning for best accuracy.
""")

# --- Sidebar for Preprocessing Options ---
st.sidebar.header("Image Preprocessing Options")
st.sidebar.subheader("Basic Adjustments")
apply_grayscale = st.sidebar.checkbox("Convert to Grayscale", value=True,
                                      help="Converts the image to black and white tones. Highly recommended as most OCR is optimized for grayscale or binary images.")
scale_factor = st.sidebar.slider("Image Scale Factor", 0.5, 3.0, 1.0, 0.1,
                                  help=f"Scales the image up or down BEFORE other processing. Can boost resolution for OCR.")

st.sidebar.subheader("Brightness, Contrast & Sharpening")
brightness_factor = st.sidebar.slider("Brightness", 0.5, 2.0, 1.20, 0.05,
                                       help="Adjust the overall brightness of the image. 1.0 is original.")
contrast_factor = st.sidebar.slider("Contrast", 0.5, 2.0, 1.40, 0.05,
                                     help="Adjust the contrast of the image. 1.0 is original.")
sharpen_intensity = st.sidebar.slider("Sharpening Intensity", 0, 3, 3, 1,
                                       help="Controls the sharpness of edges. Higher values make text crisper.")

st.sidebar.subheader("Denoising & Enhancement")
bilateral_filter_option = st.sidebar.checkbox("Apply Bilateral Filter (Edge-Preserving Smooth)", value=True, # Changed default to False, often not needed
                                               help="Smooths images while preserving edges. Good for noise reduction without blurring text, but can be slow.")
noise_reduction_option = st.sidebar.selectbox(
    "Noise Reduction Type",
    ('None', 'Median Blur', 'Fast N-Means Denoising'),
    index=2, # Default to Fast N-Means Denoising
    help="Removes unwanted noise. 'Median Blur' is fast. 'Fast N-Means' is advanced for complex noise."
)
apply_contrast_enhance = st.sidebar.checkbox("Enhance Contrast (CLAHE - Adaptive)", value=False, # Default to True
                                               help="Applies Contrast Limited Adaptive Histogram Equalization. Excellent for images with uneven lighting.")


st.sidebar.subheader("Layout & Binarization")
apply_deskew = st.sidebar.checkbox("Auto-Deskew Image", value=False, # Changed default to True
                                    help="Automatically corrects slight rotations in the image. Highly recommended.")
apply_final_thresholding_for_ocr = st.sidebar.checkbox("Apply Final Thresholding (Binarization for OCR)", value=True,
                                                       help="Converts the image to pure black and white. Essential for optimal OCR.")

selected_threshold_type = 'none'
if apply_final_thresholding_for_ocr:
    threshold_option = st.sidebar.selectbox(
        "Thresholding Type",
        ('Simple Binary (Otsu)', 'Adaptive Gaussian', 'Adaptive Mean'), # Reordered, Simple Binary often best
        index=0,
        help="Converts image to black/white. 'Adaptive' types are better for uneven lighting."
    )
    threshold_map_current = {
        'Simple Binary (Otsu)': 'simple',
        'Adaptive Mean': 'adaptive_mean',
        'Adaptive Gaussian': 'adaptive_gaussian'
    }
    selected_threshold_type = threshold_map_current[threshold_option]

st.sidebar.subheader("Post-Binarization Cleaning")
apply_morph_open = False
apply_morph_close = False
if apply_final_thresholding_for_ocr:
    apply_morph_open = st.sidebar.checkbox("Morphological Opening (Remove small noise)", value=True,
                                           help="Removes small white noise from background and breaks small connections.")
    apply_morph_close = st.sidebar.checkbox("Morphological Closing (Fill small holes)", value=False,
                                           help="Fills small holes inside characters and closes small gaps.")

noise_reduction_map = {
    'None': 'none',
    'Median Blur': 'median_blur',
    'Fast N-Means Denoising': 'fast_n_means'
}

# --- Region Definitions ---
# These are initial estimated coordinates for passport fields.
# For MRZ, auto_refine_field_region will be used.
# For Date of Birth, these will be default values for manual adjustment.
# These coordinates are based on the 'pp.jpg' image at 1280x818 resolution.
default_base_regions = {
    "Date of Birth": (400, 320, 400, 60), # Adjusted slightly for DOB
    "MRZ": (20, 680, 1240, 120), # Broad region for auto-detection of MRZ, slightly adjusted height
}

# Session state to store deskew angle if detected
if 'angle_detected_for_display' not in st.session_state:
    st.session_state.angle_detected_for_display = 0.0

# Session state for manual date region adjustments
if 'manual_regions' not in st.session_state:
    st.session_state.manual_regions = {
        "Date of Birth": default_base_regions["Date of Birth"],
    }

# --- OCR Configuration per field (Page Segmentation Mode - PSM) ---
# PSM 6: Assume a single uniform block of text.
# PSM 7: Treat the image as a single text line. (Might be better for DOB)
# PSM 8: Treat the image as a single word. (If DOB is known to be always one word/number)
# OEM 3: Use the default Tesseract engine with LSTM.
ocr_psm_configs = {
    "Date of Birth": '--oem 3 --psm 6',
    "MRZ": '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<', # Strict whitelist for MRZ characters
}

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload an image (PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    original_image_pil = Image.open(io.BytesIO(image_bytes))

    st.columns(1)[0].subheader("Original Image")
    st.columns(1)[0].image(original_image_pil, caption="Uploaded Image", use_column_width=True)

    with st.spinner(f"Applying preprocessing steps..."):
        # For deskewing, detect angle once on a standardized size
        temp_img_for_angle_detection = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        
        # Ensure the image for angle detection is not None and has valid dimensions
        if temp_img_for_angle_detection is not None and temp_img_for_angle_detection.shape[0] > 0 and temp_img_for_angle_detection.shape[1] > 0:
            # Resize temp image for angle detection to REFERENCE_WIDTH, REFERENCE_HEIGHT consistently
            temp_img_for_angle_detection = cv2.resize(temp_img_for_angle_detection, (REFERENCE_WIDTH, REFERENCE_HEIGHT), interpolation=cv2.INTER_AREA)

            if apply_deskew:
                try:
                    # Apply Otsu's thresholding for better text/background separation to find contours
                    _, temp_threshold_img = cv2.threshold(temp_img_for_angle_detection, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # Find coordinates of non-zero (text) pixels
                    coords = np.column_stack(np.where(temp_threshold_img > 0))

                    if len(coords) > 50: # Ensure sufficient data points for a reliable angle calculation
                        angle = cv2.minAreaRect(coords)[-1]
                        if angle < -45: # Adjust angle to be between -45 and 45 degrees
                            angle = -(90 + angle)
                        else:
                            angle = -angle
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
            st.sidebar.warning("Could not process image for deskewing. Image might be invalid or too small.")


        # Process the image with all selected options
        processed_image_for_ocr_cv2, processed_image_for_display_cv2 = preprocess_image(
            image_bytes,
            target_dims=(REFERENCE_WIDTH, REFERENCE_HEIGHT), # Target dims applied first
            grayscale=apply_grayscale,
            brightness_factor=brightness_factor,
            contrast_factor=contrast_factor,
            sharpen_intensity=sharpen_intensity,
            bilateral_filter=bilateral_filter_option,
            deskew_angle=st.session_state.angle_detected_for_display, # Pass the detected angle
            threshold_type=selected_threshold_type,
            noise_reduction_type=noise_reduction_map[noise_reduction_option],
            morph_open=apply_morph_open,
            morph_close=apply_morph_close,
            scale_factor=scale_factor, # Additional scale factor applied after initial resize
            contrast_enhance=apply_contrast_enhance
        )

    if processed_image_for_ocr_cv2 is not None and processed_image_for_display_cv2 is not None:
        # Convert processed CV2 images to PIL for Streamlit display and Tesseract input
        if len(processed_image_for_display_cv2.shape) == 2: # If grayscale for display
            processed_image_pil_display = Image.fromarray(processed_image_for_display_cv2).convert('RGB')
        else: # If color for display
            processed_image_pil_display = Image.fromarray(cv2.cvtColor(processed_image_for_display_cv2, cv2.COLOR_BGR2RGB))
        
        # Ensure image for OCR is grayscale for Tesseract
        if len(processed_image_for_ocr_cv2.shape) == 3:
            processed_image_pil_ocr = Image.fromarray(cv2.cvtColor(processed_image_for_ocr_cv2, cv2.COLOR_BGR2GRAY))
        else:
            processed_image_pil_ocr = Image.fromarray(processed_image_for_ocr_cv2)
        
        if processed_image_pil_ocr.mode != 'L': # Tesseract typically works best with 8-bit grayscale images ('L' mode)
            processed_image_pil_ocr = processed_image_pil_ocr.convert('L')
        
        st.subheader("Manual Region Adjustment for Date of Birth")
        st.info(f"The image is now at **{processed_image_pil_display.width}x{processed_image_pil_display.height} pixels**. Adjust coordinates (X, Y, Width, Height) below for Date of Birth. If the DOB is already good, no need to adjust.")
        
        with st.expander("Adjust Date of Birth Region"):
            st.session_state.manual_regions["Date of Birth"] = (
                st.number_input("DOB X", value=st.session_state.manual_regions["Date of Birth"][0], min_value=0, max_value=processed_image_pil_display.width, key="dob_x"),
                st.number_input("DOB Y", value=st.session_state.manual_regions["Date of Birth"][1], min_value=0, max_value=processed_image_pil_display.height, key="dob_y"),
                st.number_input("DOB Width", value=st.session_state.manual_regions["Date of Birth"][2], min_value=10, max_value=processed_image_pil_display.width, key="dob_w"),
                st.number_input("DOB Height", value=st.session_state.manual_regions["Date of Birth"][3], min_value=10, max_value=processed_image_pil_display.height, key="dob_h")
            )
        
        st.markdown("---")
        st.subheader(f"Preprocessed Image & Field Regions ({processed_image_pil_display.width}x{processed_image_pil_display.height})")
        st.info("Gray box: initial MRZ area. Red box: auto-refined MRZ. Blue box: Date of Birth (manual).")

        visual_image_with_regions = processed_image_pil_display.copy()
        draw = ImageDraw.Draw(visual_image_with_regions)
        try:
            # Load a font for drawing text on the image, defaulting to system font if Arial isn't found
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()

        final_ocr_regions = {}

        # 1. Draw and refine MRZ region
        mrz_base_bbox = default_base_regions["MRZ"]
        # Ensure base bbox coordinates are within the bounds of the processed image
        mb_x = max(0, mrz_base_bbox[0])
        mb_y = max(0, mrz_base_bbox[1])
        mb_w = min(mrz_base_bbox[2], processed_image_pil_ocr.width - mb_x)
        mb_h = min(mrz_base_bbox[3], processed_image_pil_ocr.height - mb_y)
        mb_w = max(1, mb_w) # Ensure width is at least 1
        mb_h = max(1, mb_h) # Ensure height is at least 1
        mrz_base_bbox = (mb_x, mb_y, mb_w, mb_h)

        draw.rectangle([mrz_base_bbox[0], mrz_base_bbox[1], mrz_base_bbox[0] + mrz_base_bbox[2], mrz_base_bbox[1] + mrz_base_bbox[3]], outline="gray", width=1)
        draw.text((mrz_base_bbox[0] + 5, mrz_base_bbox[1] + 5), "MRZ (Base Auto)", fill="gray", font=font)

        # Refine MRZ bounding box using the auto_refine_field_region function
        refined_mrz_bbox = auto_refine_field_region(np.array(processed_image_pil_ocr), mrz_base_bbox, padding=40)
        final_ocr_regions["MRZ"] = refined_mrz_bbox

        mrz_rx, mrz_ry, mrz_rw, mrz_rh = refined_mrz_bbox
        mrz_rw = max(1, mrz_rw) # Ensure width is at least 1
        mrz_rh = max(1, mrz_rh) # Ensure height is at least 1
        draw.rectangle([mrz_rx, mrz_ry, mrz_rx + mrz_rw, mrz_ry + mrz_rh], outline="red", width=2)
        draw.text((mrz_rx + 5, mrz_ry + 5), "MRZ (Auto-Refined)", fill="red", font=font)

        # 2. Draw Date of Birth region (manual)
        dob_bbox = st.session_state.manual_regions["Date of Birth"]
        # Ensure DOB bbox coordinates are within the bounds of the processed image
        mx = max(0, dob_bbox[0])
        my = max(0, dob_bbox[1])
        mw = min(dob_bbox[2], processed_image_pil_ocr.width - mx)
        mh = min(dob_bbox[3], processed_image_pil_ocr.height - my)
        mw = max(1, mw) # Ensure width is at least 1
        mh = max(1, mh) # Ensure height is at least 1
        dob_bbox = (mx, my, mw, mh)
        final_ocr_regions["Date of Birth"] = dob_bbox

        draw.rectangle([mx, my, mx + mw, my + mh], outline="blue", width=2)
        draw.text((mx + 5, my + 5), "Date of Birth (Manual)", fill="blue", font=font)

        st.image(visual_image_with_regions, caption="Preprocessed Image with Regions", use_container_width=True)

        st.markdown("---")
        st.subheader("OCR Result")

        if st.button("Perform OCR on Defined Regions"):
            extracted_data = {}
            st.write("---")
            st.markdown("### Extracted Data by Field")

            progress_bar = st.progress(0)
            total_fields = 2 # Date of Birth and MRZ for progress tracking

            # --- Date of Birth OCR ---
            processed_fields_count = 0
            bbox = final_ocr_regions["Date of Birth"]
            x, y, w, h = bbox
            # Create a robust crop box
            crop_box = (max(0, x), max(0, y), min(processed_image_pil_ocr.width, x + w), min(processed_image_pil_ocr.height, y + h))
            
            st.subheader("**Field:** Date of Birth")
            if (crop_box[2] > crop_box[0] + 5) and (crop_box[3] > crop_box[1] + 5): # Check if crop box has valid dimensions
                cropped_image_for_ocr = processed_image_pil_ocr.crop(crop_box)
                st.image(cropped_image_for_ocr, caption="Cropped Image for Date of Birth (OCR Input)", width=200)
                
                ocr_config_dob = ocr_psm_configs.get("Date of Birth", '--oem 3 --psm 6')
                raw_dob_text = pytesseract.image_to_string(cropped_image_for_ocr, lang='eng', config=ocr_config_dob).strip()
                
                if raw_dob_text:
                    parsed_dob = parse_dob(raw_dob_text)
                    st.success(f"**Extracted Raw Text:** `{raw_dob_text}`")
                    st.success(f"**Parsed Date of Birth:** `{parsed_dob}`")
                    extracted_data["Date of Birth"] = parsed_dob
                else:
                    st.warning("No text extracted for Date of Birth.")
                    extracted_data["Date of Birth"] = "N/A"
            else:
                st.warning(f"Region for Date of Birth is too small or invalid ({w}x{h}). Skipping OCR.")
                extracted_data["Date of Birth"] = "Invalid Region (Too Small)"
            
            processed_fields_count += 1
            progress_bar.progress(processed_fields_count / total_fields)

            st.markdown("---")

            # --- MRZ OCR and Parsing ---
            st.subheader("**Field:** MRZ (Overall Region OCR)")
            mrz_bbox = final_ocr_regions["MRZ"]
            mrz_x, mrz_y, mrz_w, mrz_h = mrz_bbox
            # Create a robust crop box for MRZ
            mrz_crop_box = (max(0, mrz_x), max(0, mrz_y), min(processed_image_pil_ocr.width, mrz_x + mrz_w), min(processed_image_pil_ocr.height, mrz_y + mrz_h))
            
            mrz_raw_text = ""
            if (mrz_crop_box[2] > mrz_crop_box[0] + 5) and (mrz_crop_box[3] > mrz_crop_box[1] + 5): # Check if crop box has valid dimensions
                cropped_mrz_image = processed_image_pil_ocr.crop(mrz_crop_box)
                st.image(cropped_mrz_image, caption="Cropped Image for MRZ (Main OCR Input)", width=400)
                
                ocr_config_mrz = ocr_psm_configs.get("MRZ", '--oem 3 --psm 6')
                mrz_raw_text = pytesseract.image_to_string(cropped_mrz_image, lang='eng', config=ocr_config_mrz).strip()
                st.info(f"**Raw MRZ Text (from auto-refined MRZ box):** `{mrz_raw_text}`")
            else:
                st.warning("Auto-refined MRZ region is too small or invalid. MRZ parsing skipped.")
                mrz_raw_text = "Invalid MRZ Region"
            
            extracted_data["MRZ Raw"] = mrz_raw_text

            if mrz_raw_text and mrz_raw_text != "Invalid MRZ Region":
                parsed_mrz_data = parse_mrz_td3(mrz_raw_text)
                # Update extracted_data only with new keys from parsed_mrz_data to avoid overwriting "Date of Birth"
                for key, value in parsed_mrz_data.items():
                    if key not in extracted_data or key == "MRZ Raw" or key == "Parse Status": # Always update MRZ raw and status
                        extracted_data[key] = value

                st.success(f"**MRZ Parsing Status:** {parsed_mrz_data['Parse Status']}")
                
                st.write(f"""
                **{{
                Document Type: '{extracted_data.get('Document Type', 'N/A')}',
                Surname: '{extracted_data.get('Surname', 'N/A')}',
                Given Names: '{extracted_data.get('Given Names', 'N/A')}',
                Passport Number: '{extracted_data.get('Passport Number', 'N/A')}',
                Nationality: '{extracted_data.get('Nationality', 'N/A')}',
                }}**
                """)
                
            processed_fields_count += 1
            progress_bar.progress(processed_fields_count / total_fields)
            
            st.success("OCR and Parsing complete!")

            # --- Summary of All Extracted Data ---
            st.markdown("### Summary of All Extracted Data")
            # Define the order of fields for summary display
            summary_fields_order = [
                "Date of Birth", # From manual DOB OCR and parsing
                "Document Type", "Surname", "Given Names",
                "Passport Number",
                "MRZ Raw", "Parse Status" # Raw MRZ and its parse status
            ]
            for field_name in summary_fields_order:
                value = extracted_data.get(field_name, None)
                if value is not None:
                    st.write(f"**{field_name}:** {value if value else '*No data extracted*'} ")

            # --- Download Extracted Data as CSV ---
            csv_data = io.StringIO()
            csv_writer = csv.writer(csv_data)
            csv_writer.writerow(summary_fields_order) # Write header row
            row_values = [str(extracted_data.get(field_name, "")).replace('\n', ' | ') for field_name in summary_fields_order]
            csv_writer.writerow(row_values) # Write data row
            st.download_button(
                label="Download Extracted Data (CSV)",
                data=csv_data.getvalue(),
                file_name="extracted_passport_data.csv",
                mime="text/csv"
            )
    else:
        st.error("Image preprocessing failed. Please try another image or adjust settings in the sidebar.")
else:
    st.info("Please upload an image to start scanning.")

st.markdown("---")
st.markdown("""
<small>Built with Streamlit and OpenCV. OCR powered by Tesseract.</small>
""", unsafe_allow_html=True)