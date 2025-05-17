import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import tempfile
import os

# Function to find closest color name
def find_closest_color(rgb, color_df):
    r, g, b = rgb
    differences = []
    for index, row in color_df.iterrows():
        try:
            color_rgb = np.array([int(x) for x in row['R;G;B Dec'].split(';')])
            diff = np.sqrt(sum((rgb - color_rgb) ** 2))
            differences.append((diff, row['Color Name']))
        except (ValueError, KeyError):
            continue
    return min(differences, key=lambda x: x[0])[1] if differences else "Unknown Color"

# Load color dataset
@st.cache_data
def load_color_dataset():
    try:
        color_df = pd.read_csv('colours_rgb_shades.csv')
        if not all(col in color_df.columns for col in ['Color Name', 'R;G;B Dec']):
            raise ValueError("Invalid color dataset format. Required columns: 'Color Name', 'R;G;B Dec'")
        color_df = color_df[color_df['R;G;B Dec'].str.match(r'^\d+;\d+;\d+$', na=False)]
        if color_df.empty:
            raise ValueError("No valid RGB data found in the dataset")
        return color_df
    except FileNotFoundError:
        st.error("colours_rgb_shades.csv file not found")
        return None
    except Exception as e:
        st.error(f"Error loading colours_rgb_shades.csv: {str(e)}")
        return None

# Process a single frame for color detection
def process_frame(frame, color_df):
    # Convert to HSV
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define wider HSV ranges
    red_lower1 = np.array([0, 100, 50], np.uint8)    # Hue near 0
    red_upper1 = np.array([15, 255, 255], np.uint8)
    red_lower2 = np.array([165, 100, 50], np.uint8)  # Hue near 180
    red_upper2 = np.array([180, 255, 255], np.uint8)
    green_lower = np.array([35, 30, 30], np.uint8)    # Hue 35-85 for green
    green_upper = np.array([85, 255, 255], np.uint8)
    blue_lower = np.array([90, 100, 50], np.uint8)    # Hue 90-140 for blue
    blue_upper = np.array([140, 255, 255], np.uint8)

    # Create masks
    red_mask1 = cv2.inRange(hsvFrame, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsvFrame, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    # Dilate masks
    kernel = np.ones((5, 5), "uint8")
    red_mask = cv2.dilate(red_mask, kernel)
    green_mask = cv2.dilate(green_mask, kernel)
    blue_mask = cv2.dilate(blue_mask, kernel)

    # Set to store unique detected colors (color_name, rgb)
    detected_colors = set()

    # Debug: Sample HSV values at various points
    h, w, _ = hsvFrame.shape
    debug_info = "Sample HSV Values:\n"
    sample_points = [(0, 0), (w//2, h//2), (w-1, h-1)]
    for x, y in sample_points:
        hsv_value = hsvFrame[y, x]
        debug_info += f"Pixel ({x}, {y}): HSV {hsv_value}\n"

    # Fallback: Check if the entire image is a solid color
    avg_hsv = np.mean(hsvFrame, axis=(0, 1)).astype(int)
    debug_info += f"Average HSV of image: {avg_hsv}\n"
    if (red_lower1[0] <= avg_hsv[0] <= red_upper1[0] or red_lower2[0] <= avg_hsv[0] <= red_upper2[0]) and \
       avg_hsv[1] >= 100 and avg_hsv[2] >= 50:
        avg_rgb = np.mean(frame, axis=(0, 1)).astype(int)
        b, g, r = avg_rgb
        rgb = np.array([r, g, b])
        color_name = find_closest_color(rgb, color_df)
        detected_colors.add((color_name, tuple(rgb)))
    elif green_lower[0] <= avg_hsv[0] <= green_upper[0] and avg_hsv[1] >= 30 and avg_hsv[2] >= 30:
        avg_rgb = np.mean(frame, axis=(0, 1)).astype(int)
        b, g, r = avg_rgb
        rgb = np.array([r, g, b])
        color_name = find_closest_color(rgb, color_df)
        detected_colors.add((color_name, tuple(rgb)))
    elif blue_lower[0] <= avg_hsv[0] <= blue_upper[0] and avg_hsv[1] >= 100 and avg_hsv[2] >= 50:
        avg_rgb = np.mean(frame, axis=(0, 1)).astype(int)
        b, g, r = avg_rgb
        rgb = np.array([r, g, b])
        color_name = find_closest_color(rgb, color_df)
        detected_colors.add((color_name, tuple(rgb)))

    # Function to process contours and label colors
    def process_contours(mask, box_color):
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        debug_info_contours = f"Number of contours (area > 100): {len([c for c in contours if cv2.contourArea(c) > 100])}\n"
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Lowered threshold
                x, y, w, h = cv2.boundingRect(contour)
                roi = frame[y:y+h, x:x+w]
                if roi.size == 0:
                    continue
                avg_color = np.mean(roi, axis=(0, 1)).astype(int)
                b, g, r = avg_color
                rgb = np.array([r, g, b])
                color_name = find_closest_color(rgb, color_df)
                detected_colors.add((color_name, tuple(rgb)))
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                cv2.putText(frame, f"{color_name}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, box_color)
        return debug_info_contours

    # Process each color and collect debug info
    debug_info += "Red Mask:\n" + process_contours(red_mask, (0, 0, 255))
    debug_info += "Green Mask:\n" + process_contours(green_mask, (0, 255, 0))
    debug_info += "Blue Mask:\n" + process_contours(blue_mask, (255, 0, 0))

    return frame, detected_colors, debug_info

# Main app
def main():
    st.title("Multiple Color Detection Application")
    st.write("Upload an image or video to detect colors (red, green, blue regions).")

    # Load color dataset
    color_df = load_color_dataset()
    if color_df is None:
        return

    # File upload
    uploaded_file = st.file_uploader("Choose an image or video", type=['png', 'jpg', 'jpeg', 'mp4', 'avi'])

    if uploaded_file is not None:
        # Determine file type
        file_type = uploaded_file.type
        is_image = file_type.startswith('image')

        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4' if not is_image else '.png') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        try:
            if is_image:
                # Process as image
                image = cv2.imread(tmp_file_path)
                if image is None:
                    st.error("Failed to load image")
                    return
                processed_frame, detected_colors, debug_info = process_frame(image, color_df)
                st.image(processed_frame, channels="BGR", caption="Detected Colors")
                
                # Display detected colors below the image
                if detected_colors:
                    detected_colors = sorted(detected_colors, key=lambda x: x[0])
                    color_text = " | ".join([f"{color_name} {rgb}" for color_name, rgb in detected_colors])
                    st.write("**Detected Colors:**")
                    st.text(color_text)
                else:
                    st.write("**Detected Colors:** None")
                
                # Display debug info
                st.write("**Debug Information:**")
                st.text(debug_info)

            else:
                # Process as video
                cap = cv2.VideoCapture(tmp_file_path)
                if not cap.isOpened():
                    st.error("Failed to load video")
                    return

                stframe = st.empty()
                color_display = st.empty()
                debug_display = st.empty()
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    processed_frame, detected_colors, debug_info = process_frame(frame, color_df)
                    stframe.image(processed_frame, channels="BGR", caption="Detected Colors")
                    
                    # Display detected colors below the video frame
                    if detected_colors:
                        detected_colors = sorted(detected_colors, key=lambda x: x[0])
                        color_text = " | ".join([f"{color_name} {rgb}" for color_name, rgb in detected_colors])
                        color_display.write(f"**Detected Colors:** {color_text}")
                    else:
                        color_display.write("**Detected Colors:** None")
                    
                    # Display debug info
                    debug_display.write(f"**Debug Information:** {debug_info}")
                cap.release()

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

if __name__ == "__main__":
    main()
