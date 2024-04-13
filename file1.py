import cv2
import numpy as np
import os

# Step 1: Video Analysis
def extract_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

# Function to identify regions where the advertisement can be inserted
def identify_insertion_regions(frame):
    # Example: Define a single insertion region covering an area of the frame
    frame_height, frame_width, _ = frame.shape
    x1 = int(frame_width * 0.25)  # Example: 25% of frame width
    y1 = int(frame_height * 0.25)  # Example: 25% of frame height
    x2 = int(frame_width * 0.75)  # Example: 75% of frame width
    y2 = int(frame_height * 0.75)  # Example: 75% of frame height
    return [(x1, y1, x2, y2)]

# Step 2: Advertisement Image Processing
def preprocess_advertisement_image(ad_image):
    # Convert advertisement image to RGBA format (if not already in RGBA)
    if ad_image.shape[2] == 3:  # If the image is in RGB format
        ad_image = cv2.cvtColor(ad_image, cv2.COLOR_BGR2BGRA)  # Convert to RGBA format

    # Extract alpha channel from the advertisement image
    _, _, _, alpha = cv2.split(ad_image)

    # Normalize alpha channel values to range [0, 1]
    alpha = alpha / 255.0

    # Apply alpha blending to make the advertisement image transparent
    ad_image[:, :, 3] = cv2.GaussianBlur(alpha, (5, 5), 0)  # Apply Gaussian blur to alpha channel

    return ad_image


def track_objects(frame):
    # Convert the frame to grayscale for background subtraction
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform background subtraction to detect moving objects
    # Here, we use a simple background subtractor provided by OpenCV
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    fg_mask = bg_subtractor.apply(gray_frame)

    # Perform morphological operations to remove noise and enhance the mask
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through detected contours to find object positions
    object_positions = []
    for contour in contours:
        # Compute the bounding box of the contour
        x, y, w, h = cv2.boundingRect(contour)
        # Compute the center of the bounding box
        center_x = x + w // 2
        center_y = y + h // 2
        # Add the center coordinates to the list of object positions
        object_positions.append((center_x, center_y))

    return object_positions


def occlusion_detection(frame, insertion_regions, object_positions):
    occluded_regions = []
    for region in insertion_regions:
        x1, y1, x2, y2 = region
        for obj_x, obj_y in object_positions:
            # Check if the object position overlaps with the insertion region
            if x1 < obj_x < x2 and y1 < obj_y < y2:
                # If there's an overlap, add the insertion region to the list of occluded regions
                occluded_regions.append(region)
                break  # Break out of the loop for this insertion region
    return occluded_regions

def fill_occluded_regions(ad_image, occluded_regions):
    # Placeholder implementation: Mask out occluded regions of the advertisement image

    # Create a copy of the advertisement image to avoid modifying the original image
    filled_ad_image = ad_image.copy()

    # Loop through each occluded region
    for region in occluded_regions:
        x1, y1, x2, y2 = region
        # Fill the occluded region with a black mask (zero intensity)
        filled_ad_image[y1:y2, x1:x2] = 0

    return filled_ad_image

# Step 4: Advertisement Insertion
def insert_advertisement(frame, ad_image, insertion_regions, filled_ad_image):
    # Iterate through insertion regions
    for region in insertion_regions:
        x1, y1, x2, y2 = region
        
        # Resize the advertisement image to fit the insertion region
        ad_height, ad_width, _ = ad_image.shape
        ad_resize_width = x2 - x1
        ad_resize_height = y2 - y1
        resized_ad_image = cv2.resize(ad_image, (ad_resize_width, ad_resize_height))

        # Blend the advertisement image with the video frame using alpha blending
        alpha = resized_ad_image[:, :, 3] / 255.0  # Extract alpha channel and normalize to range [0, 1]
        beta = 1.0 - alpha  # Inverse of alpha

        # Extract the region of interest from the frame
        roi = frame[y1:y2, x1:x2]

        # Blend the advertisement image with the region of interest in the frame
        blended_roi = cv2.addWeighted(roi, 1.0, resized_ad_image[:, :, :3], alpha, 0.0)

        # Replace the region of interest in the frame with the blended result
        frame[y1:y2, x1:x2] = cv2.addWeighted(blended_roi, 1.0, roi, beta, 0.0)

    return frame


# Step 5: Quality Assessment and Optimization
def evaluate_quality(output_frames, sample_video_frames):
    # Convert sample video frames to grayscale for comparison
    sample_video_frames_gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in sample_video_frames]

    # Convert output frames to grayscale for comparison
    output_frames_gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in output_frames]

    # Calculate Structural Similarity Index (SSI) between each pair of sample and output frames
    ssi_scores = []
    for sample_frame, output_frame in zip(sample_video_frames_gray, output_frames_gray):
        ssi_score, _ = cv2.compare_SSIM(sample_frame, output_frame, full=True)
        ssi_scores.append(ssi_score)

    # Calculate the average SSI score across all frames
    average_ssi_score = np.mean(ssi_scores)

    # Normalize the SSI score to a range of [0, 1]
    normalized_ssi_score = (average_ssi_score + 1) / 2  # SSI ranges from -1 to 1, so add 1 and divide by 2

    return normalized_ssi_score


# Step 6: Documentation
def documentation():
    """
    Documentation for Computer Vision Solution

    Objective:
    The goal of this computer vision solution is to seamlessly insert a specified advertisement image
    into a given video, ensuring graceful handling of occlusions during the insertion process.

    Steps:
    1. Video Analysis: Extract frames from the input video.
    2. Advertisement Image Processing: Preprocess the advertisement image.
    3. Occlusion Handling: Track moving objects in the video frames and handle occlusions.
    4. Advertisement Insertion: Insert the preprocessed advertisement image into suitable regions in the video frames.
    5. Quality Assessment and Optimization: Evaluate the quality of the output frames and optimize as needed.
    6. Documentation: Provide documentation for the computer vision solution.

    Expectations:
    - The advertisement image is seamlessly integrated with the video.
    - Occlusions, such as hand movements, are gracefully handled during the insertion process.
    """
    pass


# Main function
def main():
    # Step 1: Video Analysis
    video_path = "C:/Users/LENOVO/Desktop/It_preneure/Java Script/AI/ML_Internship_Project/Input Video 2.mp4"
    frames = extract_frames(video_path)

    # Step 2: Advertisement Image Processing
    ad_image = cv2.imread('C:/Users/LENOVO/Desktop/It_preneure/Java Script/AI/ML_Internship_Project/Advertisement Image.jpg')
    preprocessed_ad_image = preprocess_advertisement_image(ad_image)

    # Create output directory if it doesn't exist
    output_dir = 'output_frames'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Step 3: Occlusion Handling and Step 4: Advertisement Insertion
    for i, frame in enumerate(frames):
        insertion_regions = identify_insertion_regions(frame)
        object_positions = track_objects(frame)
        occluded_regions = occlusion_detection(frame, insertion_regions, object_positions)
        filled_ad_image = fill_occluded_regions(preprocessed_ad_image, occluded_regions)

        # Insert advertisement
        frame_with_ad = insert_advertisement(frame, preprocessed_ad_image, insertion_regions, filled_ad_image)

        # Save the frame with advertisement
        cv2.imwrite(os.path.join(output_dir, f'frame_{i}.jpg'), frame_with_ad)

    # Step 5: Quality Assessment and Optimization
    # Placeholder implementation

    # Step 6: Documentation
    documentation()

if __name__ == "__main__":
    main()
