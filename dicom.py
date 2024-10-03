import cv2
import os
import time
import pydicom
import matplotlib.pyplot as plt
import numpy as np

# Path to the directory containing DICOM frames
dicom_folder = 'path_to_dicom_frames/'

# Get a list of all DICOM files in the folder
dicom_files = sorted([f for f in os.listdir(dicom_folder) if f.endswith('.dcm')])

# Define the frame rate (fps)
fps = 200  # Adjust this value as needed

# Scaling factor from pixels to meters
pixels_to_meters = 0.001  # Example: 1 pixel = 1 millimeter, so 0.001 meters

# Threshold for drawing arrows based on flow magnitude
flow_threshold = 1.5  # Set this to your desired threshold value

# Create a figure for live plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Initialize a fixed x-axis for the equalizer visualization
x_axis = np.arange(50)

# Function to calculate average speed between two frames
def calculate_average_speed(gradient1, gradient2):
    gradient_diff = np.sqrt((gradient1 - gradient2) ** 2)
    average_speed = np.mean(gradient_diff)
    average_speed_cm_s = average_speed * fps * pixels_to_meters
    return average_speed_cm_s

# Turn on interactive mode for live plotting
plt.ion()

# Initialize speed history for equalizer and speed vs. time plots
speed_history_equalizer = [0] * 50
speed_history_time = []

# Loop over DICOM frames
for i in range(1, len(dicom_files)):
    # Read the previous and current DICOM frames
    prev_dicom = pydicom.dcmread(os.path.join(dicom_folder, dicom_files[i - 1]))
    curr_dicom = pydicom.dcmread(os.path.join(dicom_folder, dicom_files[i]))

    # Extract pixel data from the DICOM files (convert to grayscale)
    prevgray = prev_dicom.pixel_array.astype(np.uint8)
    gray = curr_dicom.pixel_array.astype(np.uint8)

    # Start time to calculate FPS
    start = time.time()

    # Compute gradients for both frames
    gradient1 = cv2.Sobel(prevgray, cv2.CV_64F, 1, 1, ksize=5)
    gradient2 = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)

    # End time to calculate FPS
    end = time.time()

    # Calculate the FPS for the current frame detection
    processing_time = end - start
    if processing_time > 0.001:
        fps = 1 / processing_time
    else:
        fps = 30  # Set default FPS if time difference is too small

    # Calculate the average speed
    average_speed_cm_s = calculate_average_speed(gradient1, gradient2)

    # Append the average speed to the history lists
    speed_history_equalizer.append(average_speed_cm_s)
    if len(speed_history_equalizer) > 50:
        speed_history_equalizer.pop(0)
    speed_history_time.append(average_speed_cm_s)

    # Calculate optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.2, 6, 15, 3, 5, 1.2, 0)

    # Add arrows representing the flow direction on the image
    step = 20  # Increase step size to reduce the number of arrows
    h, w = gray.shape
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    # Scale the flow vectors to make arrows larger
    scale = 5
    fx = fx * scale
    fy = fy * scale

    # Calculate magnitude of flow vectors
    magnitude = np.sqrt(fx**2 + fy**2)

    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)

    # Create an RGB image from the grayscale DICOM frame for visualization
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Draw arrows only if magnitude exceeds the threshold
    for line, (dx, dy, mag) in zip(lines, zip(fx, fy, magnitude)):
        if mag > flow_threshold:  # Check if flow magnitude exceeds threshold
            start_point = tuple(line[0])
            end_point = (int(start_point[0] + dx), int(start_point[1] + dy))
            cv2.arrowedLine(img, start_point, end_point, (0, 0, 255), 2)

    # Display the overall speed as text on the image
    cv2.putText(img, f"Average Speed: {average_speed_cm_s:.2f} cm/s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 0, 0), 5)

    # Show the result frame
    cv2.imshow('frame', img)

    # Clear previous plots
    ax1.cla()
    ax2.cla()

    # Plot the equalizer-style bar with updated speed values
    ax1.bar(x_axis, speed_history_equalizer, color='blue')
    ax1.set_xlabel('Fixed X-Axis')
    ax1.set_ylabel('Average Speed (cm/s)')
    ax1.set_title('Equalizer Speed Plot')
    ax1.grid(True)

    # Plot the speed vs. time plot
    ax2.plot(speed_history_time, color='red')
    ax2.set_xlabel('Time (frames)')
    ax2.set_ylabel('Average Speed (cm/s)')
    ax2.set_title('Speed Over Time')

    plt.pause(0.01)

    key = cv2.waitKey(5)
    if key == ord('q'):
        break

# Turn off interactive mode
plt.ioff()

cv2.destroyAllWindows()
