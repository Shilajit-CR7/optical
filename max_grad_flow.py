import cv2
import time
import matplotlib.pyplot as plt
import numpy as np

# Define the frame rate of your video (fps)
fps = 200  # Adjust this value to match your video frame rate

# Define the scaling factor from pixels to meters (adjust as needed)
pixels_to_meters = 0.001  # Example: 1 pixel = 1 millimeter, so 0.001 meters

# Create a figure for live plotting
plt.figure(figsize=(12, 6))

def calculate_average_speed(gradient1, gradient2):
    # Calculate the absolute difference in gradient between two frames
    gradient_diff = np.abs(gradient1 - gradient2)

    # Sort the gradient values in descending order and take the top 100 gradients
    sorted_gradients = np.sort(gradient_diff.flatten())[::-1][:100]

    # Compute the average speed as the mean of the top 100 gradients
    average_speed = np.mean(sorted_gradients)

    # Convert average_speed to centimeters per second
    average_speed_cm_s = average_speed * fps * pixels_to_meters

    return average_speed_cm_s

cap = cv2.VideoCapture('main.mp4')

suc, prev = cap.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

# Turn on interactive mode for live plotting
plt.ion()

# Initialize lists for plotting the speed history and gradient directions
speed_history_equalizer = [0] * 50
speed_history_time = []
gradient_directions = []

while True:
    suc, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # start time to calculate FPS
    start = time.time()

    # Compute the gradient of the current frame
    gradient1 = cv2.Sobel(prevgray, cv2.CV_64F, 1, 1, ksize=5)
    gradient2 = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)

    # End time
    end = time.time()

    # Calculate the FPS for the current frame detection
    if end - start > 0.001:  # Check if the time difference is greater than a small threshold
        fps = 1 / (end - start)
    else:
        fps = 30  # Set a default FPS value (e.g., 30 frames per second)

    # Calculate the average speed based on the top 100 gradients
    average_speed_cm_s = calculate_average_speed(gradient1, gradient2)

    # Append the average speed to the speed history list for the equalizer visualization
    speed_history_equalizer.append(average_speed_cm_s)
    if len(speed_history_equalizer) > 50:
        speed_history_equalizer.pop(0)  # Remove the oldest speed value

    # Append the average speed to the speed history list for speed vs. time plot
    speed_history_time.append(average_speed_cm_s)

    # Calculate and store gradient directions for the top 100 gradients
    gradient_diff = np.abs(gradient1 - gradient2)
    sorted_indices = np.argsort(gradient_diff.flatten())[::-1][:100]
    top_gradients = gradient_diff.flatten()[sorted_indices]
    gradient_directions = [np.arctan2(gradient1.flatten()[i], gradient2.flatten()[i]) for i in sorted_indices]

    # Display the overall speed in meters per second as text on the image
    cv2.putText(img, f"Average Speed: {average_speed_cm_s:.2f} cm/s", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)

    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.2, 6, 15, 3, 5, 1.2, 0)

    # Get the magnitude of the optical flow vectors
    magnitude = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)

    # Sort the flow vectors based on magnitude and take the top 100
    sorted_indices = np.argsort(magnitude.flatten())[::-1][:100]
    top_flow = flow.reshape(-1, 2)[sorted_indices]

    # Draw the top 100 optical flow vectors on the image
    for dx, dy in top_flow:
        start_point = (int(img.shape[1] / 2), int(img.shape[0] / 2))
        end_point = (int(start_point[0] + dx), int(start_point[1] + dy))
        cv2.arrowedLine(img, start_point, end_point, (0, 0, 255), 2)  # Increase arrow size and density

    cv2.imshow('frame', img)

    # Plot the equalizer-style bar with updated speed values
    plt.clf()
    plt.subplot(121)
    plt.bar(np.arange(len(speed_history_equalizer)), speed_history_equalizer, color='blue')
    plt.xlabel('Frames')
    plt.ylabel('Average Speed (cm/s)')
    plt.title('Average Speed Over Time')

    # Plot the speed vs. time plot
    plt.subplot(122)
    plt.plot(np.arange(len(speed_history_time)), speed_history_time, color='red')
    plt.xlabel('Frames')
    plt.ylabel('Average Speed (cm/s)')
    plt.title('Average Speed Over Time')

    plt.pause(0.01)

    key = cv2.waitKey(5)
    if key == ord('q'):
        break

    prevgray = gray

# Turn off interactive mode
plt.ioff()

cap.release()
cv2.destroyAllWindows()
