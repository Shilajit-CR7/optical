import cv2
import time
import matplotlib.pyplot as plt
import numpy as np

# Define the frame rate of your video (fps)
fps = 200  # Adjust this value to match your video frame rate

# Define the scaling factor from pixels to meters (adjust as needed)
pixels_to_meters = 0.001  # Example: 1 pixel = 1 millimeter, so 0.001 meters

# Create a figure for live plotting
plt.figure(figsize=(12, 12))

# Divide the frame into 4 quadrants
num_rows = 2
num_cols = 2

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

while True:
    suc, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate the size of each quadrant
    height, width = gray.shape
    quadrant_height = height // num_rows
    quadrant_width = width // num_cols

    # Initialize a list to store the average speeds for each quadrant
    quadrant_speeds = []

    for row in range(num_rows):
        for col in range(num_cols):
            # Define the coordinates for the current quadrant
            y1 = row * quadrant_height
            y2 = (row + 1) * quadrant_height
            x1 = col * quadrant_width
            x2 = (col + 1) * quadrant_width

            # Extract the current quadrant from the frames
            quadrant1 = prevgray[y1:y2, x1:x2]
            quadrant2 = gray[y1:y2, x1:x2]

            # Compute the gradient of the current quadrant
            gradient1 = cv2.Sobel(quadrant1, cv2.CV_64F, 1, 1, ksize=5)
            gradient2 = cv2.Sobel(quadrant2, cv2.CV_64F, 1, 1, ksize=5)

            # Calculate the average speed for the current quadrant
            average_speed_cm_s = calculate_average_speed(gradient1, gradient2)

            # Append the average speed to the list for visualization
            quadrant_speeds.append(average_speed_cm_s)

            # Display the average speed in meters per second as text on the quadrant
            cv2.putText(img, f"Quad {row+1}-{col+1} Speed: {average_speed_cm_s:.2f} cm/s", (x1 + 10, y1 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Draw a rectangle to represent the quadrant
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Calculate and display the principal component arrow at the center of the quadrant
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            principal_component = np.array([0.5, 0.5])  # Adjust arrow direction for each quadrant
            arrow_start = (center_x, center_y)
            arrow_end = (int(center_x + principal_component[0] * 20), int(center_y + principal_component[1] * 20))
            cv2.arrowedLine(img, arrow_start, arrow_end, (0, 255, 0), 2)

    # Append the average speeds for all quadrants to the speed history list
    speed_history_equalizer.append(quadrant_speeds)
    if len(speed_history_equalizer) > 50:
        speed_history_equalizer.pop(0)  # Remove the oldest speed values

    # Calculate and store gradient directions for the top 100 gradients in each quadrant
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

    # Plot the equalizer-style bar with updated speed values for all quadrants
    '''plt.clf()
    for i in range(num_rows * num_cols):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.bar(np.arange(len(speed_history_equalizer)), [speeds[i] for speeds in speed_history_equalizer], color='blue')
        plt.xlabel('Frames')
        plt.ylabel('Average Speed (cm/s)')
        plt.title(f'Quad {i+1} Speed Over Time')

    plt.pause(0.01)'''

    key = cv2.waitKey(5)
    if key == ord('q'):
        break

    prevgray = gray

# Turn off interactive mode
plt.ioff()

cap.release()
cv2.destroyAllWindows()
