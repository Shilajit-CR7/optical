import cv2
import time
import matplotlib.pyplot as plt
import numpy as np

# Define the frame rate of your video (fps)
fps = 10# Adjust this value to match your video frame rate

# Define the scaling factor from pixels to meters (adjust as needed)
pixels_to_meters = 0.001  # Example: 1 pixel = 1 millimeter, so 0.001 meters

# Create a figure for live plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Initialize a fixed x-axis for the equalizer visualization
x_axis = np.arange(50)  # Adjust the range as needed

def calculate_average_speed(gradient1, gradient2):
    # Calculate the difference in gradient between two frames
    gradient_diff = np.sqrt((gradient1 - gradient2) ** 2)

    # Compute the average speed as the mean gradient difference
    average_speed = np.mean(gradient_diff)

    # Convert average_speed to centimeters per second
    average_speed_cm_s = average_speed * fps * pixels_to_meters

    return average_speed_cm_s

cap = cv2.VideoCapture('hh.mp4')

suc, prev = cap.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

# Turn on interactive mode for live plotting
plt.ion()

# Initialize the speed history list for the equalizer visualization and speed vs. time plot
speed_history_equalizer = [0] * 50
speed_history_time = []

while True:
    suc, img = cap.read()
    if not suc:
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # start time to calculate FPS
    start = time.time()

    # Compute the gradient of the current frame
    gradient1 = cv2.Sobel(prevgray, cv2.CV_64F, 1, 1, ksize=5)
    gradient2 = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)

    # End time
    end = time.time()

    # Calculate the FPS for the current frame detection
    processing_time = end - start
    if processing_time > 0.001:  # Check if the time difference is greater than a small threshold
        fps = 1 / processing_time
    else:
        fps = 30  # Set a default FPS value (e.g., 30 frames per second)

    # Calculate the average speed based on the gradient difference
    average_speed_cm_s = calculate_average_speed(gradient1, gradient2)

    # Append the average speed to the speed history list for the equalizer visualization
    speed_history_equalizer.append(average_speed_cm_s)
    if len(speed_history_equalizer) > 50:
        speed_history_equalizer.pop(0)  # Remove the oldest speed value

    # Append the average speed to the speed history list for speed vs. time plot
    speed_history_time.append(average_speed_cm_s)

    # Display the overall speed in meters per second as text on the image
    cv2.putText(img, f"Average Speed: {average_speed_cm_s:.2f} cm/s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 0, 0), 5)

    # Calculate flow direction using optical flow
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.2, 6, 15, 3, 5, 1.2, 0)

    # Add arrows representing the flow direction on the image
    step = 20  # Increase step size to reduce the number of arrows
    h, w = gray.shape
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    # Scale the flow vectors to make arrows larger
    scale = 5  # Increase the scale value to make arrows bigger
    fx = fx * scale
    fy = fy * scale

    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)
    for line, (dx, dy) in zip(lines, zip(fx, fy)):
        start_point = tuple(line[0])
        end_point = (int(start_point[0] + dx), int(start_point[1] + dy))
        cv2.arrowedLine(img, start_point, end_point, (0, 0, 255), 2)  # Increase thickness to 2

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

cap.release()
cv2.destroyAllWindows()
