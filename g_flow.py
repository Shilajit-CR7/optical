import cv2
import time
import matplotlib.pyplot as plt
import numpy as np

# Define the frame rate of your video (fps)
fps = 200  # Adjust this value to match your video frame rate

# Define the scaling factor from pixels to meters (adjust as needed)
pixels_to_meters = 0.001  # Example: 1 pixel = 1 millimeter, so 0.001 meters

# Initialize a list to store average speeds at each instance
average_speeds = []

# Initialize data for the time vs. average speed plot
time_values = []
average_speed_values = []

# Create a figure for live plotting
plt.figure(figsize=(10, 5))


def calculate_average_speed(gradient1, gradient2):
    # Calculate the difference in gradient between two frames
    gradient_diff = np.sqrt((gradient1 - gradient2) ** 2)

    # Compute the average speed as the mean gradient difference
    average_speed = np.mean(gradient_diff)

    # Convert average_speed to Cmeters per second
    average_speed_mps = average_speed * fps * pixels_to_meters

    return average_speed_mps


cap = cv2.VideoCapture('main.mp4')

suc, prev = cap.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

# Turn on interactive mode for live plotting
plt.ion()

while True:
    i = 0
    suc, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # start time to calculate FPS
    start = time.time()

    # Compute the gradient of the current frame
    gradient1 = cv2.Sobel(prevgray, cv2.CV_64F, 1, 1, ksize=5)
    gradient2 = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)

    # End time
    end = time.time()
    # calculate the FPS for the current frame detection
    # Calculate the FPS for the current frame detection
    if end - start > 0.001:  # Check if the time difference is greater than a small threshold
        fps = 1 / (end - start)
    else:
        fps = 30  # Set a default FPS value (e.g., 30 frames per second)

    print(f"{fps:.2f} FPS")
    print(i)
    i = i + 1
    # Calculate the average speed based on the gradient difference
    average_speed_mps = calculate_average_speed(gradient1, gradient2)
    average_speeds.append(average_speed_mps)  # Append the average speed to the list

    # Display the overall speed in meters per second as text on the image
    cv2.putText(img, f"Average Speed: {average_speed_mps:.2f} cm/s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 0, 0), 5)

    # Calculate flow direction using optical flow
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.2, 6, 15, 3, 5, 1.2, 0)

    # Add arrows representing the flow direction on the image
    step = 5  # Adjust arrow spacing as needed
    h, w = gray.shape
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)
    for line, (dx, dy) in zip(lines, zip(fx, fy)):
        start_point = tuple(line[0])
        end_point = (int(start_point[0] + dx), int(start_point[1] + dy))
        cv2.arrowedLine(img, start_point, end_point, (0, 0, 255), 1)

    cv2.imshow('frame', img)

    key = cv2.waitKey(5)
    if key == ord('q'):
        break

    # Automatically close the window after 5 seconds (5000 milliseconds)
    if (time.time() - start) > 5:
        break

    # Update the time vs. average speed plot data
    time_values.append(time.time())
    average_speed_values.append(average_speeds[-1])

    # Clear the previous plot and create a new one
    plt.clf()

    # Plot the live average speed data
    plt.subplot(121)
    plt.plot(average_speeds)
    plt.xlabel('Frame')
    plt.ylabel('Average Speed (m/s)')
    plt.title('Average Speed Over Time')
    plt.grid(True)

    # Plot the live time vs. average speed data
    plt.subplot(122)
    plt.plot(time_values, average_speed_values)
    plt.xlabel('Time (s)')
    plt.ylabel('Average Speed (m/s)')
    plt.title('Average Speed Over Time')
    plt.grid(True)

    # Pause to update the plot
    plt.pause(0.01)

# Turn off interactive mode
plt.ioff()

cap.release()
cv2.destroyAllWindows()
