import cv2
import time
import matplotlib.pyplot as plt
import numpy as np

# Define the frame rate of your video (fps)
fps = 15  # Adjust this value to match your video frame rate

# Define the scaling factor from pixels to meters (adjust as needed)
pixels_to_meters = 0.1  # Example: 1 pixel = 1 millimeter, so 0.001 meters

# Initialize a list to store average speeds at each instance
average_speeds = []

# Initialize data for the time vs. average speed plot
time_values = []
average_speed_values = []

# Create a figure for live plotting
plt.figure(figsize=(10, 5))

# Create a subplot for the equalizer-style plot
plt.subplot(121)
plt.xlabel('Time (frames)')
plt.ylabel('Average Speed (cm/s)')
plt.title('Average Speed Over Time')
plt.grid(True)

# Create a subplot for the speed vs. time plot
plt.subplot(122)
speed_history_time = []
speed_history_equalizer = [0] * 50
x_axis = np.arange(50)  # Adjust the range as needed
plt.bar(x_axis, speed_history_equalizer, color='blue')
plt.xlabel('Fixed X-Axis')
plt.ylabel('Average Speed (cm/s)')
plt.title('Average Speed Over Time')
plt.grid(True)

def calculate_average_speed(flow):
    # Calculate the magnitude (speed) of motion for each flow vector
    speed = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)

    # Compute the average speed across all flow vectors
    average_speed = np.mean(speed)

    # Convert average_speed to meters per second
    average_speed_mps = average_speed * fps * pixels_to_meters

    return average_speed_mps

def draw_flow_with_average_speed(img, flow, step=10):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for xi, yi, fxi, fyi in zip(x, y, fx, fy):
        start_point = (xi, yi)
        end_point = (xi + int(fxi), yi + int(fyi))
        color = (0, 255, 0)  # Green color for the arrow
        thickness = 4
        line_type = 8
        shift = 0
        cv2.arrowedLine(img_bgr, start_point, end_point, color, thickness, line_type, shift)

    average_speed_mps = calculate_average_speed(flow)
    average_speeds.append(average_speed_mps)  # Append the average speed to the list

    # Display the overall speed in meters per second as text on the image
    cv2.putText(img_bgr, f"Average Speed: {average_speed_mps:.2f} cm/s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 0, 255), 2)

    return img_bgr

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

    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.1, 10, 15, 3, 5, 1.2, 0)
    prevgray = gray

    # End time
    end = time.time()
    # calculate the FPS for the current frame detection
    fps = 1 / (end - start)

    print(f"{fps:.2f} FPS")
    print(i)
    i = i + 1
    cv2.imshow('flow with average speed (m/s)', draw_flow_with_average_speed(gray, flow))

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
    plt.plot(time_values, average_speed_values, color='red')
    plt.xlabel('Time (frames)')
    plt.ylabel('Average Speed (cm/s)')
    plt.title('Average Speed Over Time')

    # Plot the equalizer-style bar with updated speed values
    plt.subplot(122)
    speed_history_time.append(time_values[-1])
    speed_history_equalizer.append(average_speeds[-1])
    if len(speed_history_equalizer) > 50:
        speed_history_equalizer.pop(0)  # Remove the oldest speed value
    plt.bar(x_axis, speed_history_equalizer, color='blue')
    plt.xlabel('Fixed X-Axis')
    plt.ylabel('Average Speed (cm/s)')
    plt.title('Average Speed Over Time (Equalizer)')
    plt.grid(True)

    # Pause to update the plot
    plt.pause(0.01)

# Turn off interactive mode
plt.ioff()

cap.release()
cv2.destroyAllWindows()
