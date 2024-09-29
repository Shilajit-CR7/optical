import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

# Enable interactive mode for live plotting
plt.ion()

# Create figures and axes for live plotting
fig1, ax1 = plt.subplots()  # For the bar plot
bar = ax1.bar([0], [0], width=0.5)  # Create an empty bar
ax1.set_xlabel('Frame')
ax1.set_ylabel('Speed in Principal Direction (Normalized)')
ax1.set_title('Speed in Principal Direction Over Time')
ax1.set_ylim(0, 0.3)  # Set the y-axis limits to range from 0 to 1
ax1.grid(True)

fig2, ax2 = plt.subplots()  # For the time vs. average speed plot
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Average Speed (cm/s)')
ax2.set_title('Time vs. Average Speed')
ax2.grid(True)

# Initialize lists to store data
average_speeds = []
time_values = []
frame_count = 0

def calculate_speed_in_principal_direction(flow):
    # Calculate the magnitude (speed) of motion for each flow vector
    speed = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)

    # Calculate the direction of motion (in radians)
    angle = np.arctan2(flow[:, :, 1], flow[:, :, 0])

    # Compute the average speed in the principal direction
    principal_direction_speed = np.mean(np.abs(speed * np.cos(angle)))

    # Normalize the speed to the range [0, 1]
    principal_direction_speed_normalized = principal_direction_speed / 10.0  # You can adjust the scaling factor as needed

    return principal_direction_speed_normalized, principal_direction_speed

cap = cv2.VideoCapture('main.mp4')

suc, prev = cap.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

start_time = time.time()

while True:
    frame_count += 1
    suc, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Start time to calculate FPS
    start = time.time()

    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prevgray = gray

    # End time
    end = time.time()
    # Calculate the FPS for the current frame detection
    fps = 1 / (end - start)

    print(f"{fps:.2f} FPS")

    # Calculate the average speed for the current frame
    principal_direction_speed_normalized, principal_direction_speed = calculate_speed_in_principal_direction(flow)
    average_speeds.append(principal_direction_speed)
    time_values.append(time.time() - start_time)

    # Update the live bar plot with the normalized speed
    bar[0].set_height(principal_direction_speed_normalized)
    ax1.relim()
    ax1.autoscale_view()

    # Update the time vs. average speed plot
    ax2.plot(time_values, average_speeds, color='b')
    ax2.relim()
    ax2.autoscale_view()

    # Display the average speed as text on the video frame
    cv2.putText(img, f"Average Speed: {principal_direction_speed:.2f} cm/s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    fig1.canvas.flush_events()
    fig2.canvas.flush_events()

    cv2.imshow('flow with speed in principal direction (m/s)', img)

    key = cv2.waitKey(5)
    if key == ord('q'):
        break

    # Automatically close the window after processing all frames
    if frame_count >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
        break

cap.release()
cv2.destroyAllWindows()

# Disable interactive mode at the end to keep the plot open
plt.ioff()

# Plot the speeds in the principal direction (optional)
plt.plot(average_speeds)
plt.xlabel('Frame')
plt.ylabel('Speed in Principal Direction (m/s)')
plt.title('Speed in Principal Direction Over Time')
plt.grid(True)
plt.show()
