import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

# Define the frame rate of your video (fps)
fps = 15  # Adjust this value to match your video frame rate

# Define the scaling factor from pixels to meters (adjust as needed)
pixels_to_meters = 0.1  # Example: 1 pixel = 1 millimeter, so 0.001 meters

# Initialize a list to store average speeds at each instance
average_speeds = []

def calculate_average_speed(flow):
    # Calculate the magnitude (speed) of motion for each flow vector
    speed = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)

    # Compute the average speed across all flow vectors
    average_speed = np.mean(speed)

    # Convert average_speed to meters per second
    average_speed_mps = average_speed * fps * pixels_to_meters

    return average_speed_mps

def draw_flow_with_average_speed(img, flow, step=5):
    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x - fx, y - fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 5)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 5, (0, 255, 0))

    average_speed_mps = calculate_average_speed(flow)
    average_speeds.append(average_speed_mps)  # Append the average speed to the list

    # Display the overall speed in meters per second as text on the image
    cv2.putText(img_bgr, f"Average Speed: {average_speed_mps:.2f} cm/s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    return img_bgr

cap = cv2.VideoCapture('main.mp4')

suc, prev = cap.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

while True:
    i = 0
    suc, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # start time to calculate FPS
    start = time.time()

    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prevgray = gray

    # End time
    end = time.time()
    # calculate the FPS for the current frame detection
    fps = 1 / (end - start)

    print(f"{fps:.2f} FPS")
    print(i)
    i = i+1
    cv2.imshow('flow with average speed (m/s)', draw_flow_with_average_speed(gray, flow))

    key = cv2.waitKey(5)
    if key == ord('q'):
        break

    # Automatically close the window after 5 seconds (5000 milliseconds)
    if (time.time() - start) > 5:
        break

cap.release()
cv2.destroyAllWindows()

# Plot the average speeds
plt.plot(average_speeds)
plt.xlabel('Frame')
plt.ylabel('Average Speed (m/s)')
plt.title('Average Speed Over Time')
plt.grid(True)
plt.show()
