import cv2
import time
import matplotlib.pyplot as plt
import numpy as np

# Define the frame rate of your video (fps)
fps = 4  # Adjust this value to match your video frame rate

# Define the scaling factor from pixels to meters (adjust as needed)
pixels_to_meters = 0.17857 # Example: 1 pixel = 1 millimeter, so 0.001 meters

# Create a figure for live plotting
plt.figure(figsize=(12, 12))

# Divide the frame into 8 quadrants (4 rows, 2 columns)
num_rows = 3
num_cols = 3


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


def calculate_principal_component(flow, mask):
    # Calculate the principal component using optical flow vectors within a mask
    u = flow[..., 0]
    v = flow[..., 1]

    # Apply the mask to retain only optical flow vectors within the quadrant
    u = u[mask != 0]
    v = v[mask != 0]

    # Calculate the principal component
    principal_component = np.array([np.mean(u), np.mean(v)])

    # Normalize the principal component vector
    principal_component /= np.linalg.norm(principal_component)

    return principal_component


cap = cv2.VideoCapture('main.mp4')

suc, prev = cap.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

# Turn on interactive mode for live plotting
plt.ion()

# Initialize lists for storing average speeds for all quadrants over time
quadrant_speeds_over_time = [[] for _ in range(num_rows * num_cols)]

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
            cv2.putText(img, f" Speed: {average_speed_cm_s:.2f} cm/s", (x1 + 10, y1 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            #cv2.putText(img, f"Quad {row + 1}-{col + 1} Speed: {average_speed_cm_s:.2f} cm/s", (x1 + 10, y1 + 30),
             #           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Draw a rectangle to represent the quadrant
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Calculate and display the principal component arrow at the center of the quadrant
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Create a mask for optical flow in the current quadrant
            mask = np.zeros_like(gray, dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255

            # Calculate optical flow
            flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.2, 6, 15, 3, 5, 1.2, 0)

            # Calculate the principal component for the current quadrant
            principal_component = calculate_principal_component(flow, mask)

            # Draw the principal component arrow at the center of the quadrant
            arrow_end = (int(center_x + principal_component[0] * 20), int(center_y + principal_component[1] * 20))
            cv2.arrowedLine(img, (center_x, center_y), arrow_end, (0, 255, 0), 4)

            # Append the average speed to the list for the current quadrant
            quadrant_speeds_over_time[row * num_cols + col].append(average_speed_cm_s)
            #cv2.waitKey(50)

    # Append the average speeds for all quadrants to the speed history list
        '''   speed_history_equalizer.append(quadrant_speeds)
    if len(speed_history_equalizer) > 50:
        speed_history_equalizer.pop(0)'''  # Remove the oldest speed values

    # Display the overall speed in meters per second as text on the image
    # cv2.putText(img, f"Average Speed: {average_speed_cm_s:.2f} cm/s", (10, 30),
    #           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)

    cv2.imshow('frame', img)

    key = cv2.waitKey(5)
    if key == ord('q'):
        # Plot thespeed history for all quadrants when 'q' is pressed
        pass

    if key == ord('q'):
        break

    prevgray = gray

# Turn off interactive mode
plt.ioff()

cap.release()
cv2.destroyAllWindows()
plt.clf()
for i in range(num_rows * num_cols):
            plt.subplot(num_rows, num_cols, i + 1)
            plt.plot(np.arange(len(quadrant_speeds_over_time[i])), quadrant_speeds_over_time[i], color='blue')
            plt.xlabel('Frames')
            plt.ylabel('Average Speed (cm/s)')
            plt.title(f'Quad {i + 1} Speed Over Time')

plt.show()