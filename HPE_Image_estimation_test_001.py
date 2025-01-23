import cv2
import mediapipe as mp

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Load an image or import the image from device to estimation
image_path = r"G:\Internship-24\Human Pose Estimation using Machine Learning\test23.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Unable to load image at path: {image_path}")

# Convert the image to RGB (Mediapipe requires RGB input)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform pose estimation
results = pose.process(image_rgb)

# Draw landmarks only (no lines)
if results.pose_landmarks:
    print("Pose landmarks detected!")

    # Get image dimensions
    h, w, c = image.shape

    # Extract and draw landmarks
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        print(f"Landmark {idx}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")
        cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)  # Draw blue keypoints

    # Optional: Draw connections on the image
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
    )

    # Display the output image
    cv2.imshow("Pose Landmarks", image)#for the landmark images
    cv2.imshow("Pose Drawing", annotated_image) # for the pose estimates images
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Release resources
pose.close()
