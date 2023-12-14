from Detector import detect
import cv2

def read_video_feed_and_get_contours(filename):
    # Create a VideoCapture object
    cap = cv2.VideoCapture(filename)
    
    # Read each frame from the video
    while True:
        # Capture frame-by-frame
        _, frame = cap.read()

        centers = detect(frame)
        # Draw the contours and their corresponding areas
        for center in centers:
            center = tuple(center.astype(int).ravel())
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
        
        # Display the resulting frame
        cv2.imshow('Video Feed', frame)

        # Check if the Esc key is pressed
        if cv2.waitKey(30) & 0xFF == 27:
            break

    # When done, release the video capture object
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    read_video_feed_and_get_contours("randomball.avi")