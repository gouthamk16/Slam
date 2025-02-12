import cv2
import time
from extract import VehicleTracker


def main():
    # Initialize tracker
    tracker = VehicleTracker()
    
    # Access RTSP stream
    cap = cv2.VideoCapture("data/test-forest.mp4")

    # Target FPS and frame timing
    target_fps = 60
    frame_time = 1/target_fps
    
    # Performance monitoring
    frame_count = 0
    start_time = time.time()

    i = 0
    
    while True:

        # if i < 100:
        #     i += 1
        #     continue

        # i += 1

        loop_start = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1920, 1080))

        frame_count += 1
        print(f"Frame: {frame_count}")

        # Process frame
        processed_frame, black_frame = tracker.process_frame(frame)

        cv2.imshow('Black Image', black_frame)
        
        # Calculate and display FPS
        
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"FPS: {fps:.2f}")
        
        # Maintain target FPS
        processing_time = time.time() - loop_start
        delay = max(1, int((frame_time - processing_time) * 1000))
        
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":   
    main()