import face_recognition
import cv2
import numpy as np
import os
import time
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
import threading # Used for signaling the recognition event

# --- 1. FastAPI App Initialization ---
app = FastAPI(
    title="Face Verification API",
    description="API for real-time face verification with live visual feedback for demos.",
    version="1.0.0"
)

# --- 2. Load Known Faces and Their Encodings ---
KNOWN_FACES_DIR = "known_people"
known_face_encodings = []
known_face_names = []

def load_known_faces():
    """Loads face encodings and names from the 'known_faces' directory."""
    global known_face_encodings, known_face_names
    print("Loading known faces...")
    
    if not os.path.exists(KNOWN_FACES_DIR):
        print(f"Error: Directory '{KNOWN_FACES_DIR}' not found. Please create it and add face images.")
        return

    for name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        if os.path.isdir(person_dir):
            for filename in os.listdir(person_dir):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(person_dir, filename)
                    try:
                        image = face_recognition.load_image_file(image_path)
                        # Ensure there's at least one face in the image
                        face_encodings_in_img = face_recognition.face_encodings(image)
                        if face_encodings_in_img:
                            known_face_encodings.append(face_encodings_in_img[0])
                            known_face_names.append(name)
                            print(f"  Loaded: {name} from {filename}")
                        else:
                            print(f"Warning: No face found in {image_path}. Skipping.")
                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}. Skipping.")
    print(f"Finished loading. Total known faces: {len(known_face_names)}")
    if not known_face_names:
        print("WARNING: No known faces loaded. Face recognition will not be possible.")
        print("Please ensure your 'known_faces' directory contains subfolders with face images.")


# Load faces when the application starts
load_known_faces()

# --- Global variable/event to control the recognition loop ---
# Used to signal when the target person has been found
recognition_event = threading.Event()
# A simple dictionary to hold the name of the person found (allows modification from nested scope)
found_person_name_holder = {"name": None} 

# --- API Endpoint for Face Verification ---
@app.get("/verify_face/{target_name}", 
         summary="Verify a specific person's face from live webcam with visual feedback",
         response_description="Verification status and recognized person's name")
async def verify_face_api(target_name: str):
    """
    Initiates a webcam feed to find and verify a specific person.
    A local OpenCV window will pop up showing the live feed with markings.
    
    - **Red Square:** If face is not identified or not the target.
    - **Green Square:** If the `target_name` is identified.
    
    The camera will run for a maximum of 30 seconds or until the `target_name` is verified.
    """
    
    # --- Reset state for a new verification request ---
    recognition_event.clear() 
    found_person_name_holder["name"] = None
    print(f"\nAPI request received: Verifying '{target_name}'...")

    video_capture = cv2.VideoCapture(0) # Open the default webcam (index 0)
    if not video_capture.isOpened():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not open webcam. Please ensure it's not in use by another application "
                   "(e.g., Zoom, other browser tabs, or another instance of this API)."
        )

    start_time = time.time()
    timeout_seconds = 30 # Max time to look for the person

    # Variables for performance optimization (process every N frames)
    frame_count_for_processing = 0
    frames_to_skip = 2 # Process every 3rd frame (0, 1, 2, then process)

    # Initialize lists to hold face data. These will persist the last detected faces
    # even if a frame is skipped for processing, so drawing can be consistent.
    current_face_locations = []
    current_face_encodings = []

    try:
        while not recognition_event.is_set() and (time.time() - start_time) < timeout_seconds:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to grab frame from webcam. Retrying...")
                # Add a small delay to prevent tight loop if camera is struggling
                time.sleep(0.1) 
                continue # Skip to the next iteration

            # Create a copy of the frame to draw on, leaving the original clean for processing
            display_frame = frame.copy() 

            # Only process face recognition on a subset of frames for performance
            if frame_count_for_processing % frames_to_skip == 0:
                # Resize frame to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                # Convert the image from BGR color (OpenCV) to RGB color (face_recognition)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                current_face_locations = face_recognition.face_locations(rgb_small_frame)
                current_face_encodings = face_recognition.face_encodings(rgb_small_frame, current_face_locations)
            
            frame_count_for_processing += 1 # Increment frame counter for skipping logic

            # --- Visual Markings Logic (Draw on every frame, using last processed data) ---
            if current_face_locations and current_face_encodings: # Only draw if faces were detected
                for (top, right, bottom, left), face_encoding in zip(current_face_locations, current_face_encodings):
                    # Scale back up face locations since the frame we detected in was 1/4 size
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    name_for_display = "Unauthorized" # Default display label
                    box_color = (0, 0, 255) # Default red color for unauthorized/unknown

                    # Perform comparison against known faces
                    if known_face_encodings:
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        recognition_threshold = 0.6 # Typical threshold for face_recognition library

                        if face_distances[best_match_index] < recognition_threshold:
                            recognized_name = known_face_names[best_match_index]
                            
                            # Check if the recognized face is our target
                            if recognized_name.lower() == target_name.lower():
                                name_for_display = f"Client {recognized_name} Authenticated"
                                box_color = (0, 255, 0) # Green for authenticated target
                                
                                # --- Found the target! Signal event and store name ---
                                print(f"Target '{recognized_name}' identified! Signalling completion.")
                                found_person_name_holder["name"] = recognized_name
                                recognition_event.set() # Stop the loop
                                
                            else:
                                # A known person, but not the target
                                name_for_display = f"Known: {recognized_name}"
                                box_color = (255, 165, 0) # Orange for other known faces
                    
                    # Draw a box around the face
                    cv2.rectangle(display_frame, (left, top), (right, bottom), box_color, 2)

                    # Draw a label with a name below the face
                    # Filled rectangle as background for the text for better readability
                    cv2.rectangle(display_frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(display_frame, name_for_display, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

            # Display the resulting image
            cv2.imshow('Face Verification Demo', display_frame)
            
            # Allow user to quit the display window by pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User manually quit the live verification display.")
                break # Exit the while loop

            # If the target was found and event was set in the drawing logic, break loop
            if recognition_event.is_set():
                break

    finally:
        # Release the webcam and close all OpenCV windows when done or on error
        video_capture.release()
        cv2.destroyAllWindows()
        print("Webcam released and OpenCV windows closed.")

    # --- API Response based on verification outcome ---
    if found_person_name_holder["name"]:
        # If the target was found, return success.
        # Adding a small delay for demo purposes so the user can see the green box.
        print(f"Verification successful for '{found_person_name_holder['name']}'.")
        time.sleep(3) # Hold green box for 3 seconds for demo effect
        return JSONResponse(
            content={"status": "Verified", "person": found_person_name_holder["name"]},
            status_code=status.HTTP_200_OK
        )
    else:
        # If timeout reached or loop broken without finding target
        print(f"Verification failed for '{target_name}'. Timeout or manual exit.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Target '{target_name}' not found within {timeout_seconds} seconds. Status: Unauthorized"
        )

# --- Run the FastAPI application ---
if __name__ == '__main__':
    import uvicorn
    # Use reload=True for development. For production, remove this and run with `uvicorn main:app --host 0.0.0.0 --port 8000`
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)