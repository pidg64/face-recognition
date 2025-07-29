import os
import cv2
import time
import numpy as np
import threading # Used for signaling the recognition event
import face_recognition

from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, status

app = FastAPI(
    title='Face Verification API',
    description='API for real-time face verification with live visual feedback.',
    version='1.0.0'
)

KNOWN_FACES_DIR = 'known_people'
known_face_encodings = []
known_face_names = []

def load_known_faces():
    """Loads face encodings and names from the 'known_faces' directory."""
    global known_face_encodings, known_face_names
    print('Loading known faces...')    
    if not os.path.exists(KNOWN_FACES_DIR):
        print(
            f'Error: Directory "{KNOWN_FACES_DIR}" not found. Please '
            'create it and add face images.')
        return
    for name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        if os.path.isdir(person_dir):
            for filename in os.listdir(person_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(person_dir, filename)
                    try:
                        image = face_recognition.load_image_file(image_path)
                        # Ensure there's at least one face in the image
                        face_encodings_in_img = face_recognition.face_encodings(
                            image
                        )
                        if face_encodings_in_img:
                            known_face_encodings.append(face_encodings_in_img[0])
                            known_face_names.append(name)
                            print(f'  Loaded: {name} from {filename}')
                        else:
                            print(
                                f'No face found in {image_path}. Skipping.'
                            )
                    except Exception as e:
                        print(
                            f'Error loading image {image_path}: {e}. Skipping.'
                        )
    print(f'Finished loading. Total known faces: {len(known_face_names)}')
    if not known_face_names:
        print(
            'No known faces loaded. Face recognition will not be possible.'
        )
        print(
            'Please ensure your "known_faces" directory contains subfolders '
            'with face images.'
        )

load_known_faces()

# Used to signal when the target person has been found
recognition_event = threading.Event()
found_person_name_holder = {'name': None} 

@app.get(
    '/verify_face/{target_name}', 
    summary="Verify a specific person's face from webcam with visual feedback",
    response_description="Verification status and recognized person's name"
)
async def verify_face_api(target_name: str):
    """
    Initiates a webcam feed to find and verify a specific person.
    A local OpenCV window will pop up showing the live feed with markings.
    
    - **Red Square:** If face is not identified or not the target.
    - **Green Square:** If the `target_name` is identified.
    
    The camera will run for a maximum of 30 seconds or until the `target_name` 
    is verified.
    """
    
    # --- Reset state for a new verification request ---
    recognition_event.clear() 
    found_person_name_holder['name'] = None
    print(f'\nAPI request received: Verifying "{target_name}"')
    video_capture = cv2.VideoCapture(0) # Open the default webcam (index 0)
    if not video_capture.isOpened():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                'Could not open webcam. Please ensure it is not in use by '
                'another application (e.g., Zoom, other browser tabs, or '
                'another instance of this API).'
            )
        )
    start_time = time.time()
    timeout_seconds = 30
    frame_count_for_processing = 0
    frames_to_skip = 2 # Process every 3rd frame (0, 1, 2, then process)
    current_face_locations = []
    current_face_encodings = []
    try:
        while not recognition_event.is_set() and (time.time() - start_time) < timeout_seconds:
            ret, frame = video_capture.read()
            if not ret:
                print('Failed to grab frame from webcam. Retrying...')
                # Add delay to prevent tight loop if camera is struggling
                time.sleep(0.1) 
                continue # Skip to the next iteration
            display_frame = frame.copy() 
            if frame_count_for_processing % frames_to_skip == 0:
                # Resize frame to 1/4 size for faster face recognition
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                current_face_locations = face_recognition.face_locations(
                    rgb_small_frame
                )
                current_face_encodings = face_recognition.face_encodings(
                    rgb_small_frame, current_face_locations
                )            
            frame_count_for_processing += 1
            if current_face_locations and current_face_encodings:
                for (top, right, bottom, left), face_encoding in zip(
                    current_face_locations, current_face_encodings
                ):
                    # Scale back up since the frame we detected in was 1/4 size
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    name_for_display = 'Unauthorized'
                    box_color = (0, 0, 255) # Red
                    # Perform comparison against known faces
                    if known_face_encodings:
                        face_distances = face_recognition.face_distance(
                            known_face_encodings, face_encoding
                        )
                        best_match_index = np.argmin(face_distances)
                        recognition_threshold = 0.6 # Typical threshold
                        if face_distances[best_match_index] < recognition_threshold:
                            recognized_name = known_face_names[best_match_index]
                            if recognized_name.lower() == target_name.lower():
                                name_for_display = 'Authenticated'
                                box_color = (0, 255, 0) # Green
                                print(
                                    f'Target "{recognized_name}" identified! '
                                    'Signalling completion.'
                                )
                                found_person_name_holder['name'] = recognized_name
                                recognition_event.set() # Stop the loop                                                   
                    # Draw a box around the face
                    cv2.rectangle(
                        display_frame, (left, top), (right, bottom), box_color, 2
                    )
                    cv2.rectangle(
                        display_frame,
                        (left, bottom - 35),
                        (right, bottom),
                        box_color,
                        cv2.FILLED
                    )
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(
                        display_frame,
                        name_for_display,
                        (left + 6, bottom - 6),
                        font,
                        0.6,
                        (255, 255, 255),
                        1
                    )
            cv2.imshow('Face Verification Demo', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('User manually quit the live verification display.')
                break
            if recognition_event.is_set():
                print('Target found. Waiting for any key before continuing.')
                cv2.waitKey(0)
                break
    finally:
        # Release the webcam and close all OpenCV windows when done or on error
        video_capture.release()
        cv2.destroyAllWindows()
        print('Webcam released and OpenCV windows closed.')
    if found_person_name_holder['name']:
        print(
            f'Verification successful for {found_person_name_holder["name"]}.'
        )
        return JSONResponse(
            content={
                'status': 'Verified', 'person': found_person_name_holder['name']
            },
            status_code=status.HTTP_200_OK
        )
    else:
        print(f'Verification failed for "{target_name}". Timeout or manual exit.')
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f'Target "{target_name}" not found within {timeout_seconds} '
                'seconds. Status: Unauthorized'
            )
        )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8002)