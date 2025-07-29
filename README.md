# Face Verification API

This project provides a real-time face verification API using FastAPI and OpenCV. It captures video from a webcam, identifies faces, and verifies them against a directory of known individuals.

## Features

- **Real-time Face Verification**: Uses a live webcam feed to verify a person's identity.
- **Visual Feedback**: An OpenCV window displays the camera feed with colored boxes around detected faces:
  - **Green**: The target person is successfully authenticated.
  - **Orange**: A known person is detected, but they are not the target.
  - **Red**: An unknown face is detected.
- **Simple REST API**: A single API endpoint to initiate the verification process.
- **Timeout**: The verification process automatically times out after 30 seconds if the target is not found.

## Prerequisites

- Python 3.8+
- A webcam connected to your system.
- The `dlib` library, which has its own system dependencies (like CMake and a C++ compiler). Please refer to the [dlib installation guide](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf) for your specific OS.

## Installation

1. **Clone the repository (if applicable)**

   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
   ```
2. **Create a virtual environment (recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. **Install the required Python packages**
   It's recommended to create a `requirements.txt` file with the following content:

   ```
   fastapi
   uvicorn[standard]
   numpy
   opencv-python
   face_recognition
   ```

   Then, install them using pip:

   ```bash
   pip install -r requirements.txt
   ```
4. **Set up the Known Faces Directory**
   The application needs a directory of images to learn from.

   - Create a directory named `known_people` in the same folder as `main.py`.
   - Inside `known_people`, create a sub-directory for each person you want to recognize. The name of the sub-directory will be used as the person's name (e.g., `john_doe`).
   - Place one or more clear images (in `.jpg`, `.jpeg`, or `.png` format) of that person inside their respective sub-directory.

   The directory structure should look like this:

   ```
   .
   ├── main.py
   └── known_people/
       ├── jane_doe/
       │   ├── image1.jpg
       │   └── image2.png
       └── john_doe/
           └── image1.jpeg
   ```

## How to Run

1. **Start the API Server**
   Run the following command in your terminal from the project's root directory:

   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8002
   ```

   The server will start, and the application will load the known faces from the `known_people` directory.
2. **Initiate Face Verification**
   To start the verification process for a person (e.g., `jane_doe`), open your web browser or use a tool like `curl` to access the following URL:

   ```
   http://localhost:8002/verify_face/jane_doe
   ```

   Or using `curl`:

   ```bash
   curl http://localhost:8002/verify_face/jane_doe
   ```

   This will trigger the API, and a window titled "Face Verification Demo" will pop up, showing your webcam feed.
3. **Verification Process**

   - Position your face in front of the webcam.
   - The application will attempt to identify you.
   - If you are identified as `jane_doe`, the box around your face will turn green, the window will pause, and the API will return a success message. You can press any key to close the window.
   - If the verification times out or you press `q` to quit, the API will return an error.

## API Endpoint

### `GET /verify_face/{target_name}`

- **Description**: Initiates a webcam feed to find and verify a specific person.
- **URL Parameters**:
  - `target_name` (string, required): The name of the person to verify. This must match one of the sub-directory names in the `known_people` folder.
- **Success Response (200 OK)**:
  ```json
  {
    "status": "Verified",
    "person": "jane_doe"
  }
  ```
- **Failure Response (404 Not Found)**:
  ```json
  {
    "detail": "Target 'jane_doe' not found within 30 seconds. Status: Unauthorized"
  }
  ```
- **Error Response (500 Internal Server Error)**:
  Returned if the webcam cannot be accessed.
