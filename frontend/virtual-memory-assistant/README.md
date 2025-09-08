# Virtual Memory Assistant

## Overview
The Virtual Memory Assistant is a web application that allows users to upload and query memories in various formats, including text, audio, video, and images. The application utilizes a FastAPI backend for handling requests and a SQLite database for storing metadata about the memories.

## Project Structure
```
virtual-memory-assistant
├── backend
│   ├── main.py               # FastAPI backend implementation
│   └── requirements.txt      # Python packages required for the backend
├── frontend
│   ├── index.html            # Main HTML file for the frontend
│   ├── styles.css            # CSS styles for the frontend
│   └── app.js                # JavaScript code for frontend interactions
└── README.md                 # Documentation for the project
```

## Installation

### Backend
1. Navigate to the `backend` directory:
   ```
   cd backend
   ```

2. Install the required Python packages:
   ```
   pip install -r requirements.txt
   ```

3. Initialize the database by running the FastAPI application:
   ```
   uvicorn main:app --reload
   ```

### Frontend
1. Navigate to the `frontend` directory:
   ```
   cd frontend
   ```

2. Open `index.html` in a web browser to access the application.

## Usage
- **Uploading Memories**: Use the form in the frontend to upload memories. You can upload text, audio, video, or images.
- **Querying Memories**: Enter a query in the provided field to search for memories based on their content.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.