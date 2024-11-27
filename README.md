## Transcript Summarization using FastAPI, AI21 Jambo 1.5 Mini, and Faster-Whisper for Audio Transcription

This mini-project demonstrates how to use FastAPI to create an application for summarizing transcripts using AI21 Jambo 1.5 Mini and Faster-Whisper for audio transcription. The application allows users to paste a YouTube video link or upload an audio file to get a summary of their transcript.

### Components

- **FastAPI**: A modern web framework for building APIs with Python.
- **AI21 Jambo 1.5 Mini**: A pre-trained language model for text generation and summarization.
- **Faster-Whisper**: A pre-trained model for audio transcription.

### How to Test the Application

1. **Clone the repository**:
    ```bash
    git clone <repository-url>
    ```

2. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Start the FastAPI server**:
    ```bash
    uvicorn main:app --reload
    ```

    This will start the FastAPI server, and you can access the application at `http://localhost:8000`.

4. **Test the application**:
    - Open the Swagger UI at `http://localhost:8000/docs`.
    - Paste a YouTube video link or upload an audio file.
    - The application will return a summary of the transcript.

