import os
import uuid
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from dotenv import load_dotenv
from pytube import extract
from youtube_transcript_api import YouTubeTranscriptApi
from pydantic import BaseModel
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from starlette.responses import JSONResponse
from background_task import (
    audio_transcription_task,
    TranscriptionResult,
    transcription_tasks,
)

app = FastAPI()

load_dotenv()

endpoint = "https://models.inference.ai.azure.com"
model_name = "AI21-Jamba-1.5-Mini"
token = os.getenv("GITHUB_TOKEN")

if not token:
    raise HTTPException(status_code=404, detail="Token not found")

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)


class URLRequest(BaseModel):
    url: str


@app.get("/")
async def root():
    return {"status": "This service is running!"}


@app.post("/transcript/url")
async def getTranscriptfromURL(request: URLRequest):
    url = request.url
    v_id = extract.video_id(url)
    try:
        transcript_info = YouTubeTranscriptApi.get_transcript(v_id)
        transcript = " ".join([elem["text"] for elem in transcript_info])
        print(f"Succesfully fetched transcript: {transcript[:50]}...")

    except Exception:
        raise HTTPException(status_code=404, detail="Transcript not found")

    response = client.complete(
        messages=[
            SystemMessage(
                content=(
                    """
                        Summarize the provided transcript in one paragraph, not exceeding 100 words. 
                        Always start with phrases like 'This video is about...' or 'In this video...'. 
                        The summary should be concise and informative.
                    """
                )
            ),
            UserMessage(content=transcript),
        ],
        temperature=1.0,
        top_p=1.0,
        max_tokens=200,
        model=model_name,
    )

    return {"summary": response.choices[0].message.content}


@app.post("/transcript/audio/task")
async def getTranscriptfromAudio(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
):
    try:
        # Validate file type
        if not file.filename.endswith((".mp3", ".wav")):
            raise HTTPException(status_code=400, detail="Unsupported file format")

        folder = "/audio/uploads"
        os.makedirs(folder, exist_ok=True)

        # Save the uploaded file
        file_location = os.path.join(folder, file.filename)
        with open(file_location, "wb") as f:
            f.write(file.file.read())

        # Generate a unique task ID
        task_id = str(uuid.uuid4())

        # Perform the transcription in the background
        background_tasks.add_task(audio_transcription_task, file_location, task_id)

        return JSONResponse(content={"task_id": task_id}, status_code=201)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Check the status of the transcription task
@app.post("/transcript/audio/task/{task_id}", response_model=TranscriptionResult)
async def getTranscriptionResult(task_id: str):
    # Find the task with list comprehension and return it
    task = next((task for task in transcription_tasks if task.task_id == task_id), None)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")

    return JSONResponse(content=task.dict(), status_code=200)


@app.post("/transcript/audio")
async def getSummaryFromAudioTranscript(task_id: str):
    try:
        # Find the task
        task = next(
            (task for task in transcription_tasks if task.task_id == task_id), None
        )
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")

        if task.status != "success":
            raise HTTPException(
                status_code=400, detail="Transcription not yet complete: " + task.status
            )

        if not task.transcription:
            raise HTTPException(status_code=400, detail="No transcription available")

        response = client.complete(
            messages=[
                SystemMessage(
                    content=(
                        "Summarize the provided transcript in one paragraph, not exceeding 100 words. "
                        "The summary should be concise and informative. "
                    )
                ),
                UserMessage(content=task.transcription),
            ],
            temperature=1.0,
            top_p=1.0,
            max_tokens=200,
            model=model_name,
        )

        task.summary = response.choices[0].message.content
        return JSONResponse({"summary": task.summary})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
