from pydantic import BaseModel
from transcribe_wrapper import transcribe_audio

transcription_tasks = []

class TranscriptionResult(BaseModel):
    task_id: str
    transcription: str = None
    summary: str = None
    status: str = "processing" # success/error/processing
    message: str = None



def audio_transcription_task(file_location: str, task_id: str):
    # Create a new Transcription task
    new_task = TranscriptionResult(task_id=task_id)

    # Add the task to the list of transcription tasks
    transcription_tasks.append(new_task)

    # Perform the transcription
    transcription, error = transcribe_audio(file_location)

    if error:
        new_task.status = "error"
        new_task.message = error
        return
    else:
        new_task.transcription = transcription

    new_task.status = "success"