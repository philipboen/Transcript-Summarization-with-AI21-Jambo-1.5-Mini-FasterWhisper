from faster_whisper import WhisperModel

def transcribe_audio(audio_file):
    output_text = ""

    try:
        # Load the model
        model_size = "small"

        model = WhisperModel(model_size, device="cpu", compute_type="int8")

        # Transcribe the audio
        segments, info = model.transcribe(audio_file, beam_size=5, language="en")

        #  Combine the segments into a single string
        for segment in segments:
            output_text += segment.text + " "

        # Return the transcribed text
        return output_text, None

    except Exception as e:
        return None, str(e)