from faster_whisper import WhisperModel, BatchedInferencePipeline
import time
import os
import math
from pydub import AudioSegment


def transcribe_audio(audio_file):
    output_text = ""

    try:
        # Load the model
        model_size = "small"
        # model = WhisperModel(model_size, device="cpu", compute_type="int8")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        batched_model = BatchedInferencePipeline(model=model)

        MAX_DURATION = 10 * 60 * 1000  # 10 minutes in milliseconds

        # Load the audio file to check duration
        audio = AudioSegment.from_file(audio_file)
        audio_length = len(audio)

        transcription_texts = []

        if audio_length <= MAX_DURATION:
            # Start timer
            start_time = time.time()
            segments, info = batched_model.transcribe(
                audio_file, beam_size=5, batch_size=16
            )
            print(
                "Detected language '%s' with probability %f"
                % (info.language, info.language_probability)
            )
            for segment in segments:
                print(segment.text)
                transcription_texts.append(segment.text)
            end_time = time.time()
            duration_minutes = (end_time - start_time) / 60
            print(f"Transcription took {duration_minutes:.2f} minutes.")
        else:
            total_segments = math.ceil(audio_length / MAX_DURATION)
            print(
                f"Audio length exceeds 10 minutes; processing in {total_segments} chunks."
            )
            for i in range(total_segments):
                start_ms = i * MAX_DURATION
                end_ms = min((i + 1) * MAX_DURATION, audio_length)
                chunk = audio[start_ms:end_ms]
                temp_chunk_path = f"{audio_file}_chunk_{i}.mp3"
                print(f"Exporting chunk {i + 1}/{total_segments} to {temp_chunk_path}")
                chunk.export(
                    temp_chunk_path,
                    format="mp3",
                    parameters=["-ac", "1", "-ar", "16000"],
                )

                # Start timer for each chunk
                chunk_start = time.time()
                segments, info = batched_model.transcribe(
                    temp_chunk_path, beam_size=5, batch_size=16
                )
                print(
                    "Detected language '%s' with probability %f"
                    % (info.language, info.language_probability)
                )
                for segment in segments:
                    print(segment.text)
                    transcription_texts.append(segment.text)
                chunk_end = time.time()
                duration_minutes = (chunk_end - chunk_start) / 60
                print(
                    f"Chunk {i + 1} transcription took {duration_minutes:.2f} minutes."
                )

                if os.path.exists(temp_chunk_path):
                    os.remove(temp_chunk_path)
                    print(f"Removed temporary file: {temp_chunk_path}")

        # Combine transcriptions from all segments/chunks
        output_text = " ".join(transcription_texts)
        return output_text, None

    except Exception as e:
        return None, str(e)
