import os
import logging
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor

import cv2
import pytesseract
import yt_dlp
from dotenv import load_dotenv
from httpx import Client
from openai import OpenAI

###############################################################################
# CONFIGS & CONSTANTS
###############################################################################
NOTE_REGEX = re.compile(r'[A-G][#b]?')  # For the "Regular" approach
FRAME_STEP = 30  # Increase frame step to skip more frames
PARALLEL = True  # Enable parallel processing
MAX_WORKERS = 4  # Number of workers for parallel processing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


###############################################################################
# 1) HELPER: LOAD ENV & SET OPENAI API KEY
###############################################################################
def load_openai_key_from_env():
    """
    Loads the OPENAI_API_KEY from .env file using python-dotenv.
    """
    load_dotenv()  # loads environment variables from .env
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file!")
    logging.info("OpenAI API key loaded successfully.")


###############################################################################
# 2) DOWNLOAD VIDEO WITH YT-DLP
###############################################################################
def download_youtube_video(youtube_url, download_folder=".", output_filename="tutorial.mp4"):
    """
    Download a single progressive MP4 (includes audio+video).
    Uses '22/18' format so no ffmpeg merging is required.
    """
    if not os.path.exists(download_folder):
        os.makedirs(download_folder, exist_ok=True)

    outtmpl = os.path.join(download_folder, output_filename)

    ydl_opts = {
        'outtmpl': outtmpl,
        'format': '22/18',  # 720p or 360p MP4 with audio+video
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            logging.info(f"Video info: {info}")
    except Exception as download_error:
        logging.error(f"Failed to download video: {youtube_url}. Error: {download_error}")
        raise RuntimeError("Video download failed!") from download_error

    if not os.path.exists(outtmpl):
        logging.error(f"Downloaded file not found: {outtmpl}")
        raise FileNotFoundError(f"Downloaded video is missing at: {outtmpl}")

    file_size = os.path.getsize(outtmpl)
    logging.info(f"Video downloaded to: {outtmpl} (Size: {file_size / (1024 * 1024):.2f} MB)")
    return outtmpl


###############################################################################
# 3) FRAME EXTRACTION
###############################################################################
def preprocess_frame(frame):
    """
    Crop + convert to grayscale for better OCR speed/accuracy.
    You can add thresholding if needed.
    """
    cropped = frame[0:200, 0:frame.shape[1]]  # top ~200 rows
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    return gray


def get_ocr_text(frame):
    """
    OCR the frame using Tesseract with psm=6 (block of text).
    """
    processed = preprocess_frame(frame)
    rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
    config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(rgb, config=config)
    return text.strip()


def format_timestamp(seconds_float):
    """
    Convert a float of seconds into mm:ss or hh:mm:ss.
    """
    total_seconds = int(round(seconds_float))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    if hours > 0:
        # logging.info(f"Video duration: {hours}h {minutes}m {seconds}s")
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        # logging.info(f"Video duration: {minutes}m {seconds}s")
        return f"{minutes:02d}:{seconds:02d}"


###############################################################################
# 4) AI-BASED EXTRACTION (GPT)
###############################################################################
def gpt_parse_text(raw_text, song_name, film_name):
    """
    Calls an OpenAI ChatCompletion model to separate raw text into 'lyrics' and 'notes'.
    Writes the response to a text file.
    """
    if not raw_text:
        return

    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "developer",
             "content": "You are a helpful music assistant. You parse the given text into separate lyrics (or normal text) "
                        "and musical notes (A, A#, B, Gb, etc.)."},
            {"role": "user", "content": "I am trying to extract lyrics ,notes and chords from a given text.\n"
                                        "" + raw_text + " \n for the song " + song_name + " from the movie " + film_name + "."
                                                                                                                           "This is a Bollywood song. Present your response in the form of lyrics, notes and chords in a well defined structure line by line of the lyrics.\n"
                                                                                                                           "Example: \n"
                                                                                                                           "Lyrics: Line 1 of lyrics\n"
                                                                                                                           "Notes: A, B, C\n"
                                                                                                                           "Chords: C, E, G\n"
                                                                                                                           "Extract the required information from the given text only."},
        ]
    )

    response = completion.choices[0].message
    print(response)
    with open("gpt_response.txt", "w", encoding="utf-8") as f:
        f.write(response.content.strip())
    logging.info("GPT response written to gpt_response.txt")


def process_frame(frame, frame_idx, fps):
    """
    Process a single frame: OCR and format the result.
    """
    timestamp_sec = frame_idx / fps
    raw_text = get_ocr_text(frame)
    if raw_text:
        return f"Time {format_timestamp(timestamp_sec)}:\n  Lyrics: {raw_text}"
    return None


def ai_extraction(video_path, song_name, film_name, frame_step=FRAME_STEP, parallel=PARALLEL, max_workers=MAX_WORKERS):
    """
    1) Read frames from video.
    2) For each sampled frame, do OCR -> raw_text.
    3) Make a single GPT call with all OCR data.
    4) Return a list of (lyrics, notes, timestamp).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Could not open video.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"Video FPS: {fps}, Total Frames: {total_frames}")
    all_ocr_text = []
    frame_idx = 0

    if parallel:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_step == 0:
                    futures.append(executor.submit(process_frame, frame, frame_idx, fps))

                frame_idx += 1

            for future in futures:
                result = future.result()
                if result:
                    all_ocr_text.append(result + "\n")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                result = process_frame(frame, frame_idx, fps)
                if result:
                    all_ocr_text.append(result + "\n")

            frame_idx += 1

    cap.release()

    combined_text = "".join(all_ocr_text)
    gpt_parse_text(combined_text, song_name, film_name)

    results = []
    for entry in all_ocr_text:
        time_str, lyrics_text = entry.split(":\n  Lyrics: ")
        results.append((lyrics_text, "", time_str))

    return results


###############################################################################
# 5) OUTPUT RESULTS & MAIN
###############################################################################
def write_results_to_file(results, output_file):
    """
    Each result: (lyrics, notes, timestamp)
    We write:
      line 1 -> lyrics
      line 2 -> notes
      line 3 -> timestamp
      blank line
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for (lyrics, notes, t_str) in results:
            f.write(f"{lyrics}\n")
            f.write(f"{notes}\n")
            f.write(f"{t_str}\n\n")
    logging.info(f"Results written to {output_file}")


def main():
    """
    1) Load API key from .env
    2) Prompt user for video URL
    3) Download video
    4) Run AI-based extraction
    5) Write results
    """
    # 1) Load .env
    load_openai_key_from_env()

    # 2) Prompt user for video URL
    url = input("Enter YouTube video URL: ").strip()
    if not url:
        logging.error("No URL provided. Exiting.")
        return

    song_name = input("Enter the song name: ").strip()
    film_name = input("Enter the film name: ").strip()

    start_time = time.time()

    # 3) Download video
    video_path = download_youtube_video(url, output_filename=f"{song_name}.mp4")

    # 4) Run AI-based extraction
    logging.info("Using AI-based method.")
    results = ai_extraction(video_path, song_name, film_name)

    # 5) Save results
    output_file = "notes_and_words.txt"
    write_results_to_file(results, output_file)

    end_time = time.time()
    logging.info(f"Total time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
