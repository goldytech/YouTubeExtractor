import cv2
import os
import pytesseract
import yt_dlp as youtube_dl
import logging
import re
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MUSICAL_NOTES_PATTERN = re.compile(r'\b(C|C#|Db|D|D#|Eb|E|F|F#|Gb|G|G#|Ab|A|A#|Bb|B)\b')

def download_video(url, output_path='video.mp4'):
    ydl_opts = {
        'format': 'best',
        'outtmpl': output_path,
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    # Remove noise
    denoised = cv2.fastNlMeansDenoising(thresh, None, 30, 7, 21)
    return denoised

def process_frame(frame, text_only, unique_texts):
    preprocessed_frame = preprocess_image(frame)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(preprocessed_frame, config=custom_config)
    if text_only:
        if MUSICAL_NOTES_PATTERN.search(text):
            unique_texts.add(text)
            logging.info(f'Extracted text: {text}')
    return unique_texts

def extract_frames(video_path, output_folder='frames', frame_skip=1, text_only=False, diff_threshold=3500, resize_factor=0.5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    prev_frame = None
    unique_texts = set()

    with ThreadPoolExecutor() as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_skip == 0:
                frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
                norm_diff = int(cv2.norm(frame, prev_frame)) if prev_frame is not None else None
                if prev_frame is None or norm_diff > diff_threshold:
                    logging.info(f'The norm difference between frames {frame_count} and {frame_count - frame_skip} is {norm_diff}.')
                    unique_texts = executor.submit(process_frame, frame, text_only, unique_texts).result()
                    prev_frame = frame
            frame_count += 1

    cap.release()
    logging.info('Frame extraction complete.')

    # Write unique texts to a single file
    with open(os.path.join(output_folder, 'unique_texts.txt'), 'w') as f:
        for text in unique_texts:
            f.write(text + '\n')
    logging.info('Unique texts saved to unique_texts.txt.')

def main(url):
    try:
        logging.info("Starting video download...")
        video_path = download_video(url)
        logging.info(f"Video downloaded to: {video_path}")

        print("Choose frame extraction method:")
        print("1. Extract all frames")
        print("2. Skip frames")
        print("3. Extract frames with text")
        choice = input("Enter your choice (1/2/3): ")

        if choice == '1':
            extract_frames(video_path)
        elif choice == '2':
            frame_skip = int(input("Enter the number of frames to skip: "))
            extract_frames(video_path, frame_skip=frame_skip)
        elif choice == '3':
            extract_frames(video_path, text_only=True)
        else:
            logging.error("Invalid choice. Exiting.")
            print("Invalid choice. Exiting.")

    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")

if __name__ == '__main__':
    video_url = input("Enter the YouTube video URL: ")
    main(video_url)