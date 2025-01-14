# YouTubeExtractor

This project uses OpenAI's GPT model to parse text from video frames into lyrics, notes, and chords. The parsed data is then saved into a text file.

## Features

- Extracts frames from a video file.
- Performs OCR on each frame to extract text.
- Uses GPT to parse the extracted text into lyrics, notes, and chords.
- Saves the parsed data into a text file.

## Requirements

- Python 3.11+
- OpenAI API key
- Poetry for dependency management
- Tesseract OCR (must be pre-installed)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/goldytech/YouTubeExtractor.git
    cd YouTubeExtractor
    ```

2. Install Poetry if you haven't already:
    ```sh
    curl -sSL https://install.python-poetry.org | python3 -
    ```

3. Install the dependencies:
    ```sh
    poetry install
    ```

4. Install Tesseract OCR:
    - On macOS:
      ```sh
      brew install tesseract
      ```
    - On Ubuntu:
      ```sh
      sudo apt-get install tesseract-ocr
      ```
    - On Windows, download the installer from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki).

5. Set up your OpenAI API key:
    - Create a `.env` file in the project root directory.
    - Add your OpenAI API key to the `.env` file:
      ```
      OPENAI_API_KEY=your_openai_api_key
      ```

## Usage

1. Run the main script:
    ```sh
    poetry run python main2.py
    ```

2. Follow the prompts to enter the YouTube video URL, song name, and film name.

3. The script will download the video, extract frames, perform OCR, and use GPT to parse the text. The results will be saved in `notes_and_words.txt`.

## Project Structure

- `main2.py`: The main script to run the project.
- `ai_extraction.py`: Contains the function to extract frames, perform OCR, and call GPT.
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `README.md`: This file.

## Outputs

- `notes_and_words.txt`: Contains the parsed lyrics, notes, and chords.
- `gpt_response.txt`: Contains the raw response from the GPT model.

## License

This project is licensed under the MIT License.