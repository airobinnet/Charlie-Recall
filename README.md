# Charlie Recall

Charlie Recall is a Flask-based web application that captures desktop screenshots at regular intervals, analyzes them using AI, and provides a searchable interface for reviewing the captured images and associated metadata.

## Screenshot

![Charlie Recall Screenshot](images/screenshot.png)

## Features

- Automatic screenshot capture at configurable intervals
- AI-powered image analysis using OpenAI's GPT-4 Vision
- Optical Character Recognition (OCR) for text extraction from images
- Real-time display of captured screenshots and analysis results
- Searchable database of screenshots with metadata
- Ability to edit and delete entries
- Responsive web interface with sorting and pagination

## Prerequisites

- Python 3.7+
- OpenAI API key

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/airobinnet/charlie-recall.git
   cd charlie-recall
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your OpenAI API key:
   ```bash
   OPENAI_API_KEY=your_api_key_here
   BASE_PATH=/path/to/store/screenshots
   ```

## Usage

1. Start the application:
   ```bash
   python main.py
   ```

2. The application will automatically open in your default web browser. If it doesn't, navigate to `http://localhost:5001` in your browser.

3. Use the "Start Capturing" and "Stop Capturing" buttons to control the screenshot capture process.

4. Adjust the capture interval in the Settings modal.

5. Use the Search page to find and review captured screenshots.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.