from dotenv import load_dotenv
import os
import cv2
import threading
import base64
import time
import json
from datetime import datetime
from queue import Queue
from flask import Flask, render_template, jsonify, request, send_from_directory, Response, url_for, abort
from flask_socketio import SocketIO
from PIL import Image, ImageGrab
import numpy as np
from openai import OpenAI
import webbrowser
import chromadb
from chromadb.config import Settings
import signal
import sys
from math import ceil
import easyocr
import concurrent.futures
import traceback
import math
import shutil

# Load environment variables from .env file
load_dotenv()

# Set the API key for OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Base screenshot path
base_path = os.getenv("BASE_PATH")

# set up the default interval for capturing screenshots
capture_interval = 10

# Define paths for frames
frames_path = os.path.join(base_path, "frames")

# Create directories if they don't exist
os.makedirs(frames_path, exist_ok=True)

# Initialize Flask app and SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Configure the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

running = True

# Create the chroma_db directory if it doesn't exist
os.makedirs("./chroma_db", exist_ok=True)

# Initialize ChromaDB
chroma_client = chromadb.Client(Settings(
    persist_directory="./chroma_db",
    is_persistent=True
))
collection = chroma_client.get_or_create_collection(name="screenshots")

text_queue = Queue()

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

def add_to_chroma(embedding, metadata):
    collection.add(
        embeddings=[embedding],
        documents=[metadata["combined_text"]],
        metadatas=[metadata],
        ids=[metadata["id"]]
    )


def generate_embedding(text):
    try:
        response = client.embeddings.create(input=text, model="text-embedding-ada-002")
        return response.data[0].embedding
    except Exception as e:
        print(f"Error in generate_embedding: {e}")
        return None

def analyze_image(encoded_image):
    try:
        messages = [
            {
                "role": "system",
                "content": "Your job is to generate a concise and informative description of the given screenshot, focusing on the main subject and any relevant details. URL's, prices, names are also important! Limit your response to a maximum of 100 words.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please provide a concise and informative description of the main subject and any relevant details in this screenshot.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                    },
                ],
            },
        ]
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=300,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in analyze_image: {e}")
        return ""

def perform_ocr(image_path):
    try:
        result = reader.readtext(image_path)
        text = ' '.join([entry[1] for entry in result])
        return text
    except Exception as e:
        print(f"Error in perform_ocr: {e}")
        return ""

def capture_images():
    global capture_interval, running
    last_capture_time = time.time()  # Initialize with current time
    while running:
        try:
            current_time = time.time()
            if current_time - last_capture_time < capture_interval:
                time.sleep(0.1)
                continue

            print(f"Attempting to capture screenshot at {datetime.now()}")

            # Capture the desktop screen using ImageGrab
            screenshot = ImageGrab.grab()
            frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

            # Resize and compress the image
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            max_size = 1920
            ratio = max_size / max(pil_img.size)
            new_size = tuple([int(x * ratio) for x in pil_img.size])
            resized_img = pil_img.resize(new_size, Image.LANCZOS)
            frame = cv2.cvtColor(np.array(resized_img), cv2.COLOR_RGB2BGR)

            # Save the screenshot with a unique filename
            filename = f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            frames_full_path = os.path.join(frames_path, filename)

            # Save
            cv2.imwrite(frames_full_path, frame)
            print(f"Screenshot saved: {frames_full_path}")

            encoded_image = encode_image(frames_full_path)
            if not encoded_image:
                print("Failed to encode image. Retrying in 5 seconds...")
                time.sleep(5)
                continue

            socketio.emit("stream", {"image": encoded_image})

            # Use ThreadPoolExecutor for parallel processing
            future_response = executor.submit(analyze_image, encoded_image)
            future_ocr = executor.submit(perform_ocr, frames_full_path)

            response_text = future_response.result()
            ocr_text = future_ocr.result()
            combined_text = f"Description: {response_text}\nOCR: {ocr_text}"

            metadata_entry = {
                "id": str(int(time.time() * 1000)),  # Use timestamp as ID
                "timestamp": datetime.now().isoformat(),
                "combined_text": combined_text,
                "screenshot_path": frames_full_path,
                "subject": analyze_subject(encoded_image),
            }
            embedding = generate_embedding(combined_text)
            if embedding is not None:
                add_to_chroma(embedding, metadata_entry)

            with text_queue.mutex:
                text_queue.queue.clear()

            text_queue.put(combined_text)
            socketio.emit("text", {"message": combined_text})

            last_capture_time = current_time
            print(f"Screenshot processing completed at {datetime.now()}")

        except Exception as e:
            print(f"Error in capture_images: {e}")
            print(traceback.format_exc())
            time.sleep(5)

        time.sleep(0.1)

    print("Capture thread has stopped.")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_subject(encoded_image):
    try:
        messages = [
            {
                "role": "system",
                "content": "Your job is to generate a subject for the given screenshot, reply with only a few keywords, nothing else. Example: 'cat, dog, tree'",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is the subject of this screenshot?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                    },
                ],
            },
        ]
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.0,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in analyze_subject: {e}")
        return ""


@app.route("/")
def index():
    # Get all screenshots
    results = collection.get(
        include=["metadatas"],
        where={},
    )

    recent_screenshots = []
    last_screenshot = None
    last_description = "No screenshots available"
    last_ocr = "No OCR data available"
    last_keywords = []

    if results["metadatas"]:
        # Sort the results by timestamp in descending order
        sorted_metadatas = sorted(results["metadatas"], key=lambda x: x["timestamp"], reverse=True)
        
        # Get the last 5 screenshots
        for meta in sorted_metadatas[:5]:
            recent_screenshots.append({
                "path": f"/frames/{os.path.basename(meta['screenshot_path'])}",
                "timestamp": meta["timestamp"],
                "description": meta["combined_text"][:30] + "..."
            })
        
        last_meta = sorted_metadatas[0]
        last_screenshot = f"/frames/{os.path.basename(last_meta['screenshot_path'])}"
        
        combined_text = last_meta["combined_text"]
        description, ocr = combined_text.split('\nOCR:', 1)
        last_description = description.replace('Description:', '').strip()
        last_ocr = ocr.strip() if ocr else "No OCR data available"
        last_keywords = last_meta["subject"].split(", ")

    return render_template("index.html", 
                           last_screenshot=last_screenshot, 
                           last_description=last_description,
                           last_ocr=last_ocr,
                           last_keywords=last_keywords,
                           recent_screenshots=recent_screenshots)

@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(os.path.join(app.root_path, 'images'), filename)

@app.route("/search_page")
def search_page():
    page = request.args.get('page', 1, type=int)
    per_page = 12
    sort_by = request.args.get('sort_by', 'similarity')
    sort_order = request.args.get('sort_order', 'desc')
    query = request.args.get('query', '')

    if query:
        embedding = generate_embedding(query)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=1000,
            include=["metadatas", "documents", "distances"]
        )
        entries = [
            {**meta, "similarity": 1 - dist}
            for meta, dist in zip(results["metadatas"][0], results["distances"][0])
        ]
        entries.sort(key=lambda x: x["similarity"], reverse=True)
        sort_by = 'similarity'  # Force sort_by to be 'similarity' when there's a query
        sort_order = 'desc'  # Force sort_order to be 'desc' for similarity
    else:
        results = collection.get(
            include=["metadatas"],
            where={},
            limit=1000000
        )
        entries = results["metadatas"]
        if sort_by == "similarity":
            sort_by = "timestamp"  # Default to timestamp if no query (similarity doesn't apply)

        if sort_by != "similarity":
            entries.sort(key=lambda x: x[sort_by], reverse=(sort_order == 'desc'))

    total_entries = len(entries)
    total_pages = math.ceil(total_entries / per_page)
    
    start = (page - 1) * per_page
    end = start + per_page
    
    paginated_entries = entries[start:end]
    
    for entry in paginated_entries:
        entry['screenshot_path'] = url_for('download_file', filename=os.path.basename(entry['screenshot_path']))
        if 'similarity' not in entry:
            entry['similarity'] = 0  # Add a default similarity for non-search results
    
    return render_template("search.html", entries=paginated_entries, page=page, total_pages=total_pages, 
                           sort_by=sort_by, sort_order=sort_order, query=query, per_page=per_page)


@app.route("/stop")
def stop():
    global running
    running = False
    return Response(status=200)


@app.route("/resume")
def resume():
    global running
    global capture_thread
    running = True
    if not capture_thread.is_alive():
        capture_thread = threading.Thread(target=capture_images)
        capture_thread.start()
    return Response(status=200)


@app.route("/set_interval", methods=["POST"])
def set_interval():
    global capture_interval
    interval = request.json.get("interval")
    if interval is not None and interval >= 1:
        capture_interval = interval
        return jsonify({"status": "interval updated", "interval": capture_interval})
    return jsonify({"status": "failed", "message": "Invalid interval"}), 400


@app.route("/search", methods=["POST"])
def search():
    cleanup_database()
    query = request.json.get("query")
    if not query:
        return jsonify({"status": "failed", "message": "Query is required"}), 400

    embedding = generate_embedding(query)
    if embedding is None:
        return jsonify({"status": "failed", "message": "Failed to generate embedding"}), 500

    results = collection.query(
        query_embeddings=[embedding],
        n_results=10,
        include=["metadatas", "documents", "distances"]
    )

    if not results["ids"]:
        return jsonify({"status": "success", "results": [], "message": "No results found"})

    formatted_results = [
        {
            "id": id,
            "timestamp": meta["timestamp"],
            "combined_text": meta["combined_text"],
            "screenshot_path": f"/frames/{os.path.basename(meta['screenshot_path'])}",
            "subject": meta["subject"],
            "keywords": meta["subject"].split(", "),
            "similarity": 1 - dist
        }
        for id, meta, dist in zip(results["ids"][0], results["metadatas"][0], results["distances"][0])
    ]

    return jsonify({"status": "success", "results": formatted_results})


@app.route("/frames/<path:filename>")
def download_file(filename):
    return send_from_directory(frames_path, filename, as_attachment=False)


@app.route("/edit_entry/<string:entry_id>", methods=["POST"])
def edit_entry(entry_id):
    new_subject = request.form.get("subject")
    new_combined_text = request.form.get("combined_text")

    results = collection.get(
        ids=[entry_id],
        include=["metadatas", "embeddings"]
    )

    if not results["ids"]:
        abort(404)

    metadata = results["metadatas"][0]
    metadata["subject"] = new_subject
    metadata["combined_text"] = new_combined_text

    new_embedding = generate_embedding(new_combined_text)

    collection.update(
        ids=[entry_id],
        embeddings=[new_embedding],
        metadatas=[metadata],
        documents=[new_combined_text]
    )

    return jsonify({"status": "success", "message": "Entry updated successfully"})


@app.route("/delete_entry/<string:entry_id>", methods=["POST"])
def delete_entry(entry_id):
    results = collection.get(
        ids=[entry_id],
        include=["metadatas"]
    )

    if not results["ids"]:
        return jsonify({"status": "failed", "message": "Entry not found"}), 404

    metadata = results["metadatas"][0]
    screenshot_path = metadata["screenshot_path"]

    # Delete the entry from the collection
    collection.delete(ids=[entry_id])

    # Delete the screenshot file
    if os.path.exists(screenshot_path):
        os.remove(screenshot_path)

    # Get the current page and query parameters
    page = request.args.get('page', 1, type=int)
    query = request.args.get('query', '')
    sort_by = request.args.get('sort_by', 'timestamp')
    sort_order = request.args.get('sort_order', 'desc')

    # Recalculate pagination
    per_page = 12
    if query:
        embedding = generate_embedding(query)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=1000,
            include=["metadatas", "documents", "distances"]
        )
        entries = [
            {**meta, "similarity": 1 - dist}
            for meta, dist in zip(results["metadatas"][0], results["distances"][0])
        ]
        entries.sort(key=lambda x: x["similarity"], reverse=True)
    else:
        results = collection.get(
            include=["metadatas"],
            where={},
            limit=1000000
        )
        entries = results["metadatas"]

    if sort_by != "similarity":
        entries.sort(key=lambda x: x[sort_by], reverse=(sort_order == 'desc'))

    total_entries = len(entries)
    total_pages = math.ceil(total_entries / per_page)

    # If the current page is now empty, go to the previous page
    if page > total_pages:
        page = max(1, total_pages)

    return jsonify({
        "status": "success", 
        "message": "Entry deleted successfully",
        "new_page": page,
        "total_pages": total_pages
    })

def cleanup_database():
    results = collection.get(
        include=["metadatas"],
        where={},
    )

    for metadata in results["metadatas"]:
        if not os.path.exists(metadata['screenshot_path']):
            collection.delete(ids=[metadata['id']])
            print(f"Removed entry {metadata['id']} from database due to missing screenshot.")

cleanup_database()

@app.route("/get_interval")
def get_interval():
    global capture_interval
    return jsonify({"interval": capture_interval})


@app.route("/get_status")
def get_status():
    global running
    return jsonify({"is_capturing": running})


@app.route("/get_latest_data")
def get_latest_data():
    results = collection.get(
        include=["metadatas"],
        where={},
    )

    recent_screenshots = []
    last_screenshot = None
    last_description = "No screenshots available"
    last_ocr = "No OCR data available"
    last_keywords = []

    if results["metadatas"]:
        sorted_metadatas = sorted(results["metadatas"], key=lambda x: x["timestamp"], reverse=True)
        
        for meta in sorted_metadatas[:5]:
            recent_screenshots.append({
                "path": f"/frames/{os.path.basename(meta['screenshot_path'])}",
                "timestamp": meta["timestamp"],
                "description": meta["combined_text"][:30] + "..."
            })
        
        last_meta = sorted_metadatas[0]
        last_screenshot = f"/frames/{os.path.basename(last_meta['screenshot_path'])}"
        
        combined_text = last_meta["combined_text"]
        description, ocr = combined_text.split('\nOCR:', 1)
        last_description = description.replace('Description:', '').strip()
        last_ocr = ocr.strip() if ocr else "No OCR data available"
        last_keywords = last_meta["subject"].split(", ")

    return jsonify({
        "last_screenshot": last_screenshot,
        "last_description": last_description,
        "last_ocr": last_ocr,
        "last_keywords": last_keywords,
        "recent_screenshots": recent_screenshots
    })


def signal_handler(sig, frame):
    global running
    running = False
    print("Exiting...")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    global capture_thread
    running = True
    capture_thread = threading.Thread(target=capture_images)
    capture_thread.start()
    print(f"Capture thread started at {datetime.now()}")

    # Open the default web browser to the server link
    webbrowser.open("http://localhost:5001")

    try:
        print(f"Starting Flask server at {datetime.now()}")
        socketio.run(app, host="0.0.0.0", port=5001)
    except Exception as e:
        print(f"Error in main execution: {e}")
        print(traceback.format_exc())
    finally:
        print(f"Shutting down at {datetime.now()}")
        running = False
        capture_thread.join()
        executor.shutdown(wait=True)
        text_queue.put(None)

