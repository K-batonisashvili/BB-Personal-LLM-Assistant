# Bibi AI Assistant

This is a fully local, real-time voice and vision assistant. It streams raw audio and camera frames from a phone or desktop browser directly to a Python backend over WebSockets. Once it hears the wake word (or manually clicking/tapping the listen button), it uses local machine learning models to transcribe your voice, look at your camera feed, and speak an answer back to you.

Everything runs locally. No cloud APIs, no subscriptions, no data tracking, no cache or cookies, and no corporate data scraping. Everything can be run on your local machine.

### How it works

* **Frontend (hud.html):** A lightweight web interface that accesses your camera and microphone. It streams highly compressed audio bytes over a WebSocket and features a hardware lens selector to ensure you get the best camera quality.
* **Backend (Bibi server.py):** A Flask server that catches the incoming audio and video streams. 
* **Speech-to-Text:** Uses faster-whisper to constantly monitor the audio stream for silence and the wake word.
* **The Brain:** Uses Ollama (running a vision-language model) to understand prompts and analyze a captured camera frame.
* **Text-to-Speech:** Uses Kokoro-ONNX to generate a voice response and streams the audio back to the initializing device.

### Prerequisites

You will need Ollama installed on your machine and a vision model pulled.
`ollama pull <your_preferred_vision_model>`

You also need Python installed along with a few packages:
`pip install flask flask-sock faster-whisper kokoro-onnx opencv-python numpy sounddevice`

### Setup

1. Clone or download this repository.
2. Create a file named `secrets.json` in the same directory as the python script. This keeps your network info and model preferences out of the main code. Make sure to add `secrets.json` to your `.gitignore` file.

Add this exact structure to your `secrets.json` and fill in your details:

{
  "network": {
    "SERVER_IP": "YOUR_IP_ADDRESS",
    "SERVER_PORT": 1234
  },
  "ai_models": {
    "STT_MODEL": "YOUR_SPEECH2TEXT_MODEL",
    "VLM_MODEL": "YOUR_VISION_MODEL",
    "TTS_VOICE": "YOUR_PREFERRED_VOICE"
  },
  "preferences": {
    "WAKE_WORD": "your wake word"
  }
}

### Running the server

1. Run the backend script:
`python "Bibi server.py"`

2. Open your browser and navigate to the server IP (or your Tailscale IP) over port 1234.
Example: `https://YOUR_IP_ADDRESS:1234`

Note: Modern mobile browsers require a secure context to access the camera and microphone. If you are not using localhost, you need to route it through a secure tunnel like Tailscale HTTPS or accept the ad-hoc self-signed certificate warning in your browser.

### Usage

* Click "Initialize" on the web interface to grant camera and mic permissions.
* Use the dropdown at the top to select your preferred camera lens.
* Say the wake word followed by your question, or tap "Manual Listen" to bypass the wake word.
* If Bibi is giving a long answer you want to cut off, tap "Stop Bibi".