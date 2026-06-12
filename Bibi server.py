import os, threading, queue, time, base64, io, logging, wave, json, string
import numpy as np
import cv2
import ollama

from flask import Flask, request, jsonify, send_from_directory
from flask_sock import Sock
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro

# setup environment and config
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger("Bibi")

with open('secrets.json', 'r') as f:
    secrets = json.load(f)

# secrets config for now
TAILSCALE_IP    = secrets["network"]["SERVER_IP"]   
PORT            = secrets["network"]["SERVER_PORT"]
STT_MODEL       = secrets["ai_models"]["STT_MODEL"]
VLM_MODEL       = secrets["ai_models"]["VLM_MODEL"]
TTS_VOICE       = secrets["ai_models"]["TTS_VOICE"]

# hardcoded these 
WHISPER_DEVICE  = "cpu"
WAKE_WORD       = "Bibi wake up"
STOP_WORDS      = ["Bibi stop", "go to sleep", "stop listening"]
KOKORO_MODEL    = "kokoro-v1.0.onnx"
KOKORO_VOICES   = "voices-v1.0.bin"

# global state tracking
bibi_speaking   = False
bibi_awake      = False
bibi_thinking   = False
state_lock      = threading.Lock()
latest_frame    = None
frame_lock      = threading.Lock()

ws_clients: set = set()
ws_lock = threading.Lock()

audio_queue = queue.Queue()
client_sample_rate = 44100  

# initialize flask and websockets
app  = Flask(__name__)
sock = Sock(app)

# load ai models
log.info("Loading Whisper Model...")
whisper = WhisperModel(STT_MODEL, device=WHISPER_DEVICE, compute_type="int8")
log.info("✅ Whisper Ready.")

try:
    kokoro_tts = Kokoro(KOKORO_MODEL, KOKORO_VOICES)
    log.info("✅ Kokoro TTS Ready.")
except Exception as e:
    kokoro_tts = None
    log.warning(f"Kokoro offline: {e}")

# handle websocket connections
@sock.route('/ws')
def ws_handler(ws):
    global client_sample_rate, bibi_awake, bibi_speaking, bibi_thinking
    with ws_lock:
        ws_clients.add(ws)
    try:
        while True:
            msg = ws.receive()
            if msg is None: continue
            
            if isinstance(msg, bytes):
                audio_queue.put(msg)
            elif isinstance(msg, str):
                try:
                    data = json.loads(msg)
                    if data.get("type") == "init":
                        client_sample_rate = int(data.get("sampleRate", 44100))
                    
                    elif data.get("type") == "manual_wake":
                        with state_lock:
                            bibi_awake = True
                            bibi_speaking = False
                            bibi_thinking = False
                        while not audio_queue.empty():
                            try: audio_queue.get_nowait()
                            except: break
                        push_overlay({"type": "status", "text": "LISTENING (MANUAL)..."})
                        log.info("🎯 Manual Wake Triggered. Listening...")
                        
                    elif data.get("type") == "interrupt":
                        with state_lock:
                            bibi_speaking = False
                            bibi_thinking = False
                            bibi_awake = False
                        while not audio_queue.empty():
                            try: audio_queue.get_nowait()
                            except: break
                        push_overlay({"type": "status", "text": "AWAITING WAKE WORD"})
                        push_overlay({"type": "transcript", "text": "—"})
                        log.info("🛑 User Interrupted bibi.")
                except Exception:
                    pass
    except Exception:
        pass
    finally:
        with ws_lock:
            ws_clients.discard(ws)

# send data to all connected ui clients
def push_overlay(data: dict):
    with ws_lock:
        clients = set(ws_clients)
    for ws in clients:
        try: ws.send(json.dumps(data))
        except Exception: pass

# convert raw float arrays to wav bytes
def pcm_to_wav_bytes(pcm: np.ndarray, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        pcm_int16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16)
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()

# automatically unmute mic after tts finishes
def auto_unmute_mic(sleep_time):
    time.sleep(sleep_time)
    global bibi_speaking
    with state_lock:
        bibi_speaking = False
    log.info("🔊 Mic Unmuted - Ready for Wake Word.")
    push_overlay({"type": "status", "text": "AWAITING WAKE WORD"})
    push_overlay({"type": "transcript", "text": "—"})

# main ai logic for vision and text
def activate_bibi_brain(query: str):
    global bibi_thinking, latest_frame, bibi_speaking, bibi_awake
    with state_lock:
        if bibi_thinking: return
        bibi_thinking = True

    log.info(f"🧠 Processing: '{query}'")
    push_overlay({"type": "status", "text": "THINKING..."})
    
    try:
        img_bytes = None
        with frame_lock:
            if latest_frame is not None:
                h, w = latest_frame.shape[:2]
                max_dim = 960
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    ai_frame = cv2.resize(latest_frame, (int(w * scale), int(h * scale)))
                else:
                    ai_frame = latest_frame
                    
                _, buf = cv2.imencode('.jpg', ai_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                img_bytes = buf.tobytes()

        images_list = [img_bytes] if img_bytes else []
        prompt = f"You are Bibi, an advanced AI interface. Respond to the query: '{query}'"

        resp = ollama.generate(
            model=VLM_MODEL,
            prompt=prompt,
            images=images_list,
            options={"num_predict": 100, "temperature": 0.2} 
        )
        text = resp['response'].strip()
        
        # fallback safety catch
        if not text:
            text = "I'm having trouble processing that right now."
            
        log.info(f"🤖 Bibi: {text}")
        push_overlay({"type": "response", "text": text})

        if kokoro_tts:
            samples, sr = kokoro_tts.create(text, voice=TTS_VOICE, speed=1.0, lang="en-us")
            wav_bytes = pcm_to_wav_bytes(samples, sr)
            b64_audio = base64.b64encode(wav_bytes).decode('utf-8')
            
            audio_duration_seconds = len(samples) / sr
            with state_lock:
                bibi_speaking = True
                
            push_overlay({"type": "tts_audio", "data": b64_audio})
            threading.Thread(target=auto_unmute_mic, args=(audio_duration_seconds + 0.5,), daemon=True).start()

    except Exception as e:
        log.error(f"Pipeline error: {e}")
    finally:
        with state_lock:
            bibi_thinking = False
            bibi_awake = False
        log.info("💤 Returned to standby.")

# process streaming audio and handle vad/wake words
def audio_processor_loop():
    global client_sample_rate, bibi_awake
    local_accumulator = np.array([], dtype=np.float32)
    active_phrase_chunks = []
    is_speaking = False
    silence_ms  = 0
    CHUNK_SAMPLES = 6400 
    
    log.info("🚀 bibi Streaming Core Online.")
    
    while True:
        raw_bytes = audio_queue.get()
        in_pcm = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        if client_sample_rate != 16000:
            n = int(len(in_pcm) * 16000 / client_sample_rate)
            if n > 0:
                in_pcm = np.interp(np.linspace(0, len(in_pcm)-1, n), np.arange(len(in_pcm)), in_pcm).astype(np.float32)
                
        local_accumulator = np.concatenate((local_accumulator, in_pcm))
        
        while len(local_accumulator) >= CHUNK_SAMPLES:
            eval_window = local_accumulator[:CHUNK_SAMPLES]
            local_accumulator = local_accumulator[CHUNK_SAMPLES:]
            
            if bibi_thinking or bibi_speaking:
                active_phrase_chunks = []
                is_speaking = False
                silence_ms = 0
                while not audio_queue.empty():
                    try: audio_queue.get_nowait()
                    except: break
                continue

            vol = np.abs(eval_window).mean()
            if vol < 0.003: 
                if is_speaking:
                    active_phrase_chunks.append(eval_window)
                    silence_ms += 400
                    
                    if silence_ms >= 800: 
                        full_phrase = np.concatenate(active_phrase_chunks)
                        f_segments, _ = whisper.transcribe(io.BytesIO(pcm_to_wav_bytes(full_phrase, 16000)), language="en", vad_filter=True)
                        final_text = " ".join(s.text for s in f_segments).strip()
                        
                        if final_text:
                            if any(w in final_text.lower() for w in STOP_WORDS):
                                bibi_awake = False
                                log.info("💤 Sleeping.")
                                push_overlay({"type": "status", "text": "AWAITING WAKE WORD"})
                                push_overlay({"type": "transcript", "text": "—"})
                                
                            elif WAKE_WORD.lower() in final_text.lower():
                                bibi_awake = True
                                push_overlay({"type": "status", "text": "LISTENING..."})
                                cmd = final_text.lower().split(WAKE_WORD.lower())[-1].strip()
                                clean_cmd = cmd.translate(str.maketrans('', '', string.punctuation)).strip()
                                
                                if clean_cmd: 
                                    push_overlay({"type": "transcript", "text": cmd})
                                    threading.Thread(target=activate_bibi_brain, args=(cmd,), daemon=True).start()
                                else:
                                    log.info("Awake and waiting for command...")
                                    
                            elif bibi_awake:
                                push_overlay({"type": "transcript", "text": final_text})
                                threading.Thread(target=activate_bibi_brain, args=(final_text,), daemon=True).start()
                        
                        is_speaking = False
                        silence_ms = 0
                        active_phrase_chunks = []
                continue 

            segments, _ = whisper.transcribe(
                io.BytesIO(pcm_to_wav_bytes(eval_window, 16000)),
                language="en", vad_filter=True, vad_parameters={"min_silence_duration_ms": 250}
            )
            
            if len(list(segments)) > 0:
                if not is_speaking: is_speaking = True
                silence_ms = 0
                active_phrase_chunks.append(eval_window)
                
                full_phrase = np.concatenate(active_phrase_chunks)
                f_segments, _ = whisper.transcribe(io.BytesIO(pcm_to_wav_bytes(full_phrase, 16000)), language="en", vad_filter=True)
                interim_text = " ".join(s.text for s in f_segments).strip()
                
                if bibi_awake:
                    push_overlay({"type": "transcript", "text": interim_text})
            else:
                if is_speaking:
                    active_phrase_chunks.append(eval_window)
                    silence_ms += 400

# serve frontend
@app.route('/')
def index():
    return send_from_directory('.', 'hud.html')

# process incoming camera frames
@app.route('/frame', methods=['POST'])
def receive_frame():
    global latest_frame
    data = request.get_data()
    if data:
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            with frame_lock: latest_frame = img
    return '', 204

# start the server
if __name__ == '__main__':
    threading.Thread(target=audio_processor_loop, daemon=True).start()
    app.run(host=TAILSCALE_IP, port=PORT, threaded=True, debug=False, ssl_context='adhoc')