import re
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
log = logging.getLogger("BB")

with open('secrets.json', 'r') as f:
    secrets = json.load(f)

# secrets config for now
TAILSCALE_IP    = secrets["network"]["SERVER_IP"]   
PORT            = secrets["network"]["SERVER_PORT"]
STT_MODEL       = secrets["ai_models"]["STT_MODEL"]
VLM_MODEL       = secrets["ai_models"]["VLM_MODEL"]
TTS_VOICE       = secrets["ai_models"]["TTS_VOICE"]

# hardcoded these 
WHISPER_DEVICE  = "cuda"
WAKE_WORD       = ["Hey BB", "Hey BeeBee", "Hey BB"]
STOP_WORDS      = ["BB stop", "go to sleep", "stop listening"]
KOKORO_MODEL    = "kokoro-v1.0.onnx"
KOKORO_VOICES   = "voices-v1.0.bin"

# global state tracking
chat_history = []
BB_speaking       = False
BB_awake          = False
BB_thinking       = False
force_audio_flush = False
use_vision_next   = True # Flag to toggle vision on/off based on button
state_lock        = threading.Lock()
latest_frame      = None
frame_lock        = threading.Lock()

ws_clients: set = set()
ws_lock = threading.Lock()

audio_queue = queue.Queue()
transcription_queue = queue.Queue() # NEW: Decoupled queue for Whisper
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
    global client_sample_rate, BB_awake, BB_speaking, BB_thinking, force_audio_flush, use_vision_next
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
                    
                    elif data.get("type") in ["manual_wake_voice", "manual_wake_photo"]:
                        with state_lock:
                            BB_awake = True
                            BB_speaking = False
                            BB_thinking = False
                            force_audio_flush = True
                            use_vision_next = (data.get("type") == "manual_wake_photo")
                            
                        while not audio_queue.empty():
                            try: audio_queue.get_nowait()
                            except: break
                            
                        mode = "PHOTO" if use_vision_next else "VOICE"
                        push_overlay({"type": "status", "text": f"LISTENING ({mode})..."})
                        log.info(f"🎯 Manual Wake Triggered ({mode} Mode). Listening...")
                        
                    elif data.get("type") == "interrupt":
                        with state_lock:
                            BB_speaking = False
                            BB_thinking = False
                            BB_awake = False
                            force_audio_flush = True
                        while not audio_queue.empty():
                            try: audio_queue.get_nowait()
                            except: break
                        push_overlay({"type": "status", "text": "AWAITING WAKE WORD"})
                        push_overlay({"type": "transcript", "text": "—"})
                        push_overlay({"type": "interrupt_audio"}) # Tell frontend to kill queue
                        log.info("🛑 User Interrupted BB.")
                except Exception:
                    pass
    except Exception:
        pass
    finally:
        with ws_lock:
            ws_clients.discard(ws)

def push_overlay(data: dict):
    with ws_lock:
        clients = set(ws_clients)
    for ws in clients:
        try: ws.send(json.dumps(data))
        except Exception: pass

def pcm_to_wav_bytes(pcm: np.ndarray, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        pcm_int16 = (np.clip(pcm, -1.0, 1.0) * 32767).astype(np.int16)
        wf.writeframes(pcm_int16.tobytes())
    return buf.getvalue()

def auto_unmute_mic():
    global BB_speaking
    with state_lock:
        BB_speaking = False
    log.info("🔊 TTS Queue Finished. Ready for Wake Word.")
    push_overlay({"type": "status", "text": "AWAITING WAKE WORD"})
    push_overlay({"type": "transcript", "text": "—"})

def process_tts_chunk(text_chunk: str):
    if not text_chunk or not kokoro_tts: return
    try:
        samples, sr = kokoro_tts.create(text_chunk, voice=TTS_VOICE, speed=1.0, lang="en-us")
        wav_bytes = pcm_to_wav_bytes(samples, sr)
        b64_audio = base64.b64encode(wav_bytes).decode('utf-8')          
        push_overlay({"type": "tts_chunk", "data": b64_audio})
    except Exception as e:
        log.error(f"TTS Chunk Gen Failed: {e}")

# main ai logic for vision and text
def activate_BB_brain(query: str, include_vision: bool):
    global BB_thinking, latest_frame, BB_speaking, BB_awake, force_audio_flush, chat_history
    with state_lock:
        if BB_thinking: return
        BB_thinking = True

    log.info(f"🧠 Processing: '{query}'")
    push_overlay({"type": "status", "text": "THINKING..."})

    try:
        img_bytes = None
        if include_vision:
            frame_wait_deadline = time.time() + 1.0
            while latest_frame is None and time.time() < frame_wait_deadline:
                time.sleep(0.1)
            with frame_lock:
                if latest_frame is not None:
                    h, w = latest_frame.shape[:2]
                    max_dim = 960
                    if max(h, w) > max_dim:
                        scale = max_dim / max(h, w)
                        ai_frame = cv2.resize(latest_frame, (int(w * scale), int(h * scale)))
                    else:
                        ai_frame = latest_frame.copy()
                    _, buf = cv2.imencode('.jpg', ai_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    img_bytes = buf.tobytes()

        images_list = [img_bytes] if img_bytes else []
        log.info(f"📸 Vision: {'frame attached' if img_bytes else 'text only'}")
        
        if img_bytes:
            b64_preview = base64.b64encode(img_bytes).decode('utf-8')
            push_overlay({"type": "vision_preview", "data": b64_preview})
        else:
            push_overlay({"type": "hide_vision"})

        messages_payload = [
            {"role": "system", "content": "You are BB, a concise voice AI assistant. Answer in 1-3 spoken sentences only."}
        ]
        for msg in chat_history:
            messages_payload.append(msg)

        current_msg = {"role": "user", "content": query}
        if images_list:
            current_msg["images"] = images_list
            
        messages_payload.append(current_msg)

        
        # Stream the LLM response to allow immediate interruptions
        resp_stream = ollama.chat(
            model=VLM_MODEL,
            messages=messages_payload,
            options={"num_predict": 100, "temperature": 0.8},
            stream=True
        )
        
        with state_lock:
            BB_speaking = True
            
        full_text = ""
        sentence_buffer = ""
        
        # Sentences split on punctuation to stream to Kokoro quickly
        sentence_enders = re.compile(r'([.?!])\s+')

        for chunk in resp_stream:
            # Check interrupt flag constantly during generation
            with state_lock:
                if force_audio_flush:
                    log.info("🛑 LLM Generation aborted mid-stream.")
                    break
                    
            text_part = chunk['message']['content']
            if text_part:
                full_text += text_part
                sentence_buffer += text_part
                
                # If we have a complete sentence, ship it to Kokoro and clear buffer
                match = sentence_enders.search(sentence_buffer)
                if match:
                    split_idx = match.end()
                    sentence_to_speak = sentence_buffer[:split_idx].strip()
                    sentence_buffer = sentence_buffer[split_idx:]
                    
                    if sentence_to_speak:
                        push_overlay({"type": "response_update", "text": full_text})
                        process_tts_chunk(sentence_to_speak)

        # Process any leftover text
        if sentence_buffer.strip() and not force_audio_flush:
            push_overlay({"type": "response_update", "text": full_text})
            process_tts_chunk(sentence_buffer.strip())

        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": full_text.strip()})
            
        if len(chat_history) > 6:
            chat_history = chat_history[-6:]

        log.info(f"🤖 BB Completed: {full_text.strip()}")
        push_overlay({"type": "tts_complete"}) # Tell frontend generation is done

    except Exception as e:
        log.exception("Brain execution failed:")
    finally:
        with state_lock:
            BB_thinking = False
            if not BB_speaking: # if TTS aborted early
                BB_awake = False
        log.info("💤 Brain logic complete.")

# DECOUPLED TRANSCRIBER LOOP
def transcription_worker_loop():
    global BB_awake, use_vision_next
    VISUAL_TRIGGERS = ["this", "that", "these", "those", "look", "see", "color", "read", "show", "camera", "what is", "whats", "what's"]
        
    while True:
        audio_data, force_vision = transcription_queue.get()
        
        segments, _ = whisper.transcribe(io.BytesIO(audio_data), language="en", vad_filter=True)
        final_text = " ".join(s.text for s in segments).strip()
        
        if not final_text:
            continue
            
        if any(w in final_text.lower() for w in STOP_WORDS):
            with state_lock: BB_awake = False
            log.info("💤 Sleeping.")
            push_overlay({"type": "status", "text": "AWAITING WAKE WORD"})
            push_overlay({"type": "transcript", "text": "—"})
            continue
            
        if any(w.lower() in final_text.lower() for w in WAKE_WORD):
            with state_lock: BB_awake = True
            push_overlay({"type": "status", "text": "LISTENING..."})
            lower_text = final_text.lower()
            triggered_word = next(w.lower() for w in WAKE_WORD if w.lower() in lower_text)
            
            trigger_idx = lower_text.find(triggered_word)
            cmd_with_wake = final_text[trigger_idx:].strip()

            cmd_after = lower_text.split(triggered_word, 1)[-1].strip()
            clean_cmd = cmd_after.translate(str.maketrans('', '', string.punctuation)).strip()

            if clean_cmd: 
                # Check for vision triggers
                needs_vision = any(trigger in clean_cmd.lower().split() for trigger in VISUAL_TRIGGERS)
                
                # Pass the newly reconstructed string WITH the wake word
                push_overlay({"type": "transcript", "text": cmd_with_wake})
                threading.Thread(target=activate_BB_brain, args=(cmd_with_wake, needs_vision), daemon=True).start()
            else:
                log.info("Awake and waiting for command...")
                
        elif BB_awake:
            needs_vision = any(trigger in final_text.lower().split() for trigger in VISUAL_TRIGGERS)
            
            push_overlay({"type": "transcript", "text": final_text})
            threading.Thread(target=activate_BB_brain, args=(final_text, needs_vision), daemon=True).start()

# VAD AUDIO LOOP (Ultra-fast, no whisper logic)
def audio_processor_loop():
    global client_sample_rate, force_audio_flush, use_vision_next
    local_accumulator = np.array([], dtype=np.float32)
    active_phrase_chunks = []
    is_speaking = False
    silence_ms  = 0
    CHUNK_SAMPLES = 1600 
    
    log.info("🚀 BB VAD Core Online.")
    wake_timestamp = 0.0

    while True:
        try:
            raw_bytes = audio_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        with state_lock:
            flushing = force_audio_flush
            if force_audio_flush:
                local_accumulator = np.array([], dtype=np.float32)
                active_phrase_chunks = []
                is_speaking = False
                silence_ms = 0
                wake_timestamp = time.time()
                force_audio_flush = False
                flushing = True

        if flushing or (wake_timestamp > 0 and time.time() - wake_timestamp < 0.4):
            while not audio_queue.empty():
                try: audio_queue.get_nowait()
                except: break
            continue

        in_pcm = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        if client_sample_rate != 16000:
            n = int(len(in_pcm) * 16000 / client_sample_rate)
            if n > 0:
                in_pcm = np.interp(np.linspace(0, len(in_pcm)-1, n), np.arange(len(in_pcm)), in_pcm).astype(np.float32)
                
        local_accumulator = np.concatenate((local_accumulator, in_pcm))
        
        while len(local_accumulator) >= CHUNK_SAMPLES:
            with state_lock:
                if force_audio_flush:
                    local_accumulator = np.array([], dtype=np.float32)
                    active_phrase_chunks = []
                    is_speaking = False
                    break
                    
            eval_window = local_accumulator[:CHUNK_SAMPLES]
            local_accumulator = local_accumulator[CHUNK_SAMPLES:]
            
            if BB_thinking or BB_speaking:
                active_phrase_chunks = []
                is_speaking = False
                silence_ms = 0
                while not audio_queue.empty():
                    try: audio_queue.get_nowait()
                    except: break
                continue

            vol = np.abs(eval_window).mean()
            if vol > 0.003: # Speech detected
                if not is_speaking: is_speaking = True
                silence_ms = 0
                active_phrase_chunks.append(eval_window)
            elif is_speaking: # Silence detected
                active_phrase_chunks.append(eval_window)
                silence_ms += 100
                
                if silence_ms >= 400: # 600ms of silence = phrase end
                    full_phrase = np.concatenate(active_phrase_chunks)
                    wav_data = pcm_to_wav_bytes(full_phrase, 16000)
                    transcription_queue.put((wav_data, use_vision_next))
                    
                    is_speaking = False
                    silence_ms = 0
                    active_phrase_chunks = []

# serve frontend
@app.route('/')
def index():
    return send_from_directory('.', 'HUD_Frontend.html')

@app.route('/frame', methods=['POST'])
def receive_frame():
    global latest_frame
    data = request.get_data()
    if data:
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            with frame_lock: latest_frame = img
    return '', 204

@app.route('/reset', methods=['POST'])
def reset_pipeline():
    global BB_awake, BB_thinking, BB_speaking, force_audio_flush
    log.info("Reset triggered. Purging audio buffers and killing LLM...")
    
    with state_lock:
        BB_thinking = False
        BB_speaking = False
        BB_awake = False
        force_audio_flush = True
    
    while not audio_queue.empty():
        try: audio_queue.get_nowait()
        except: break
            
    push_overlay({"type": "status", "text": "AWAITING WAKE WORD"})
    push_overlay({"type": "transcript", "text": "—"})
    push_overlay({"type": "interrupt_audio"})
    
    return jsonify({"status": "pipeline_cleared"})

if __name__ == '__main__':
    threading.Thread(target=audio_processor_loop, daemon=True).start()
    threading.Thread(target=transcription_worker_loop, daemon=True).start()
    app.run(host=TAILSCALE_IP, port=PORT, threaded=True, debug=False, ssl_context='adhoc')