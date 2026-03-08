#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pyaudio",
#   "pynput",
#   "openai",
#   "python-dotenv",
#   "pystray",
#   "pillow",
# ]
# ///
"""
whisper-dictate: Press hotkey to record, press again to transcribe + type.
Uses OpenAI Whisper API. System dep: xdotool

Install uv (if needed):
    curl -LsSf https://astral.sh/uv/install.sh | sh

Install system dep:
    sudo apt install xdotool portaudio19-dev

Usage:
    uv run whisper-dictate.py

    Optional args:
    --key KEY        Hotkey to use (default: f9)
    --lang LANG      Language hint, e.g. 'en' (default: auto-detect)
    --debug          Show verbose output
"""

import argparse
import os
import subprocess
import sys
import tempfile
import threading
import wave

from dotenv import load_dotenv
import pyaudio
from pynput import keyboard
from openai import OpenAI
from PIL import Image, ImageDraw
import pystray

# Load environment variables from .env file
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 1024
FORMAT = pyaudio.paInt16
WHISPER_MODEL = "whisper-1"

# ── State ─────────────────────────────────────────────────────────────────────
recording = False
audio_frames = []
audio_thread = None
pa = pyaudio.PyAudio()
lock = threading.Lock()
tray_icon = None

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Voice dictation via OpenAI Whisper")
parser.add_argument("--key", default="f9", help="Hotkey (e.g. f9, f12, ctrl+shift+v)")
parser.add_argument("--lang", default=None, help="Language hint (e.g. 'en')")
parser.add_argument("--debug", action="store_true", help="Verbose output")
args = parser.parse_args()

# ── OpenAI client ─────────────────────────────────────────────────────────────
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY environment variable not set.")
    print("  export OPENAI_API_KEY='sk-...'")
    sys.exit(1)
client = OpenAI(api_key=api_key)


def log(msg):
    if args.debug:
        print(f"[debug] {msg}")


def load_icon(state):
    """Load a PNG icon for the given state."""
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    icon_path = os.path.join(script_dir, f"icon_{state}.png")
    return Image.open(icon_path)


def update_tray_icon(state):
    """Update the tray icon to show current state.
    state: 'ready', 'recording', or 'transcribing'
    """
    global tray_icon
    if tray_icon:
        if state == "recording":
            tray_icon.icon = load_icon("recording")
            tray_icon.title = "whisper-dictate [REC]"
        elif state == "transcribing":
            tray_icon.icon = load_icon("transcribing")
            tray_icon.title = "whisper-dictate [transcribing...]"
        else:
            tray_icon.icon = load_icon("ready")
            tray_icon.title = "whisper-dictate [ready]"


# ── Audio recording ───────────────────────────────────────────────────────────
def record_audio():
    global audio_frames
    audio_frames = []
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )
    log("Recording started")
    while recording:
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_frames.append(data)
    stream.stop_stream()
    stream.close()
    log(f"Recording stopped — {len(audio_frames)} chunks captured")


# ── Transcription ─────────────────────────────────────────────────────────────
def transcribe_and_type():
    if not audio_frames:
        print("No audio captured.")
        update_tray_icon("ready")
        return

    print("⏳ Transcribing...")
    update_tray_icon("transcribing")

    # Write WAV to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    with wave.open(tmp_path, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(pa.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(audio_frames))

    # Call Whisper API
    text = None
    try:
        with open(tmp_path, "rb") as audio_file:
            kwargs = {"model": WHISPER_MODEL, "file": audio_file}
            if args.lang:
                kwargs["language"] = args.lang
            result = client.audio.transcriptions.create(**kwargs)
        text = result.text.strip()
    except Exception as e:
        print(f"ERROR: Whisper API call failed: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    if not text:
        print("(No speech detected)")
        update_tray_icon("ready")
        return

    print(f"✅ \"{text}\"")

    # Type text at cursor via xdotool
    try:
        subprocess.run(
            ["xdotool", "type", "--clearmodifiers", "--", text],
            check=True,
        )
    except FileNotFoundError:
        print("ERROR: xdotool not found. Install with: sudo apt install xdotool")
    except subprocess.CalledProcessError as e:
        print(f"ERROR: xdotool failed: {e}")
    finally:
        update_tray_icon("ready")


# ── Hotkey handling ───────────────────────────────────────────────────────────
def parse_hotkey(key_str):
    """Parse a hotkey string like 'f9' or 'ctrl+shift+v' into pynput format."""
    parts = key_str.lower().split("+")
    key_name = parts[-1]
    modifiers = parts[:-1]

    # Resolve the main key
    if hasattr(keyboard.Key, key_name):
        main_key = getattr(keyboard.Key, key_name)
    else:
        main_key = keyboard.KeyCode.from_char(key_name)

    # Resolve modifiers
    mod_map = {
        "ctrl": keyboard.Key.ctrl,
        "shift": keyboard.Key.shift,
        "alt": keyboard.Key.alt,
        "super": keyboard.Key.cmd,
        "cmd": keyboard.Key.cmd,
    }
    mod_keys = [mod_map[m] for m in modifiers if m in mod_map]
    return main_key, set(mod_keys)


main_key, required_mods = parse_hotkey(args.key)
pressed_mods = set()


def on_press(key):
    global recording, audio_thread

    # Track modifier keys
    if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
        pressed_mods.add(keyboard.Key.ctrl)
    elif key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
        pressed_mods.add(keyboard.Key.shift)
    elif key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r):
        pressed_mods.add(keyboard.Key.alt)

    # Check if hotkey triggered
    if key == main_key and required_mods.issubset(pressed_mods):
        with lock:
            if not recording:
                recording = True
                update_tray_icon("recording")
                print(f"🎙️  Recording... (press {args.key} again to stop)")
                audio_thread = threading.Thread(target=record_audio, daemon=True)
                audio_thread.start()
            else:
                recording = False
                audio_thread.join()
                threading.Thread(target=transcribe_and_type, daemon=True).start()


def on_release(key):
    if key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
        pressed_mods.discard(keyboard.Key.ctrl)
    elif key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
        pressed_mods.discard(keyboard.Key.shift)
    elif key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r):
        pressed_mods.discard(keyboard.Key.alt)




# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global tray_icon
    
    print(f"🚀 whisper-dictate ready")
    print(f"   Hotkey : {args.key}")
    print(f"   Model  : {WHISPER_MODEL}")
    print(f"   Lang   : {args.lang or 'auto-detect'}")
    print(f"   Press {args.key} to start recording, again to transcribe.")
    print(f"   Click tray icon to quit.\n")

    # Create system tray icon
    # Note: On Linux/Xorg, menus are not supported, so we use a default action
    # On other platforms, the menu will work normally
    def on_activate(icon, item):
        icon.stop()
    
    # Use a default menu item for Xorg compatibility (primary click activates default)
    menu = pystray.Menu(
        pystray.MenuItem("Quit", on_activate, default=True)
    )
    
    tray_icon = pystray.Icon(
        "whisper-dictate",
        load_icon("ready"),  # Green = ready
        "whisper-dictate [ready]",
        menu=menu
    )
    
    # Run keyboard listener in a separate thread
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    
    # Run the tray icon (blocks until stopped)
    tray_icon.run()
    
    # Cleanup
    listener.stop()
    pa.terminate()


if __name__ == "__main__":
    main()
