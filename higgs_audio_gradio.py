import gradio as gr
import torch
import torchaudio
import os
import sys
import argparse
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent
import numpy as np
import tempfile
import re
import gc
import time
from datetime import datetime
from pydub import AudioSegment
from pydub.utils import which

# Import our custom audio processing utilities
from audio_processing_utils import enhance_multi_speaker_audio, normalize_audio_volume

# Whisper for auto-transcription
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
    print("✅ Using faster-whisper for transcription")
except ImportError:
    try:
        import whisper
        WHISPER_AVAILABLE = True
        print("✅ Using openai-whisper for transcription")
    except ImportError:
        WHISPER_AVAILABLE = False
        print("⚠️ Whisper not available - voice samples will use dummy text")

# Initialize model
MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Global instances
serve_engine = None
whisper_model = None

# Cache management for optimizations
_audio_cache = {}
_token_cache = {}

def install_ffmpeg_if_needed():
    """Check if ffmpeg is available and provide installation instructions if not"""
    if which("ffmpeg") is None:
        print("⚠️ FFmpeg not found. For full audio format support, install FFmpeg:")
        print("   Windows: Download from https://ffmpeg.org/download.html")
        print("   macOS: brew install ffmpeg")
        print("   Linux: sudo apt install ffmpeg")
        return False
    return True

def convert_audio_to_standard_format(audio_path, target_sample_rate=24000, force_mono=False):
    """
    Convert any audio file to standard format using multiple fallback methods
    Returns: (audio_data_numpy, sample_rate) or raises exception
    Preserves stereo unless force_mono=True
    """
    print(f"🔄 Converting audio file: {audio_path}")
    
    # Method 1: Try torchaudio first (fastest)
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono only if explicitly requested
        if force_mono and waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            print("🔄 Converted stereo to mono")
        
        # Resample if needed
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
            waveform = resampler(waveform)
            sample_rate = target_sample_rate
        
        # Convert to numpy (preserve channel structure)
        if waveform.shape[0] == 1:
            # Mono - squeeze to 1D
            audio_data = waveform.squeeze().numpy()
        else:
            # Stereo - keep as 2D array (channels, samples)
            audio_data = waveform.numpy()
        
        channels = waveform.shape[0]
        samples = waveform.shape[1]
        print(f"✅ Loaded with torchaudio: {'stereo' if channels == 2 else 'mono'} - {samples} samples at {sample_rate}Hz")
        return audio_data, sample_rate
        
    except Exception as e:
        print(f"⚠️ Torchaudio failed: {e}")
    
    # Method 2: Try pydub (handles more formats, especially MP3)
    try:
        # Load with pydub
        if audio_path.lower().endswith('.mp3'):
            audio = AudioSegment.from_mp3(audio_path)
        elif audio_path.lower().endswith('.wav'):
            audio = AudioSegment.from_wav(audio_path)
        else:
            # Try to auto-detect format
            audio = AudioSegment.from_file(audio_path)
        
        # Convert to mono only if explicitly requested
        original_channels = audio.channels
        if force_mono and audio.channels > 1:
            audio = audio.set_channels(1)
            print("🔄 Converted stereo to mono")
        
        # Set sample rate
        audio = audio.set_frame_rate(target_sample_rate)
        
        # Convert to numpy array
        audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
        
        # Normalize to [-1, 1] range
        if audio.sample_width == 1:  # 8-bit
            audio_data = audio_data / 128.0
        elif audio.sample_width == 2:  # 16-bit
            audio_data = audio_data / 32768.0
        elif audio.sample_width == 4:  # 32-bit
            audio_data = audio_data / 2147483648.0
        else:
            # Assume already normalized or unknown format
            audio_data = audio_data / np.max(np.abs(audio_data)) if np.max(np.abs(audio_data)) > 1 else audio_data
        
        # Handle stereo data (pydub gives interleaved samples)
        if audio.channels == 2 and not force_mono:
            # Reshape interleaved stereo data to (2, samples)
            audio_data = audio_data.reshape(-1, 2).T
        
        channel_info = f"{'stereo' if audio.channels == 2 and not force_mono else 'mono'}"
        print(f"✅ Loaded with pydub: {channel_info} - {len(audio_data)} samples at {target_sample_rate}Hz")
        return audio_data, target_sample_rate
        
    except Exception as e:
        print(f"⚠️ Pydub failed: {e}")
    
    # Method 3: Try scipy as final fallback
    try:
        from scipy.io import wavfile
        sample_rate, audio_data = wavfile.read(audio_path)
        
        # Convert to float32 and normalize
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        elif audio_data.dtype == np.uint8:
            audio_data = (audio_data.astype(np.float32) - 128.0) / 128.0
        else:
            audio_data = audio_data.astype(np.float32)
        
        # Handle stereo/mono
        if len(audio_data.shape) > 1:
            if force_mono:
                # Convert stereo to mono
                audio_data = np.mean(audio_data, axis=1)
                print("🔄 Converted stereo to mono")
            else:
                # Keep stereo, transpose to (channels, samples)
                audio_data = audio_data.T
        
        # Resample if needed (basic resampling)
        if sample_rate != target_sample_rate:
            # Simple resampling - for better quality, use librosa
            ratio = target_sample_rate / sample_rate
            if len(audio_data.shape) == 1:
                # Mono
                new_length = int(len(audio_data) * ratio)
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), new_length),
                    np.arange(len(audio_data)),
                    audio_data
                )
            else:
                # Stereo
                new_length = int(audio_data.shape[1] * ratio)
                resampled = np.zeros((audio_data.shape[0], new_length))
                for channel in range(audio_data.shape[0]):
                    resampled[channel] = np.interp(
                        np.linspace(0, audio_data.shape[1], new_length),
                        np.arange(audio_data.shape[1]),
                        audio_data[channel]
                    )
                audio_data = resampled
            sample_rate = target_sample_rate
        
        channel_info = f"{'stereo' if len(audio_data.shape) > 1 else 'mono'}"
        samples = audio_data.shape[1] if len(audio_data.shape) > 1 else len(audio_data)
        print(f"✅ Loaded with scipy: {channel_info} - {samples} samples at {sample_rate}Hz")
        return audio_data, sample_rate
        
    except Exception as e:
        print(f"⚠️ Scipy failed: {e}")
    
    raise ValueError(f"❌ Could not load audio file: {audio_path}. Tried torchaudio, pydub, and scipy.")

def save_temp_audio_robust(audio_data, sample_rate, force_mono=False):
    """
    Robust version of save_temp_audio_fixed that handles various input formats
    and ensures compatibility with soundfile. Preserves stereo unless force_mono=True
    """
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_path = temp_file.name
    temp_file.close()
    
    try:
        # Ensure audio_data is numpy array
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.numpy()
        elif not isinstance(audio_data, np.ndarray):
            audio_data = np.array(audio_data)
        
        # Ensure float32 dtype
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        
        # Handle stereo/mono conversion
        if len(audio_data.shape) == 1:
            # Mono, shape (N,)
            if force_mono:
                audio_data = audio_data
            else:
                audio_data = np.expand_dims(audio_data, axis=0)  # (1, N)
        elif len(audio_data.shape) == 2:
            # Could be (channels, samples) or (samples, channels)
            if audio_data.shape[0] > audio_data.shape[1]:
                # (samples, channels) -> (channels, samples)
                audio_data = audio_data.T
            # If force_mono, average channels
            if force_mono and audio_data.shape[0] > 1:
                audio_data = np.mean(audio_data, axis=0, keepdims=True)
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0:
            audio_data = audio_data / max_val
        
        # Convert to tensor for torchaudio
        waveform = torch.from_numpy(audio_data).float()
        
        # Ensure 2D (channels, samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        print(f"Saving audio: shape={waveform.shape}, dtype={waveform.dtype}, max={waveform.max()}, min={waveform.min()}")
        
        torchaudio.save(temp_path, waveform, sample_rate)
        
        print(f"✅ Saved audio to: {temp_path}")
        return temp_path
        
    except Exception as e:
        print(f"❌ Error saving audio: {e}")
        # Cleanup temp file on error
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        raise

def process_uploaded_audio(uploaded_audio, force_mono=False):
    """
    Process uploaded audio from Gradio, handling various formats
    Returns: (audio_data_numpy, sample_rate) ready for use
    Preserves stereo unless force_mono=True
    """
    if uploaded_audio is None:
        raise ValueError("No audio uploaded")
    
    # Handle both tuple and individual components
    if isinstance(uploaded_audio, tuple) and len(uploaded_audio) == 2:
        sample_rate, audio_data = uploaded_audio
    else:
        raise ValueError("Invalid uploaded audio format - expected (sample_rate, audio_data) tuple")
    
    # If audio_data is already numpy array from Gradio
    if isinstance(audio_data, np.ndarray):
        # Ensure float32
        if audio_data.dtype != np.float32:
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            else:
                audio_data = audio_data.astype(np.float32)
        
        # Handle stereo/mono
        if len(audio_data.shape) > 1:
            if force_mono:
                # Convert stereo to mono
                audio_data = np.mean(audio_data, axis=1)
                print("🔄 Converted stereo to mono")
            else:
                # Keep stereo, but ensure proper channel order (channels, samples)
                if audio_data.shape[1] < audio_data.shape[0]:
                    # Data is (samples, channels), transpose to (channels, samples)
                    audio_data = audio_data.T
        
        # Normalize
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0:
            audio_data = audio_data / max_val
        
        channel_info = "mono"
        if len(audio_data.shape) > 1:
            channel_info = f"stereo ({audio_data.shape[0]} channels)"
        elif not force_mono and len(audio_data.shape) == 1:
            channel_info = "mono"
            
        samples = audio_data.shape[1] if len(audio_data.shape) > 1 else len(audio_data)
        print(f"✅ Processed uploaded audio: {channel_info} - {samples} samples at {sample_rate}Hz")
        return audio_data, sample_rate
    
    else:
        raise ValueError("Unexpected audio data format from Gradio")

def enhanced_save_temp_audio_fixed(uploaded_voice, force_mono=False):
    """
    Enhanced version that replaces the original save_temp_audio_fixed function
    Preserves stereo unless force_mono=True
    """
    if uploaded_voice is None:
        raise ValueError("No uploaded voice provided")
    
    # Handle both tuple and individual components properly
    if isinstance(uploaded_voice, tuple) and len(uploaded_voice) == 2:
        # Process the uploaded audio
        processed_audio, processed_rate = process_uploaded_audio(uploaded_voice, force_mono)
        
        # Save to temporary file
        return save_temp_audio_robust(processed_audio, processed_rate, force_mono)
    else:
        raise ValueError("Invalid uploaded voice format - expected (sample_rate, audio_data) tuple")

def load_audio_file_robust(file_path, target_sample_rate=24000):
    """
    Load any audio file and convert to standard format
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    return convert_audio_to_standard_format(file_path, target_sample_rate)

def check_dependencies():
    """Check and report on available audio processing libraries"""
    print("🔍 Checking audio processing dependencies...")
    
    dependencies = {
        "torchaudio": True,  # Should always be available in your setup
        "pydub": False,
        "scipy": False,
        "ffmpeg": False
    }
    
    try:
        import pydub
        dependencies["pydub"] = True
        print("✅ pydub available")
    except ImportError:
        print("⚠️ pydub not available - install with: pip install pydub")
    
    try:
        import scipy.io
        dependencies["scipy"] = True
        print("✅ scipy available")
    except ImportError:
        print("⚠️ scipy not available - install with: pip install scipy")
    
    dependencies["ffmpeg"] = install_ffmpeg_if_needed()
    
    return dependencies

def safe_audio_processing(uploaded_voice, operation_name):
    """Wrapper for safe audio processing with detailed error messages"""
    try:
        return enhanced_save_temp_audio_fixed(uploaded_voice)
    except Exception as e:
        error_msg = f"❌ Error processing audio for {operation_name}: {str(e)}\n"
        error_msg += "💡 Try these solutions:\n"
        error_msg += "  • Ensure your audio file is a valid WAV or MP3\n"
        error_msg += "  • Try converting your file using a different audio editor\n"
        error_msg += "  • Make sure the file isn't corrupted\n"
        error_msg += "  • Install additional dependencies: pip install pydub scipy"
        raise ValueError(error_msg)

def clear_caches():
    """Clear audio and token caches to free memory"""
    global _audio_cache, _token_cache
    _audio_cache.clear()
    _token_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    print("🧹 Cleared caches and freed memory")

def get_cache_key(text, voice_ref=None, temperature=0.3, top_k=50, top_p=0.95, min_p=None, 
                 repetition_penalty=1.0, ras_win_len=7, ras_win_max_num_repeat=2, do_sample=True):
    """Generate cache key for audio generation"""
    import hashlib
    key_str = f"{text}_{voice_ref}_{temperature}_{top_k}_{top_p}_{min_p}_{repetition_penalty}_{ras_win_len}_{ras_win_max_num_repeat}_{do_sample}"
    return hashlib.sha256(key_str.encode()).hexdigest()

# Create output directories - simplified
def create_output_directories():
    base_dirs = ["output/basic_generation", "output/voice_cloning", "output/longform_generation", "output/multi_speaker", "voice_library"]
    
    for dir_path in base_dirs:
        os.makedirs(dir_path, exist_ok=True)

# Initialize output directories
create_output_directories()

def get_output_path(category, filename_base, extension=".wav"):
    """Generate organized output paths with timestamps"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{filename_base}{extension}"
    output_path = os.path.join("output", category, filename)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    return output_path

def save_transcript_if_enabled(transcript, category, filename_base):
    """Save transcript to file if whisper is available - DISABLED"""
    # Disabled - we don't want to permanently save transcripts
    return None

def save_audio_reference_if_enabled(audio_path, category, filename_base):
    """Save audio reference if whisper is available - DISABLED"""
    # Disabled - we don't want to permanently save audio references
    return None

# Voice library management
def get_voice_library_voices():
    """Get list of voices in the voice library"""
    voice_library_dir = "voice_library"
    voices = []
    if os.path.exists(voice_library_dir):
        for f in os.listdir(voice_library_dir):
            if f.endswith('.wav'):
                voice_name = f.replace('.wav', '')
                voices.append(voice_name)
    return voices

def get_voice_config_path(voice_name):
    """Get the config file path for a voice"""
    return os.path.join("voice_library", f"{voice_name}_config.json")

def get_default_voice_config():
    """Get default generation parameters for a voice"""
    return {
        "temperature": 0.3,
        "max_new_tokens": 1024,
        "seed": 12345,
        "top_k": 50,
        "top_p": 0.95,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "ras_win_len": 7,
        "ras_win_max_num_repeat": 2,
        "do_sample": True,
        "description": "",
        "tags": []
    }

def save_voice_config(voice_name, config):
    """Save generation parameters for a voice"""
    import json
    config_path = get_voice_config_path(voice_name)
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving voice config: {e}")
        return False

def load_voice_config(voice_name):
    """Load generation parameters for a voice"""
    import json
    config_path = get_voice_config_path(voice_name)
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading voice config: {e}")
    
    # Return default config if file doesn't exist or can't be loaded
    return get_default_voice_config()

def save_voice_to_library(audio_data, sample_rate, voice_name):
    """Save a voice sample to the voice library"""
    if not voice_name or not voice_name.strip():
        return "❌ Please enter a voice name"
    
    voice_name = voice_name.strip().replace(' ', '_')
    voice_path = os.path.join("voice_library", f"{voice_name}.wav")
    
    # Check if voice already exists
    if os.path.exists(voice_path):
        return f"❌ Voice '{voice_name}' already exists in library"
    
    try:
        # Save audio using robust method
        temp_path = save_temp_audio_robust(audio_data, sample_rate)
        import shutil
        shutil.move(temp_path, voice_path)
        
        # Create transcript using Whisper
        transcription = transcribe_audio(voice_path)
        txt_path = voice_path.replace('.wav', '.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(transcription)
        
        # Create default config file for the voice
        default_config = get_default_voice_config()
        save_voice_config(voice_name, default_config)
        
        return f"✅ Voice '{voice_name}' saved to library with default settings!"
    
    except Exception as e:
        return f"❌ Error saving voice: {str(e)}"

def delete_voice_from_library(voice_name):
    """Delete a voice from the library"""
    if not voice_name or voice_name == "None":
        return "❌ Please select a voice to delete"
    
    voice_path = os.path.join("voice_library", f"{voice_name}.wav")
    txt_path = os.path.join("voice_library", f"{voice_name}.txt")
    
    try:
        if os.path.exists(voice_path):
            os.remove(voice_path)
        if os.path.exists(txt_path):
            os.remove(txt_path)
        return f"✅ Voice '{voice_name}' deleted from library"
    except Exception as e:
        return f"❌ Error deleting voice: {str(e)}"

def get_all_available_voices():
    """Get combined list of predefined voices and voice library"""
    voice_prompts_dir = "examples/voice_prompts"
    predefined = [f for f in os.listdir(voice_prompts_dir) if f.endswith(('.wav', '.mp3'))] if os.path.exists(voice_prompts_dir) else []
    library = get_voice_library_voices()
    
    combined = ["None (Smart Voice)"]
    if predefined:
        combined.extend([f"📁 {voice}" for voice in predefined])
    if library:
        combined.extend([f"👤 {voice}" for voice in library])
    
    return combined

def get_voice_path(voice_selection):
    """Get the actual path for a voice selection"""
    if not voice_selection or voice_selection == "None (Smart Voice)":
        return None
    
    if voice_selection.startswith("📁 "):
        # Predefined voice
        voice_name = voice_selection[2:]
        return os.path.join(voice_prompts_dir, voice_name)
    elif voice_selection.startswith("👤 "):
        # Library voice
        voice_name = voice_selection[2:]
        return os.path.join("voice_library", f"{voice_name}.wav")
    
    return None

def apply_voice_config_to_generation(voice_selection, transcript, scene_description="", force_audio_gen=False):
    """Apply a voice's saved configuration to generate audio"""
    if not voice_selection or voice_selection == "None (Smart Voice)":
        return None
    
    # Extract voice name from selection
    voice_name = None
    if voice_selection.startswith("👤 "):
        voice_name = voice_selection[2:]  # Remove "👤 " prefix
    
    if not voice_name:
        return None
    
    # Load voice configuration
    config = load_voice_config(voice_name)
    voice_path = os.path.join("voice_library", f"{voice_name}.wav")
    
    if not os.path.exists(voice_path):
        return None
    
    try:
        # Create messages using voice reference
        system_content = "Generate audio following instruction."
        if scene_description and scene_description.strip():
            system_content += f" <|scene_desc_start|>\n{scene_description}\n<|scene_desc_end|>"
        
        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content="Please speak this text."),
            Message(role="assistant", content=AudioContent(audio_url=voice_path)),
            Message(role="user", content=transcript)
        ]
        
        # Generate with voice's saved parameters
        min_p_value = config['min_p'] if config['min_p'] > 0 else None
        output = optimized_generate_audio(
            messages, config['max_new_tokens'], config['temperature'], 
            config['top_k'], config['top_p'], min_p_value, 
            config['repetition_penalty'], config['ras_win_len'], 
            config['ras_win_max_num_repeat'], config['do_sample']
        )
        
        return output
        
    except Exception as e:
        print(f"Error applying voice config: {e}")
        return None

# Available voice prompts - this needs to be refreshed dynamically
voice_prompts_dir = "examples/voice_prompts"

def get_current_available_voices():
    """Get current available voices (refreshed each time)"""
    return get_all_available_voices()

available_voices = get_current_available_voices()

def initialize_model():
    global serve_engine
    if serve_engine is None:
        print("🚀 Initializing Higgs Audio model...")
        serve_engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH, device=device)
        print("✅ Model initialized successfully")

def initialize_whisper():
    global whisper_model
    global WHISPER_AVAILABLE
    if whisper_model is None and WHISPER_AVAILABLE:
        try:
            # Try faster-whisper first
            whisper_model = WhisperModel("large-v3", device="cuda" if torch.cuda.is_available() else "cpu")
            print("✅ Loaded faster-whisper model")
        except ImportError:
            # Fallback to openai-whisper
            import whisper
            whisper_model = whisper.load_model("large")
            print("✅ Loaded openai-whisper model")
        except Exception as e:
            print(f"⚠️ Failed to load whisper model: {e}")
            # Try base model as fallback
            try:
                if 'WhisperModel' in globals():
                    whisper_model = WhisperModel("base", device="cuda" if torch.cuda.is_available() else "cpu")
                else:
                    import whisper
                    whisper_model = whisper.load_model("base")
                print("✅ Loaded whisper base model as fallback")
            except Exception as e2:
                print(f"❌ Failed to load any whisper model: {e2}")
                WHISPER_AVAILABLE = False

def transcribe_audio(audio_path):
    """Transcribe audio file to text using Whisper"""
    if not WHISPER_AVAILABLE:
        return "This is a voice sample for cloning."
    
    try:
        initialize_whisper()
        
        # More robust whisper model type detection
        if hasattr(whisper_model, 'transcribe'):
            # Check if it's faster-whisper by looking for specific attributes
            if hasattr(whisper_model, 'model') and hasattr(whisper_model, 'feature_extractor'):
                # Using faster-whisper
                segments, info = whisper_model.transcribe(audio_path, language="en")
                transcription = " ".join([segment.text for segment in segments])
            else:
                # Using openai-whisper
                result = whisper_model.transcribe(audio_path)
                transcription = result["text"]
        else:
            # Fallback
            return "This is a voice sample for cloning."
        
        # Clean up transcription
        transcription = transcription.strip()
        if not transcription:
            transcription = "This is a voice sample for cloning."
        
        print(f"🎤 Transcribed: {transcription[:100]}...")
        return transcription
        
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return "This is a voice sample for cloning."

def create_voice_reference_txt(audio_path, transcript_sample=None):
    """Create a corresponding .txt file for the voice reference with auto-transcription"""
    # Robust extension handling - handles all common audio extensions case-insensitively
    base_path = audio_path
    audio_extensions = ['.wav', '.WAV', '.mp3', '.MP3', '.flac', '.FLAC', '.m4a', '.M4A', '.ogg', '.OGG']

    
    for ext in audio_extensions:
        if audio_path.endswith(ext):
            base_path = audio_path[:-len(ext)]
            break
    
    txt_path = base_path + '.txt'
    
    if transcript_sample is None:
        # Auto-transcribe the audio
        transcript_sample = transcribe_audio(audio_path)
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(transcript_sample)
    
    print(f"📝 Created voice reference text: {txt_path}")
    return txt_path

def robust_txt_path_creation(audio_path):
    """
    Given an audio file path, returns the corresponding .txt path,
    handling all common audio extensions case-insensitively.
    """
    base_path = audio_path
    audio_extensions = ['.wav', '.WAV', '.mp3', '.MP3', '.flac', '.FLAC', '.m4a', '.M4A', '.ogg', '.OGG']
    for ext in audio_extensions:
        if audio_path.endswith(ext):
            base_path = audio_path[:-len(ext)]
            break
    return base_path + '.txt'

def robust_file_cleanup(files):
    """
    Safely delete a list of files (or a single file path), ignoring errors if files do not exist.
    """
    if not files:
        return
    if isinstance(files, str):
        files = [files]
    for f in files:
        if f and isinstance(f, str) and os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

def save_temp_audio(audio_data, sample_rate):
    """Save numpy audio data to temporary file and return path"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_path = temp_file.name
    temp_file.close()
    
    # Convert numpy array to tensor and save
    if isinstance(audio_data, np.ndarray):
        waveform = torch.from_numpy(audio_data).float()
        # If mono, add channel dimension
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        torchaudio.save(temp_path, waveform, sample_rate)
    
    return temp_path

def parse_multi_speaker_text(text):
    """Parse multi-speaker text and extract speaker assignments"""
    # Look for [SPEAKER0], [SPEAKER1], etc.
    speaker_pattern = r'\[SPEAKER(\d+)\]\s*([^[]*?)(?=\[SPEAKER\d+\]|$)'
    matches = re.findall(speaker_pattern, text, re.DOTALL)
    
    speakers = {}
    for speaker_id, content in matches:
        speaker_key = f"SPEAKER{speaker_id}"
        if speaker_key not in speakers:
            speakers[speaker_key] = []
        speakers[speaker_key].append(content.strip())
    
    return speakers

def detect_dynamic_speakers(text):
    """Detect any speaker names in brackets and return list of unique speakers"""
    if not text or not text.strip():
        return []
    
    # Pattern to match any text in brackets at the start of lines
    # Supports formats like [Alice], [Bob], [Character Name], etc.
    speaker_pattern = r'^\s*\[([^\]]+)\]\s*[:.]?\s*(.+?)(?=^\s*\[[^\]]+\]|$)'
    
    # Find all matches across multiple lines
    matches = re.findall(speaker_pattern, text, re.MULTILINE | re.DOTALL)
    
    # Extract unique speaker names
    speakers = []
    seen_speakers = set()
    
    for speaker_name, content in matches:
        speaker_name = speaker_name.strip()
        if speaker_name and speaker_name not in seen_speakers:
            speakers.append(speaker_name)
            seen_speakers.add(speaker_name)
    
    return speakers

def parse_dynamic_speaker_text(text, speaker_mapping=None):
    """Parse text with any bracket format and convert to internal format"""
    if not text or not text.strip():
        return {}
    
    # Pattern to match any text in brackets
    speaker_pattern = r'^\s*\[([^\]]+)\]\s*[:.]?\s*(.+?)(?=^\s*\[[^\]]+\]|$)'
    matches = re.findall(speaker_pattern, text, re.MULTILINE | re.DOTALL)
    
    speakers = {}
    
    for speaker_name, content in matches:
        speaker_name = speaker_name.strip()
        content = content.strip()
        
        if not content:
            continue
            
        # Use mapping if provided, otherwise use speaker name directly
        speaker_key = speaker_mapping.get(speaker_name, speaker_name) if speaker_mapping else speaker_name
        
        if speaker_key not in speakers:
            speakers[speaker_key] = []
        speakers[speaker_key].append(content)
    
    return speakers

def convert_to_speaker_format(text, speaker_mapping):
    """Convert dynamic speaker text to SPEAKER0, SPEAKER1 format"""
    if not text or not speaker_mapping:
        return text
    
    converted_text = text
    
    # Replace each speaker name with the mapped SPEAKER format
    for speaker_name, speaker_id in speaker_mapping.items():
        # Match [Speaker Name]: or [Speaker Name] 
        pattern = rf'\[{re.escape(speaker_name)}\]\s*[:.]?\s*'
        replacement = f'[{speaker_id}] '
        converted_text = re.sub(pattern, replacement, converted_text, flags=re.MULTILINE)
    
    return converted_text

def auto_format_multi_speaker(text):
    """Auto-format text for multi-speaker if not already formatted"""
    # If already has speaker tags, return as-is
    if '[SPEAKER' in text:
        return text
    
    # Split by common dialogue indicators
    lines = text.split('\n')
    formatted_lines = []
    current_speaker = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for dialogue indicators
        if line.startswith('"') or line.startswith("'") or ':' in line:
            # Switch speakers for dialogue
            if len(formatted_lines) > 0:
                current_speaker = 1 - current_speaker
            formatted_lines.append(f"[SPEAKER{current_speaker}] {line}")
        else:
            # Regular text, assign to current speaker
            formatted_lines.append(f"[SPEAKER{current_speaker}] {line}")
    
    return '\n'.join(formatted_lines)

def smart_chunk_text(text, max_chunk_size=200):
    """Smart text chunking that respects sentence boundaries and paragraphs"""
    # First split by paragraphs
    paragraphs = text.split('\n\n')
    chunks = []
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # If paragraph is short enough, keep it as one chunk
        if len(paragraph) <= max_chunk_size:
            chunks.append(paragraph)
            continue
        
        # Split long paragraphs by sentences
        sentences = []
        # Split by multiple sentence endings
        sentence_parts = re.split(r'([.!?]+)', paragraph)
        
        current_sentence = ""
        for i in range(0, len(sentence_parts), 2):
            if i < len(sentence_parts):
                current_sentence = sentence_parts[i].strip()
                if i + 1 < len(sentence_parts):
                    current_sentence += sentence_parts[i + 1]
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
        
        # Group sentences into chunks
        current_chunk = ""
        for sentence in sentences:
            # If adding this sentence would exceed limit, save current chunk
            if current_chunk and len(current_chunk) + len(sentence) + 1 > max_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
    
    return chunks

def optimized_generate_audio(messages, max_new_tokens, temperature, top_k=50, top_p=0.95, min_p=None, 
                           repetition_penalty=1.0, ras_win_len=7, ras_win_max_num_repeat=2, do_sample=True, use_cache=True):
    """Optimized audio generation with caching and incremental decoding"""
    cache_key = None
    if use_cache:
        # Include all parameters in cache key for proper caching
        cache_key = get_cache_key(str(messages), temperature=temperature, top_k=top_k, top_p=top_p, 
                                min_p=min_p, repetition_penalty=repetition_penalty, ras_win_len=ras_win_len, 
                                ras_win_max_num_repeat=ras_win_max_num_repeat, do_sample=do_sample)
        if cache_key in _audio_cache:
            print("🚀 Using cached audio result")
            return _audio_cache[cache_key]
    
    # Generate audio with optimizations
    # Note: Only include parameters that the serve engine actually supports
    generate_kwargs = {
        "chat_ml_sample": ChatMLSample(messages=messages),
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "stop_strings": ["<|end_of_text|>", "<|eot_id|>"],
        "ras_win_len": ras_win_len,
        "ras_win_max_num_repeat": ras_win_max_num_repeat,
    }
    
    # TODO: Future parameters to implement in serve engine:
    # - min_p: Alternative to top_p sampling
    # - repetition_penalty: Penalty for repeating tokens  
    # - do_sample: Enable/disable sampling vs greedy decoding
    
    output: HiggsAudioResponse = serve_engine.generate(**generate_kwargs)
    
    # Cache result if enabled
    if use_cache and cache_key:
        _audio_cache[cache_key] = output
        # Keep cache size manageable
        if len(_audio_cache) > 50:
            # Remove oldest entries
            oldest_key = next(iter(_audio_cache))
            del _audio_cache[oldest_key]
    
    return output

# VOICE LIBRARY FUNCTIONS

def test_voice_sample(audio_data, sample_rate, test_text="Hello, this is a test of my voice. How does it sound?"):
    """Test a voice sample with default text before saving to library"""
    if audio_data is None:
        return None, "❌ Please upload an audio sample first"
    
    try:
        # Initialize model
        initialize_model()
        
        # Save temporary audio using robust method
        temp_audio_path = save_temp_audio_robust(audio_data, sample_rate)
        temp_txt_path = create_voice_reference_txt(temp_audio_path)
        
        # Generate test audio using voice cloning
        system_content = "Generate audio following instruction."
        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content="Please speak this text."),
            Message(role="assistant", content=AudioContent(audio_url=temp_audio_path)),
            Message(role="user", content=test_text)
        ]
        
        # Generate audio
        output = optimized_generate_audio(messages, 1024, 0.3, use_cache=False)
        
        # Save test output
        test_output_path = "voice_test_output.wav"
        torchaudio.save(test_output_path, torch.from_numpy(output.audio)[None, :], output.sampling_rate)
        
        # Clean up temp files
        for path in [temp_audio_path, temp_txt_path]:
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass
        
        return test_output_path, "✅ Voice test completed! Listen to the result above."
        
    except Exception as e:
        return None, f"❌ Error testing voice: {str(e)}"

def generate_basic(
    transcript,
    voice_prompt,
    temperature,
    max_new_tokens,
    seed,
    scene_description,
    top_k=50,
    top_p=0.95,
    min_p=0.0,
    repetition_penalty=1.0,
    ras_win_len=7,
    ras_win_max_num_repeat=2,
    do_sample=True,
    enable_normalization=False,
    target_volume=0.15
):
    # Initialize model if not already done
    initialize_model()
    
    # Set seed for reproducibility
    if seed > 0:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Prepare system message
    system_content = "Generate audio following instruction."
    if scene_description and scene_description.strip():
        system_content += f" <|scene_desc_start|>\n{scene_description}\n<|scene_desc_end|>"
    
    # Handle voice selection using the same method as voice cloning tab
    if voice_prompt and voice_prompt != "None (Smart Voice)":
        ref_audio_path = get_voice_path(voice_prompt)
        if ref_audio_path and os.path.exists(ref_audio_path):
            # Create corresponding txt file path using robust method
            txt_path = robust_txt_path_creation(ref_audio_path)
            
            if not os.path.exists(txt_path):
                # Auto-transcribe the audio file instead of creating dummy text
                try:
                    transcription = transcribe_audio(ref_audio_path)
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(transcription)
                    print(f"📝 Auto-transcribed and created: {txt_path}")
                except Exception as e:
                    print(f"⚠️ Failed to transcribe {ref_audio_path}: {e}")
                    # Fallback to dummy text only if transcription fails
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write("This is a voice sample.")
                    print(f"📝 Created fallback text file: {txt_path}")
            
            # Use the same pattern as working voice cloning
            messages = [
                Message(role="system", content=system_content),
                Message(role="user", content="Please speak this text."),
                Message(role="assistant", content=AudioContent(audio_url=ref_audio_path)),
                Message(role="user", content=transcript)
            ]
        else:
            # Fallback to smart voice if file doesn't exist
            messages = [
                Message(role="system", content=system_content),
                Message(role="user", content=transcript)
            ]
    else:
        # Smart voice
        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content=transcript)
        ]
    
    # Generate audio with optimizations
    min_p_value = min_p if min_p > 0 else None  # Convert 0 to None for disabled
    output = optimized_generate_audio(
        messages, max_new_tokens, temperature, top_k, top_p, min_p_value, 
        repetition_penalty, ras_win_len, ras_win_max_num_repeat, do_sample
    )
    
    # Save and return audio with organized output
    output_path = get_output_path("basic_generation", "basic_audio")
    torchaudio.save(output_path, torch.from_numpy(output.audio)[None, :], output.sampling_rate)
    
    # Apply volume normalization if enabled
    if enable_normalization:
        print(f"🔊 Applying volume normalization to basic generation...")
        
        # Load the generated audio
        audio_data, sample_rate = torchaudio.load(output_path)
        
        # Apply simple normalization
        normalized_audio = normalize_audio_volume(
            audio_data.squeeze(),
            target_rms=target_volume,
            sample_rate=sample_rate
        )
        
        # Create normalized output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        normalized_filename = f"{timestamp}_normalized_basic_audio.wav"
        normalized_path = os.path.join("output", "basic_generation", normalized_filename)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(normalized_path), exist_ok=True)
        
        # Save normalized audio
        if isinstance(normalized_audio, torch.Tensor):
            if len(normalized_audio.shape) == 1:
                normalized_audio = normalized_audio.unsqueeze(0)
            torchaudio.save(normalized_path, normalized_audio, sample_rate)
        else:
            normalized_tensor = torch.tensor(normalized_audio, dtype=torch.float32)
            if len(normalized_tensor.shape) == 1:
                normalized_tensor = normalized_tensor.unsqueeze(0)
            torchaudio.save(normalized_path, normalized_tensor, sample_rate)
        
        print(f"🎵 Saved normalized audio to: {normalized_path}")
        clear_caches()  # Clear cache after generation
        return normalized_path
    
    clear_caches()  # Clear cache after generation
    return output_path

def generate_voice_clone(
    transcript,
    uploaded_voice,
    temperature,
    max_new_tokens,
    seed,
    top_k=50,
    top_p=0.95,
    min_p=0.0,
    repetition_penalty=1.0,
    ras_win_len=7,
    ras_win_max_num_repeat=2,
    do_sample=True
):
    # Initialize model if not already done
    initialize_model()
    
    # Validate inputs
    if not transcript.strip():
        raise ValueError("Please enter text to synthesize")
    
    if uploaded_voice is None or uploaded_voice[1] is None:
        raise ValueError("Please upload a voice sample for cloning")
    
    # Set seed for reproducibility
    if seed > 0:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Initialize temp paths to None to avoid NameError
    temp_audio_path = None
    temp_txt_path = None
    
    try:
        # Save uploaded audio to temporary file using enhanced method
        temp_audio_path = enhanced_save_temp_audio_fixed(uploaded_voice)
        
        # Create corresponding txt file with auto-transcription
        temp_txt_path = create_voice_reference_txt(temp_audio_path)  # Auto-transcribes!
        
        # Use the same pattern as the official generation.py
        # The serve engine expects the voice reference format like this:
        system_content = "Generate audio following instruction."
        
        # Create messages similar to how the official code does it
        # First, add the voice reference as a user-assistant pair
        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content="Please speak this text."),  # Dummy prompt for voice ref
            Message(role="assistant", content=AudioContent(audio_url=temp_audio_path)),
            Message(role="user", content=transcript)
        ]
        
        # Generate audio with optimizations
        min_p_value = min_p if min_p > 0 else None  # Convert 0 to None for disabled
        output = optimized_generate_audio(
            messages, max_new_tokens, temperature, top_k, top_p, min_p_value, 
            repetition_penalty, ras_win_len, ras_win_max_num_repeat, do_sample, use_cache=False
        )
        
        # Save and return audio with organized output
        output_path = get_output_path("voice_cloning", "cloned_voice")
        torchaudio.save(output_path, torch.from_numpy(output.audio)[None, :], output.sampling_rate)
        clear_caches()  # Clear cache after generation
        return output_path
    
    finally:
        # Clean up temporary files using robust cleanup
        robust_file_cleanup([temp_audio_path, temp_txt_path])

def generate_voice_clone_alternative(
    transcript,
    uploaded_voice,
    temperature,
    max_new_tokens,
    seed,
    top_k=50,
    top_p=0.95,
    min_p=0.0,
    repetition_penalty=1.0,
    ras_win_len=7,
    ras_win_max_num_repeat=2,
    do_sample=True
):
    """Alternative voice cloning method using voice_ref format"""
    # Initialize model if not already done
    initialize_model()
    
    # Validate inputs
    if not transcript.strip():
        raise ValueError("Please enter text to synthesize")
    
    if uploaded_voice is None or uploaded_voice[1] is None:
        raise ValueError("Please upload a voice sample for cloning")
    
    # Set seed for reproducibility
    if seed > 0:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Initialize temp path to None to avoid NameError
    temp_audio_path = None
    
    try:
        # Save uploaded audio to temporary file using enhanced method
        temp_audio_path = enhanced_save_temp_audio_fixed(uploaded_voice)
        
        # Try the voice_ref format (this might be specific to newer versions)
        system_content = "Generate audio following instruction."
        
        # The format you were using - let's make sure the path is correct
        user_content = f"<|voice_ref_start|>\n{temp_audio_path}\n<|voice_ref_end|>\n\n{transcript}"
        
        messages = [
            Message(role="system", content=system_content),
            Message(role="user", content=user_content)
        ]
        
        # Generate audio with optimizations
        min_p_value = min_p if min_p > 0 else None  # Convert 0 to None for disabled
        output = optimized_generate_audio(
            messages, max_new_tokens, temperature, top_k, top_p, min_p_value, 
            repetition_penalty, ras_win_len, ras_win_max_num_repeat, do_sample, use_cache=False
        )
        
        # Save and return audio with organized output
        output_path = get_output_path("voice_cloning", "cloned_voice_alt")
        torchaudio.save(output_path, torch.from_numpy(output.audio)[None, :], output.sampling_rate)
        clear_caches()  # Clear cache after generation
        return output_path
    
    finally:
        # Clean up temporary file using robust cleanup
        robust_file_cleanup(temp_audio_path)

def generate_longform(
    transcript,
    voice_choice,
    uploaded_voice,
    voice_prompt,
    temperature,
    max_new_tokens,
    seed,
    scene_description,
    chunk_size
):
    # Initialize model if not already done
    initialize_model()
    
    # Check if a Voice Library voice is selected and load its config
    use_voice_library_config = False
    voice_config = None
    if voice_choice == "Predefined Voice" and voice_prompt and voice_prompt.startswith("👤 "):
        voice_name = voice_prompt[2:]  # Remove "👤 " prefix
        voice_config = load_voice_config(voice_name)
        use_voice_library_config = True
        print(f"🎤 Using Voice Library config for '{voice_name}': temp={voice_config['temperature']}, tokens={voice_config['max_new_tokens']}")
        
        # Override parameters with Voice Library config
        temperature = voice_config['temperature']
        max_new_tokens = voice_config['max_new_tokens']
        # Note: seed from UI is still used for consistency
    
    # Set seed for reproducibility
    if seed > 0:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Smart chunking
    chunks = smart_chunk_text(transcript, max_chunk_size=chunk_size)
    
    # Handle voice reference setup
    temp_audio_path = None
    temp_txt_path = None
    voice_ref_path = None
    voice_ref_text = None
    first_chunk_audio_path = None
    first_chunk_text = None
    
    try:
        # Determine initial voice reference
        if voice_choice == "Upload Voice" and uploaded_voice is not None and uploaded_voice[1] is not None:
            temp_audio_path = enhanced_save_temp_audio_fixed(uploaded_voice)
            temp_txt_path = create_voice_reference_txt(temp_audio_path)  # Auto-transcribes!
            voice_ref_path = temp_audio_path
            # Read transcription
            if temp_txt_path and os.path.exists(temp_txt_path):
                with open(temp_txt_path, 'r', encoding='utf-8') as f:
                    voice_ref_text = f.read().strip()
            else:
                voice_ref_text = "This is a voice sample for cloning."
        elif voice_choice == "Predefined Voice" and voice_prompt != "None (Smart Voice)":
            ref_audio_path = get_voice_path(voice_prompt)
            if ref_audio_path and os.path.exists(ref_audio_path):
                voice_ref_path = ref_audio_path
                # Ensure txt file exists for predefined voices - use robust path creation
                txt_path = robust_txt_path_creation(ref_audio_path)
                
                if not os.path.exists(txt_path):
                    # Auto-transcribe instead of dummy text
                    try:
                        transcription = transcribe_audio(ref_audio_path)
                        with open(txt_path, 'w', encoding='utf-8') as f:
                            f.write(transcription)
                    except Exception as e:
                        print(f"⚠️ Failed to transcribe {ref_audio_path}: {e}")
                        with open(txt_path, 'w', encoding='utf-8') as f:
                            f.write("This is a voice sample.")
                
                # Read transcription
                if os.path.exists(txt_path):
                    with open(txt_path, 'r', encoding='utf-8') as f:
                        voice_ref_text = f.read().strip()
                else:
                    voice_ref_text = "This is a voice sample."
        
        # Prepare system message
        system_content = "Generate audio following instruction."
        if scene_description and scene_description.strip():
            system_content += f" <|scene_desc_start|>\n{scene_description}\n<|scene_desc_end|>"
        
        # Generate audio for each chunk
        full_audio = []
        sampling_rate = 24000
        
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}: {chunk[:50]}...")
            
            if voice_choice == "Upload Voice" and voice_ref_path and voice_ref_text:
                messages = [
                    Message(role="system", content=system_content),
                    Message(role="user", content=voice_ref_text),
                    Message(role="assistant", content=AudioContent(audio_url=voice_ref_path)),
                    Message(role="user", content=chunk)
                ]
            elif voice_choice == "Predefined Voice" and voice_ref_path and voice_ref_text:
                messages = [
                    Message(role="system", content=system_content),
                    Message(role="user", content=voice_ref_text),
                    Message(role="assistant", content=AudioContent(audio_url=voice_ref_path)),
                    Message(role="user", content=chunk)
                ]
            elif voice_choice == "Smart Voice":
                if i == 0:
                    # First chunk: let model pick a voice
                    messages = [
                        Message(role="system", content=system_content),
                        Message(role="user", content=chunk)
                    ]
                else:
                    # Use first chunk's audio and text as reference for all subsequent chunks
                    if first_chunk_audio_path and first_chunk_text:
                        messages = [
                            Message(role="system", content=system_content),
                            Message(role="user", content=first_chunk_text),
                            Message(role="assistant", content=AudioContent(audio_url=first_chunk_audio_path)),
                            Message(role="user", content=chunk)
                        ]
                    else:
                        # Fallback if voice_ref_path or voice_ref_text is not available (shouldn't happen with Smart Voice)
                        messages = [
                            Message(role="system", content=system_content),
                            Message(role="user", content=chunk)
                        ]
            else:
                # Fallback for other voice choices
                messages = [
                    Message(role="system", content=system_content),
                    Message(role="user", content=chunk)
                ]
            
            # Generate audio with optimizations - use Voice Library config if available
            if use_voice_library_config and voice_config:
                min_p_value = voice_config['min_p'] if voice_config['min_p'] > 0 else None
                output = optimized_generate_audio(
                    messages, voice_config['max_new_tokens'], voice_config['temperature'], 
                    voice_config['top_k'], voice_config['top_p'], min_p_value, 
                    voice_config['repetition_penalty'], voice_config['ras_win_len'], 
                    voice_config['ras_win_max_num_repeat'], voice_config['do_sample'], 
                    use_cache=True
                )
            else:
                output = optimized_generate_audio(messages, max_new_tokens, temperature, use_cache=True)
            
            if voice_choice == "Smart Voice" and i == 0:
                # Save first chunk's audio and text for reference
                first_chunk_audio_path = f"first_chunk_audio_{seed}_{hash(transcript[:20])}.wav"
                torchaudio.save(first_chunk_audio_path, torch.from_numpy(output.audio)[None, :], output.sampling_rate)
                first_chunk_text = chunk
            
            # Append audio
            full_audio.append(output.audio)
            sampling_rate = output.sampling_rate
        
        # Concatenate all audio chunks and save with organized output
        if full_audio:
            full_audio = np.concatenate(full_audio, axis=0)
            
            output_path = get_output_path("longform_generation", "longform_audio")
            torchaudio.save(output_path, torch.from_numpy(full_audio)[None, :], sampling_rate)
            clear_caches()  # Clear cache after generation
            return output_path
        else:
            clear_caches()
            return None
    
    finally:
        # Clean up temporary files using robust cleanup
        robust_file_cleanup([temp_audio_path, temp_txt_path, first_chunk_audio_path])

# IMPROVED HANDLER FUNCTION FOR MULTI-SPEAKER
def handle_multi_speaker_generation(
    transcript, voice_method, speaker0_audio, speaker1_audio, speaker2_audio,
    speaker0_voice, speaker1_voice, speaker2_voice, temperature, max_new_tokens, 
    seed, scene_description, auto_format
):
    # Prepare uploaded voices list with better validation
    uploaded_voices = []
    if voice_method == "Upload Voices":
        for i, audio in enumerate([speaker0_audio, speaker1_audio, speaker2_audio]):
            if audio is not None and len(audio) == 2 and audio[1] is not None:
                # Validate audio data
                sample_rate, audio_data = audio
                if isinstance(audio_data, np.ndarray) and len(audio_data) > 0:
                    uploaded_voices.append(audio)
                    print(f"✅ Added SPEAKER{i} audio: {len(audio_data)} samples at {sample_rate}Hz")
                else:
                    uploaded_voices.append(None)
                    print(f"⚠️ Invalid audio data for SPEAKER{i}")
            else:
                uploaded_voices.append(None)
                print(f"⚠️ No audio provided for SPEAKER{i}")
    
    # Prepare predefined voices list
    predefined_voices = []
    if voice_method == "Predefined Voices":
        predefined_voices = [speaker0_voice, speaker1_voice, speaker2_voice]
    
    return generate_multi_speaker(
        transcript, voice_method, uploaded_voices, predefined_voices,
        temperature, max_new_tokens, seed, scene_description, auto_format
    )

def generate_multi_speaker(
    transcript,
    voice_method,
    uploaded_voices,
    predefined_voices,
    temperature,
    max_new_tokens,
    seed,
    scene_description,
    auto_format,
    speaker_pause_duration=0.3
):
    # Initialize model if not already done
    initialize_model()
    
    # Set seed for reproducibility
    if seed > 0:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Auto-format transcript if requested
    if auto_format:
        transcript = auto_format_multi_speaker(transcript)
    
    # Parse speaker assignments
    speakers = parse_multi_speaker_text(transcript)
    if not speakers:
        raise ValueError("No speakers found in transcript. Use [SPEAKER0], [SPEAKER1] format or enable auto-format.")
    
    print(f"🎭 Found speakers: {list(speakers.keys())}")
    
    # Prepare voice references
    voice_refs = {}
    temp_files = []
    speaker_audio_refs = {}  # For smart voice consistency
    # NEW: Store both first audio path and first text for each speaker (for Smart Voice)
    speaker_first_refs = {}  # {speaker_id: (audio_path, text_content)}
    # NEW: Store uploaded audio and transcription for each speaker (for Upload Voices)
    uploaded_voice_refs = {}  # {speaker_id: (audio_path, transcription)}
    # NEW: Store voice configurations for each speaker (for Predefined Voices)
    voice_configs = {}  # {speaker_id: config_dict}
    
    try:
        if voice_method == "Upload Voices":
            # CRITICAL FIX: Handle uploaded voices properly for each speaker
            if uploaded_voices:
                for i, audio in enumerate(uploaded_voices):
                    if audio is not None and audio[1] is not None:
                        speaker_key = f"SPEAKER{i}"
                        print(f"🎤 Processing uploaded voice for {speaker_key}...")
                        print(f"📊 Audio data: {len(audio[1])} samples at {audio[0]}Hz")
                        # Save the uploaded audio properly using enhanced method
                        temp_path = enhanced_save_temp_audio_fixed(audio)
                        # CRITICAL: Create transcription for the voice reference
                        temp_txt_path = create_voice_reference_txt(temp_path)
                        # Read the transcription for use as reference text
                        if os.path.exists(temp_txt_path):
                            with open(temp_txt_path, 'r', encoding='utf-8') as f:
                                transcription = f.read().strip()
                        else:
                            transcription = "This is a voice sample for cloning."
                        # Store both audio path and transcription for this speaker
                        uploaded_voice_refs[speaker_key] = (temp_path, transcription)
                        temp_files.extend([temp_path, temp_txt_path])
                        print(f"✅ Setup voice reference for {speaker_key}: {temp_path}")
                        print(f"📝 Created transcription file: {temp_txt_path}")
                        print(f"📋 {speaker_key} transcription: '{transcription[:50]}...'")
            print(f"🎭 Upload Voices setup complete. Voice refs: {list(uploaded_voice_refs.keys())}")
                        
        elif voice_method == "Predefined Voices":
            # Handle predefined voices
            if predefined_voices:
                for i, voice_name in enumerate(predefined_voices):
                    if voice_name and voice_name != "None (Smart Voice)":
                        speaker_key = f"SPEAKER{i}"
                        ref_audio_path = get_voice_path(voice_name)
                        if ref_audio_path and os.path.exists(ref_audio_path):
                            voice_refs[speaker_key] = ref_audio_path
                            print(f"📁 Setup voice reference for {speaker_key}: {ref_audio_path}")
                            
                            # Load voice configuration for this speaker
                            voice_name_clean = voice_name[2:] if voice_name.startswith("👤 ") else voice_name
                            voice_config = load_voice_config(voice_name_clean)
                            voice_configs[speaker_key] = voice_config
                            print(f"🎛️ Loaded voice config for {speaker_key}: temp={voice_config['temperature']}, tokens={voice_config['max_new_tokens']}, top_k={voice_config['top_k']}")
                            
                            # Ensure txt file exists - use robust extension handling
                            txt_path = robust_txt_path_creation(ref_audio_path)
                            
                            if not os.path.exists(txt_path):
                                # Auto-transcribe instead of dummy text
                                try:
                                    transcription = transcribe_audio(ref_audio_path)
                                    with open(txt_path, 'w', encoding='utf-8') as f:
                                        f.write(transcription)
                                except Exception as e:
                                    print(f"⚠️ Failed to transcribe {ref_audio_path}: {e}")
                                    with open(txt_path, 'w', encoding='utf-8') as f:
                                        f.write("This is a voice sample.")
        
        # Prepare system message - SAME AS WORKING VOICE CLONING
        system_content = "Generate audio following instruction."
        
        if scene_description and scene_description.strip():
            system_content += f" <|scene_desc_start|>\n{scene_description}\n<|scene_desc_end|>"
        
        # Generate audio for each speaker segment
        full_audio = []
        sampling_rate = 24000
        
        # Process transcript line by line
        lines = transcript.split('\n')
        
        for line_idx, line in enumerate(lines):
            original_line = line  # Keep original line for index lookup
            line = line.strip()
            if not line:
                continue
                
            # Check if line has speaker tag
            speaker_match = re.match(r'\[SPEAKER(\d+)\]\s*(.*)', line)
            if speaker_match:
                speaker_id = f"SPEAKER{speaker_match.group(1)}"
                text_content = speaker_match.group(2).strip()
                
                if not text_content:
                    continue
                
                print(f"🎭 Generating for {speaker_id}: {text_content[:50]}...")
                
                # CRITICAL FIX: Prepare messages based on voice method
                # This logic determines which voice reference to use for each speaker line
                
                if voice_method == "Upload Voices" and speaker_id in uploaded_voice_refs:
                    # UPLOAD VOICES: Always use the uploaded voice sample and its transcription as reference
                    ref_audio_path, ref_text = uploaded_voice_refs[speaker_id]
                    print(f"🎤 Using UPLOADED voice reference for {speaker_id}: {ref_audio_path} with text: '{ref_text}'")
                    messages = [
                        Message(role="system", content=system_content),
                        Message(role="user", content=ref_text),
                        Message(role="assistant", content=AudioContent(audio_url=ref_audio_path)),
                        Message(role="user", content=text_content)
                    ]
                    
                elif voice_method == "Predefined Voices" and speaker_id in voice_refs:
                    # PREDEFINED VOICES: Use the predefined voice sample as reference
                    print(f"📁 Using PREDEFINED voice reference for {speaker_id}: {voice_refs[speaker_id]}")
                    messages = [
                        Message(role="system", content=system_content),
                        Message(role="user", content="Please speak this text."),
                        Message(role="assistant", content=AudioContent(audio_url=voice_refs[speaker_id])),
                        Message(role="user", content=text_content)
                    ]
                    
                elif voice_method == "Smart Voice":
                    # SMART VOICE: Use consistency logic with auto-generated references
                    if speaker_id in speaker_first_refs:
                        # Use the FIRST generated audio and text for this speaker as reference
                        first_audio_path, first_text = speaker_first_refs[speaker_id]
                        print(f"🔄 Using FIRST SMART voice reference for {speaker_id}: {first_audio_path} with text: '{first_text}'")
                        messages = [
                            Message(role="system", content=system_content),
                            Message(role="user", content=first_text),
                            Message(role="assistant", content=AudioContent(audio_url=first_audio_path)),
                            Message(role="user", content=text_content)
                        ]
                    else:
                        # First time for this speaker - let model pick voice
                        print(f"🆕 First occurrence of {speaker_id} in SMART mode, letting AI pick voice")
                        messages = [
                            Message(role="system", content=system_content),
                            Message(role="user", content=text_content)
                        ]
                        
                else:
                    # FALLBACK: This should only happen if no voice reference is available
                    print(f"⚠️ FALLBACK: No voice reference found for {speaker_id} in {voice_method} mode")
                    print(f"📋 Available voice_refs: {list(voice_refs.keys())}")
                    print(f"📋 Available speaker_audio_refs: {list(speaker_audio_refs.keys())}")
                    messages = [
                        Message(role="system", content=system_content),
                        Message(role="user", content=text_content)
                    ]
                
                print(f"📝 Generating audio for: '{text_content}'")
                
                # Generate audio with optimizations - use voice-specific config if available
                if voice_method == "Predefined Voices" and speaker_id in voice_configs:
                    # Use the voice's individual configuration
                    config = voice_configs[speaker_id]
                    min_p_value = config['min_p'] if config['min_p'] > 0 else None
                    print(f"🎛️ Using {speaker_id} voice config: temp={config['temperature']}, tokens={config['max_new_tokens']}")
                    output = optimized_generate_audio(
                        messages, config['max_new_tokens'], config['temperature'], 
                        config['top_k'], config['top_p'], min_p_value, 
                        config['repetition_penalty'], config['ras_win_len'], 
                        config['ras_win_max_num_repeat'], config['do_sample'], use_cache=False
                    )
                else:
                    # Use global settings for Upload Voices or Smart Voice
                    output = optimized_generate_audio(messages, max_new_tokens, temperature, use_cache=False)
                
                # IMPORTANT: Smart Voice consistency logic ONLY applies to Smart Voice mode
                # For Upload Voices and Predefined Voices, we already have the voice references
                if voice_method == "Smart Voice" and speaker_id not in speaker_first_refs:
                    # Save the first generated audio and text for this speaker for future consistency
                    speaker_audio_path = f"temp_speaker_{speaker_id}_{seed}_{int(time.time())}.wav"
                    torchaudio.save(speaker_audio_path, torch.from_numpy(output.audio)[None, :], output.sampling_rate)
                    # Small delay to ensure file is written
                    time.sleep(0.1)
                    # CRITICAL: Use auto-transcription for voice reference
                    transcribed_text = transcribe_audio(speaker_audio_path)
                    speaker_txt_path = speaker_audio_path.replace('.wav', '.txt')
                    with open(speaker_txt_path, 'w', encoding='utf-8') as f:
                        f.write(transcribed_text)
                    # Save both audio path and the first text_content
                    speaker_first_refs[speaker_id] = (speaker_audio_path, text_content)
                    temp_files.extend([speaker_audio_path, speaker_txt_path])
                    print(f"✅ Saved {speaker_id} FIRST SMART voice reference: {speaker_audio_path}")
                    print(f"📝 Auto-transcribed: '{transcribed_text[:50]}...'")
                    # Verify files exist
                    if os.path.exists(speaker_audio_path) and os.path.exists(speaker_txt_path):
                        print(f"✅ Voice reference files verified for {speaker_id}")
                    else:
                        print(f"⚠️ Warning: Voice reference files not created properly for {speaker_id}")
                
                # For Upload Voices and Predefined Voices, we DON'T save additional references
                # because we already have the uploaded/predefined voice samples
                
                # Validate output before adding
                if output.audio is not None and len(output.audio) > 0:
                    full_audio.append(output.audio)
                    sampling_rate = output.sampling_rate
                    print(f"✅ Added audio segment (length: {len(output.audio)} samples)")
                else:
                    print(f"⚠️ Empty or invalid audio output for: '{text_content}'")
                
                # Add a small pause between different speakers (not between same speaker)
                if len(full_audio) > 1:
                    # Check if this is a different speaker than the previous line
                    prev_line_idx = line_idx - 1
                    if prev_line_idx >= 0:
                        prev_line = lines[prev_line_idx].strip()
                        if prev_line:
                            prev_match = re.match(r'\[SPEAKER(\d+)\]', prev_line)
                            if prev_match and prev_match.group(1) != speaker_match.group(1):
                                # Different speaker, add pause
                                if speaker_pause_duration > 0:
                                    pause_samples = int(speaker_pause_duration * sampling_rate)
                                    pause_audio = np.zeros(pause_samples, dtype=np.float32)
                                    full_audio.append(pause_audio)
                                    print(f"🔇 Added {speaker_pause_duration}s pause between speakers")
                                else:
                                    print(f"🔇 No pause between speakers (disabled)")
        
        # Concatenate all audio chunks and save with organized output
        if full_audio:
            full_audio = np.concatenate(full_audio, axis=0)
            
            output_path = get_output_path("multi_speaker", "multi_speaker_audio")
            torchaudio.save(output_path, torch.from_numpy(full_audio)[None, :], sampling_rate)
            
            print(f"🎉 Multi-speaker audio generated successfully: {output_path}")
            clear_caches()  # Clear cache after generation
            return output_path
        else:
            clear_caches()
            raise ValueError("No audio was generated. Check your transcript format and voice samples.")
    
    except Exception as e:
        print(f"❌ Error in multi-speaker generation: {e}")
        raise e
    
    finally:
        # Clean up temporary files using robust cleanup
        robust_file_cleanup(temp_files)

def refresh_voice_list():
    updated_voices = get_all_available_voices()
    return gr.update(choices=updated_voices)

def refresh_voice_list_multi():
    """Refresh voice list for multi-speaker (returns 3 updates)"""
    updated_voices = get_all_available_voices()
    return [gr.update(choices=updated_voices), gr.update(choices=updated_voices), gr.update(choices=updated_voices)]

def refresh_library_list():
    library_voices = ["None"] + get_voice_library_voices()
    return gr.update(choices=library_voices)

# Check audio processing capabilities at startup
check_dependencies()

# Gradio interface
with gr.Blocks(title="Higgs Audio v2 Generator") as demo:
    gr.HTML('<h1 style="text-align:center; margin-bottom:0.2em;"><a href="https://github.com/Saganaki22/higgs-audio-WebUI" target="_blank" style="text-decoration:none; color:inherit;">🎵 Higgs Audio v2 WebUI</a></h1>')
    gr.HTML('<div style="text-align:center; font-size:1.2em; margin-bottom:1.5em;">Generate high-quality speech from text with voice cloning, longform generation, multi speaker generation, voice library, smart batching</div>')
    with gr.Tabs():
        # Tab 1: Basic Generation with Predefined Voices
        with gr.Tab("Basic Generation"):
            with gr.Row():
                with gr.Column():
                    basic_transcript = gr.TextArea(
                        label="Transcript",
                        placeholder="Enter text to synthesize...",
                        value="The sun rises in the east and sets in the west.",
                        lines=5
                    )
                    
                    with gr.Accordion("Voice Settings", open=True):
                        basic_voice_prompt = gr.Dropdown(
                            choices=available_voices,
                            value="None (Smart Voice)",
                            label="Predefined Voice Prompts"
                        )
                        basic_refresh_voices = gr.Button("Refresh Voice List")
                        
                        basic_scene_description = gr.TextArea(
                            label="Scene Description",
                            placeholder="Describe the recording environment...",
                            value=""
                        )
                    
                    with gr.Accordion("Generation Parameters", open=False):
                        with gr.Row():
                            with gr.Column():
                                basic_temperature = gr.Slider(
                                    minimum=0.1,
                                    maximum=1.0,
                                    value=0.3,
                                    step=0.05,
                                    label="Temperature",
                                    info="Controls randomness in generation (lower = more consistent)"
                                )
                                basic_max_new_tokens = gr.Slider(
                                    minimum=128,
                                    maximum=2048,
                                    value=1024,
                                    step=128,
                                    label="Max New Tokens"
                                )
                                basic_seed = gr.Number(
                                    label="Seed (0 for random)",
                                    value=12345,
                                    precision=0
                                )
                            
                            with gr.Column():
                                basic_top_k = gr.Slider(
                                    minimum=1,
                                    maximum=100,
                                    value=50,
                                    step=1,
                                    label="Top-K",
                                    info="Limits vocabulary to top K most likely tokens"
                                )
                                basic_top_p = gr.Slider(
                                    minimum=0.1,
                                    maximum=1.0,
                                    value=0.95,
                                    step=0.05,
                                    label="Top-P (Nucleus Sampling)",
                                    info="Cumulative probability threshold for token selection"
                                )
                                basic_min_p = gr.Slider(
                                    minimum=0.0,
                                    maximum=0.2,
                                    value=0.0,
                                    step=0.01,
                                    label="Min-P",
                                    info="Minimum probability threshold (0 = disabled)"
                                )
                        
                        with gr.Accordion("Advanced Sampling", open=False):
                            with gr.Row():
                                with gr.Column():
                                    basic_repetition_penalty = gr.Slider(
                                        minimum=0.8,
                                        maximum=1.2,
                                        value=1.0,
                                        step=0.05,
                                        label="Repetition Penalty",
                                        info="Penalty for repeating tokens (1.0 = no penalty)"
                                    )
                                    basic_do_sample = gr.Checkbox(
                                        label="Enable Sampling",
                                        value=True,
                                        info="Use sampling vs greedy decoding"
                                    )
                                
                                with gr.Column():
                                    basic_ras_win_len = gr.Slider(
                                        minimum=0,
                                        maximum=20,
                                        value=7,
                                        step=1,
                                        label="RAS Window Length",
                                        info="Repetition detection window (0 = disabled)"
                                    )
                                    basic_ras_win_max_num_repeat = gr.Slider(
                                        minimum=1,
                                        maximum=5,
                                        value=2,
                                        step=1,
                                        label="RAS Max Repeats",
                                        info="Maximum allowed repetitions in window"
                                    )
                    
                    with gr.Accordion("🔊 Volume Normalization", open=False):
                        gr.Markdown("*Optional: Normalize audio volume for consistent playback*")
                        with gr.Row():
                            basic_enable_normalization = gr.Checkbox(
                                label="Enable Volume Normalization", 
                                value=False,
                                info="Normalize audio volume level"
                            )
                            basic_target_volume = gr.Slider(
                                0.05, 0.3, 
                                value=0.15, 
                                step=0.01, 
                                label="Target Volume",
                                info="RMS level (0.15 = moderate)"
                            )
                    
                    basic_generate_btn = gr.Button("Generate Audio", variant="primary")
                
                with gr.Column():
                    basic_output_audio = gr.Audio(label="Generated Audio", type="filepath", show_download_button=True)
                    gr.HTML('''
                    <div style="background:#23272e;border-radius:8px;padding:1em 1.5em;margin-top:1em;border-left:5px solid #ff9800;max-width:420px;margin-left:auto;margin-right:auto;">
                        <b>💡 Tips for Basic Generation:</b><br>
                        • For best results, use clear, natural sentences.<br>
                        • You can select a predefined voice or use Smart Voice for random high-quality voices.<br>
                        • Scene description can help set the environment (e.g., "in a quiet room").<br>
                        • Adjust temperature for more/less expressive speech.<br>
                        • Try different seeds for voice variety.
                    </div>
                    ''')
        
        # Tab 2: Voice Cloning (YOUR voice only)
        with gr.Tab("Voice Cloning"):
            with gr.Row():
                with gr.Column():
                    vc_transcript = gr.TextArea(
                        label="Transcript",
                        placeholder="Enter text to synthesize with your voice...",
                        value="Hello, this is my cloned voice speaking!",
                        lines=5
                    )
                    
                    with gr.Accordion("Voice Cloning", open=True):
                        gr.Markdown("### Upload Your Voice Sample")
                        vc_uploaded_voice = gr.Audio(label="Upload Voice Sample", type="numpy")
                        if WHISPER_AVAILABLE:
                            gr.Markdown("*Record 10-30 seconds of clear speech for best results. Audio will be auto-transcribed!* ✨")
                        else:
                            gr.Markdown("*Record 10-30 seconds of clear speech for best results. Install whisper for auto-transcription: `pip install faster-whisper`*")
                        
                        # Add a toggle to switch between methods
                        vc_method = gr.Radio(
                            choices=["Official Method", "Alternative Method"],
                            value="Official Method",
                            label="Voice Cloning Method"
                        )
                    
                    with gr.Accordion("Generation Parameters", open=False):
                        with gr.Row():
                            with gr.Column():
                                vc_temperature = gr.Slider(
                                    minimum=0.1,
                                    maximum=1.0,
                                    value=0.3,
                                    step=0.05,
                                    label="Temperature",
                                    info="Controls randomness in generation (lower = more consistent)"
                                )
                                vc_max_new_tokens = gr.Slider(
                                    minimum=128,
                                    maximum=2048,
                                    value=1024,
                                    step=128,
                                    label="Max New Tokens"
                                )
                                vc_seed = gr.Number(
                                    label="Seed (0 for random)",
                                    value=12345,
                                    precision=0
                                )
                            
                            with gr.Column():
                                vc_top_k = gr.Slider(
                                    minimum=1,
                                    maximum=100,
                                    value=50,
                                    step=1,
                                    label="Top-K",
                                    info="Limits vocabulary to top K most likely tokens"
                                )
                                vc_top_p = gr.Slider(
                                    minimum=0.1,
                                    maximum=1.0,
                                    value=0.95,
                                    step=0.05,
                                    label="Top-P (Nucleus Sampling)",
                                    info="Cumulative probability threshold for token selection"
                                )
                                vc_min_p = gr.Slider(
                                    minimum=0.0,
                                    maximum=0.2,
                                    value=0.0,
                                    step=0.01,
                                    label="Min-P",
                                    info="Minimum probability threshold (0 = disabled)"
                                )
                        
                        with gr.Accordion("Advanced Sampling", open=False):
                            with gr.Row():
                                with gr.Column():
                                    vc_repetition_penalty = gr.Slider(
                                        minimum=0.8,
                                        maximum=1.2,
                                        value=1.0,
                                        step=0.05,
                                        label="Repetition Penalty",
                                        info="Penalty for repeating tokens (1.0 = no penalty)"
                                    )
                                    vc_do_sample = gr.Checkbox(
                                        label="Enable Sampling",
                                        value=True,
                                        info="Use sampling vs greedy decoding"
                                    )
                                
                                with gr.Column():
                                    vc_ras_win_len = gr.Slider(
                                        minimum=0,
                                        maximum=20,
                                        value=7,
                                        step=1,
                                        label="RAS Window Length",
                                        info="Repetition detection window (0 = disabled)"
                                    )
                                    vc_ras_win_max_num_repeat = gr.Slider(
                                        minimum=1,
                                        maximum=5,
                                        value=2,
                                        step=1,
                                        label="RAS Max Repeats",
                                        info="Maximum allowed repetitions in window"
                                    )
                    
                    vc_generate_btn = gr.Button("Clone My Voice & Generate", variant="primary")
                
                with gr.Column():
                    vc_output_audio = gr.Audio(label="Cloned Voice Audio", type="filepath", show_download_button=True)
                    gr.HTML('''
                    <div style="background:#23272e;border-radius:8px;padding:1em 1.5em;margin-top:1em;border-left:5px solid #4caf50;max-width:420px;margin-left:auto;margin-right:auto;">
                        <b>💡 Tips for Voice Cloning:</b><br>
                        • Upload a clear 10-30 second sample of your voice, speaking naturally.<br>
                        • The sample will be auto-transcribed for best cloning results.<br>
                        • Use the "Official Method" for most cases; try "Alternative Method" if you want to experiment.<br>
                        • Longer, more expressive samples improve cloning quality.<br>
                        • Use the same seed to reproduce results.
                    </div>
                    ''')
        
        # Tab 3: Long-form Generation
        with gr.Tab("Long-form Generation"):
            with gr.Row():
                with gr.Column():
                    lf_transcript = gr.TextArea(
                        label="Long Transcript",
                        placeholder="Enter long text to synthesize...",
                        value="Artificial intelligence is transforming our world. It helps solve complex problems in healthcare, climate, and education. Machine learning algorithms can process vast amounts of data to find patterns humans might miss. As we develop these technologies, we must consider their ethical implications. The future of AI holds both incredible promise and significant challenges.",
                        lines=10
                    )
                    
                    with gr.Accordion("Voice Options", open=True):
                        lf_voice_choice = gr.Radio(
                            choices=["Smart Voice", "Upload Voice", "Predefined Voice"],
                            value="Smart Voice",
                            label="Voice Selection Method"
                        )
                        
                        with gr.Group(visible=False) as lf_upload_group:
                            lf_uploaded_voice = gr.Audio(label="Upload Voice Sample", type="numpy")
                            if WHISPER_AVAILABLE:
                                gr.Markdown("*Audio will be auto-transcribed for voice cloning!* ✨")
                            else:
                                gr.Markdown("*Install whisper for auto-transcription: `pip install faster-whisper`*")
                        
                        with gr.Group(visible=False) as lf_predefined_group:
                            lf_voice_prompt = gr.Dropdown(
                                choices=available_voices,
                                value="None (Smart Voice)",
                                label="Predefined Voice Prompts"
                            )
                            lf_refresh_voices = gr.Button("Refresh Voice List")
                        
                        lf_scene_description = gr.TextArea(
                            label="Scene Description",
                            placeholder="Describe the recording environment...",
                            value=""
                        )
                    
                    with gr.Accordion("Generation Parameters", open=False):
                        lf_temperature = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.3,
                            step=0.05,
                            label="Temperature"
                        )
                        lf_max_new_tokens = gr.Slider(
                            minimum=128,
                            maximum=2048,
                            value=1024,
                            step=128,
                            label="Max New Tokens per Chunk"
                        )
                        lf_seed = gr.Number(
                            label="Seed (0 for random)",
                            value=12345,
                            precision=0
                        )
                        lf_chunk_size = gr.Slider(
                            minimum=100,
                            maximum=500,
                            value=200,
                            step=50,
                            label="Characters per Chunk"
                        )
                    
                    lf_generate_btn = gr.Button("Generate Long-form Audio", variant="primary")
                
                with gr.Column():
                    lf_output_audio = gr.Audio(label="Generated Audio", type="filepath", show_download_button=True)
                    gr.HTML('''
                    <div style="background:#23272e;border-radius:8px;padding:1em 1.5em;margin-top:1em;border-left:5px solid #2196f3;max-width:420px;margin-left:auto;margin-right:auto;">
                        <b>💡 Tips for Long-form Generation:</b><br>
                        • Paste or write long text (stories, articles, etc.) for continuous speech.<br>
                        • Choose Smart Voice, upload your own, or select a predefined voice.<br>
                        • Adjust chunk size for smoother transitions (smaller = more natural, larger = faster).<br>
                        • Scene description can set the mood or environment.<br>
                        • Use consistent voice references for best results in long texts.
                    </div>
                    ''')
            
            # Visibility logic for voice options
            def update_voice_options(choice):
                return {
                    lf_upload_group: gr.update(visible=choice == "Upload Voice"),
                    lf_predefined_group: gr.update(visible=choice == "Predefined Voice")
                }
            
            lf_voice_choice.change(
                fn=update_voice_options,
                inputs=lf_voice_choice,
                outputs=[lf_upload_group, lf_predefined_group]
            )
        
        # Tab 4: Dynamic Multi-Speaker Generation
        with gr.Tab("Multi-Speaker Generation"):
            with gr.Row():
                with gr.Column():
                    ms_transcript = gr.TextArea(
                        label="Multi-Speaker Transcript",
                        placeholder="Enter dialogue with any speaker names in brackets:\n[Alice] Hello there!\n[Bob] How are you?\n[Charlie] Great to see you both!",
                        value="[Alice] Hello there, how are you doing today?\n[Bob] I'm doing great, thank you for asking! How about yourself?\n[Alice] I'm fantastic! It's such a beautiful day outside.\n[Bob] Yes, it really is. Perfect weather for a walk in the park.",
                        lines=8
                    )
                    
                    # Process Text Button
                    ms_process_btn = gr.Button("🔍 Process Text & Detect Speakers", variant="secondary", size="lg")
                    
                    # Speaker Detection Results
                    ms_speaker_info = gr.Markdown("*Click 'Process Text' to detect speakers in your dialogue*")
                    
                    with gr.Accordion("Voice Configuration", open=True):
                        ms_voice_method = gr.Radio(
                            choices=["Smart Voice", "Upload Voices", "Voice Library"],
                            value="Smart Voice",
                            label="Voice Method"
                        )
                        
                        # Dynamic speaker assignment area (initially hidden)
                        with gr.Column(visible=False) as ms_speaker_assignment:
                            gr.Markdown("### 🎭 Speaker Voice Assignment")
                            ms_assignment_content = gr.Markdown("*Speaker assignments will appear here*")
                        
                        # Smart Voice info
                        with gr.Group() as ms_smart_voice_group:
                            gr.Markdown("### Smart Voice Mode")
                            gr.Markdown("*AI will automatically assign distinct voices to each detected speaker*")
                        
                        # Upload voices area (initially hidden)  
                        with gr.Group(visible=False) as ms_upload_group:
                            gr.Markdown("### Upload Voice Samples")
                            gr.Markdown("*Upload a voice sample for each speaker. Files will be auto-transcribed.*")
                            
                            # Create upload slots for up to 10 speakers
                            ms_upload_slots = []
                            for i in range(10):
                                with gr.Row(visible=False) as upload_row:
                                    speaker_audio = gr.Audio(
                                        label=f"Speaker {i} Voice Sample",
                                        type="numpy",
                                        scale=3
                                    )
                                    ms_upload_slots.append((upload_row, speaker_audio))
                        
                        # Voice library selection area (initially hidden)
                        with gr.Group(visible=False) as ms_library_group:
                            gr.Markdown("### Select Voices from Library")
                            gr.Markdown("*Choose a saved voice for each speaker*")
                            
                            # Create dropdown slots for up to 10 speakers
                            ms_library_slots = []
                            for i in range(10):
                                speaker_dropdown = gr.Dropdown(
                                    choices=get_current_available_voices(),
                                    value="None (Smart Voice)",
                                    label=f"Speaker {i} Voice",
                                    visible=False
                                )
                                ms_library_slots.append(speaker_dropdown)
                            
                            ms_refresh_library_voices = gr.Button("Refresh Voice Library", visible=False)
                        
                        ms_scene_description = gr.TextArea(
                            label="Scene Description",
                            placeholder="Describe the conversation setting...",
                            value="A friendly conversation between people in a quiet room."
                        )
                    
                    with gr.Accordion("Generation Parameters", open=False):
                        ms_temperature = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.3,
                            step=0.05,
                            label="Temperature"
                        )
                        ms_max_new_tokens = gr.Slider(
                            minimum=128,
                            maximum=2048,
                            value=1024,
                            step=128,
                            label="Max New Tokens per Segment"
                        )
                        ms_seed = gr.Number(
                            label="Seed (0 for random)",
                            value=12345,
                            precision=0
                        )
                    
                    with gr.Accordion("🔊 Volume Normalization", open=True):
                        gr.Markdown("*Fix volume inconsistencies between speakers - recommended for all multi-speaker audio*")
                        with gr.Row():
                            with gr.Column(scale=3):
                                ms_enable_normalization = gr.Checkbox(
                                    label="Enable Volume Normalization", 
                                    value=True,
                                    info="Automatically balance speaker volumes"
                                )
                            with gr.Column(scale=3):
                                ms_normalization_method = gr.Dropdown(
                                    choices=["adaptive", "simple", "segment-based"],
                                    value="adaptive",
                                    label="Normalization Method",
                                    info="Adaptive = sliding windows, Simple = whole audio, Segment = detect speakers"
                                )
                            with gr.Column(scale=2):
                                ms_target_volume = gr.Slider(
                                    0.05, 0.3, 
                                    value=0.15, 
                                    step=0.01, 
                                    label="Target Volume",
                                    info="RMS level (0.15 = moderate)"
                                )
                    
                    with gr.Accordion("⏸️ Speaker Timing", open=False):
                        gr.Markdown("*Control timing and pauses between different speakers*")
                        ms_speaker_pause = gr.Slider(
                            0.0, 2.0, 
                            value=0.3, 
                            step=0.1, 
                            label="Pause Between Speakers (seconds)",
                            info="Duration of silence when speakers change (0.0 = no pause, 0.3 = default)"
                        )
                    
                    ms_generate_btn = gr.Button("Generate Multi-Speaker Audio", variant="primary", interactive=False)
                
                with gr.Column():
                    ms_output_audio = gr.Audio(label="Generated Multi-Speaker Audio", type="filepath", show_download_button=True)
                    gr.HTML('''
                    <div style="background:#23272e;border-radius:8px;padding:1em 1.5em;margin-top:1em;border-left:5px solid #e91e63;max-width:420px;margin-left:auto;margin-right:auto;">
                        <b>💡 New Dynamic Multi-Speaker System:</b><br>
                        • Use ANY speaker names in brackets: [Alice], [Bob], [Character Name]<br>
                        • Click "Process Text" to automatically detect all speakers<br>
                        • Supports unlimited number of speakers<br>
                        • Smart Voice: AI assigns distinct voices automatically<br>
                        • Upload Voices: Clone real voices for each speaker<br>
                        • Voice Library: Use your saved voices for each speaker<br>
                        • Works with your existing scripts - no need to change character names!
                    </div>
                    ''')
            
            # State variables to store detected speakers and assignments
            ms_detected_speakers = gr.State([])
            ms_speaker_mapping = gr.State({})
        
        # Tab 5: Voice Library Management
        with gr.Tab("Voice Library"):
            gr.HTML("<h2 style='text-align: center;'>🎵 Voice Library Management</h2>")
            gr.HTML("<p style='text-align: center;'>Save voices with custom generation parameters for perfect reuse!</p>")
            
            with gr.Row():
                # Left Column: Add New Voice
                with gr.Column(scale=1):
                    gr.Markdown("## 🎤 Add New Voice")
                    
                    # Step 1: Upload
                    gr.Markdown("### Step 1: Upload Voice Sample")
                    vl_new_voice_audio = gr.Audio(label="Upload Voice Sample", type="filepath")
                    
                    # Step 2: Configure Parameters
                    gr.Markdown("### Step 2: Configure Generation Parameters")
                    with gr.Accordion("Generation Parameters", open=True):
                        with gr.Row():
                            with gr.Column():
                                vl_temperature = gr.Slider(
                                    minimum=0.1, maximum=1.0, value=0.3, step=0.05,
                                    label="Temperature", info="Controls randomness"
                                )
                                vl_max_new_tokens = gr.Slider(
                                    minimum=128, maximum=2048, value=1024, step=128,
                                    label="Max New Tokens"
                                )
                                vl_seed = gr.Number(
                                    label="Seed (0 for random)", value=12345, precision=0
                                )
                            
                            with gr.Column():
                                vl_top_k = gr.Slider(
                                    minimum=1, maximum=100, value=50, step=1,
                                    label="Top-K", info="Vocabulary limit"
                                )
                                vl_top_p = gr.Slider(
                                    minimum=0.1, maximum=1.0, value=0.95, step=0.05,
                                    label="Top-P", info="Nucleus sampling"
                                )
                                vl_min_p = gr.Slider(
                                    minimum=0.0, maximum=0.2, value=0.0, step=0.01,
                                    label="Min-P", info="Min probability (0 = disabled)"
                                )
                        
                        with gr.Accordion("Advanced Parameters", open=False):
                            with gr.Row():
                                with gr.Column():
                                    vl_repetition_penalty = gr.Slider(
                                        minimum=0.8, maximum=1.2, value=1.0, step=0.05,
                                        label="Repetition Penalty", info="Prevent repetition"
                                    )
                                    vl_do_sample = gr.Checkbox(
                                        label="Enable Sampling", value=True, info="Use sampling vs greedy"
                                    )
                                
                                with gr.Column():
                                    vl_ras_win_len = gr.Slider(
                                        minimum=0, maximum=20, value=7, step=1,
                                        label="RAS Window Length", info="Repetition window"
                                    )
                                    vl_ras_win_max_num_repeat = gr.Slider(
                                        minimum=1, maximum=5, value=2, step=1,
                                        label="RAS Max Repeats", info="Max allowed repeats"
                                    )
                    
                    # Step 3: Test & Save
                    gr.Markdown("### Step 3: Test & Save")
                    vl_test_text = gr.Textbox(
                        label="Test Text", 
                        placeholder="Enter text to test your voice with these settings...",
                        value="This is a test of my voice with custom parameters.",
                        lines=3
                    )
                    
                    with gr.Row():
                        vl_test_btn = gr.Button("🎵 Test Voice", variant="primary")
                        vl_clear_test_btn = gr.Button("🔄 Clear Test", variant="secondary")
                    
                    vl_new_voice_name = gr.Textbox(
                        label="Voice Name", 
                        placeholder="Enter a unique name for this voice..."
                    )
                    vl_voice_description = gr.Textbox(
                        label="Description (Optional)", 
                        placeholder="Describe this voice or when to use it...",
                        lines=2
                    )
                    
                    vl_save_btn = gr.Button("💾 Save Voice to Library", variant="stop", size="lg")
                    
                    if WHISPER_AVAILABLE:
                        gr.HTML("<p><em>✨ Voice will be auto-transcribed when saved!</em></p>")
                
                # Right Column: Manage Existing Voices
                with gr.Column(scale=1):
                    gr.Markdown("## 🗂️ Manage Voice Library")
                    
                    # Voice Selection & Info
                    vl_voice_selector = gr.Dropdown(
                        label="Select Voice to Manage",
                        choices=["None"] + get_voice_library_voices(),
                        value="None"
                    )
                    
                    # Voice Info Display
                    vl_voice_info = gr.Markdown("*Select a voice to view details*")
                    
                    # Voice Parameters (for editing existing voices)
                    with gr.Accordion("Edit Voice Parameters", open=False) as vl_edit_accordion:
                        gr.Markdown("*Modify generation parameters for the selected voice*")
                        with gr.Row():
                            with gr.Column():
                                vl_edit_temperature = gr.Slider(
                                    minimum=0.1, maximum=1.0, value=0.3, step=0.05,
                                    label="Temperature"
                                )
                                vl_edit_max_new_tokens = gr.Slider(
                                    minimum=128, maximum=2048, value=1024, step=128,
                                    label="Max New Tokens"
                                )
                                vl_edit_seed = gr.Number(
                                    label="Seed", value=12345, precision=0
                                )
                            
                            with gr.Column():
                                vl_edit_top_k = gr.Slider(
                                    minimum=1, maximum=100, value=50, step=1,
                                    label="Top-K"
                                )
                                vl_edit_top_p = gr.Slider(
                                    minimum=0.1, maximum=1.0, value=0.95, step=0.05,
                                    label="Top-P"
                                )
                                vl_edit_min_p = gr.Slider(
                                    minimum=0.0, maximum=0.2, value=0.0, step=0.01,
                                    label="Min-P"
                                )
                        
                        with gr.Row():
                            with gr.Column():
                                vl_edit_repetition_penalty = gr.Slider(
                                    minimum=0.8, maximum=1.2, value=1.0, step=0.05,
                                    label="Repetition Penalty"
                                )
                                vl_edit_do_sample = gr.Checkbox(
                                    label="Enable Sampling", value=True
                                )
                            
                            with gr.Column():
                                vl_edit_ras_win_len = gr.Slider(
                                    minimum=0, maximum=20, value=7, step=1,
                                    label="RAS Window Length"
                                )
                                vl_edit_ras_win_max_num_repeat = gr.Slider(
                                    minimum=1, maximum=5, value=2, step=1,
                                    label="RAS Max Repeats"
                                )
                        
                        vl_edit_description = gr.Textbox(
                            label="Description", 
                            placeholder="Describe this voice...",
                            lines=2
                        )
                    
                    # Management Buttons
                    with gr.Row():
                        vl_refresh_btn = gr.Button("🔄 Refresh List", variant="secondary")
                        vl_save_changes_btn = gr.Button("💾 Save Changes", variant="primary")
                        vl_delete_btn = gr.Button("🗑️ Delete Voice", variant="stop")
                    
                    # Test Results & Status
                    vl_test_audio = gr.Audio(label="🎧 Voice Test Result", type="filepath", show_download_button=True)
                    vl_test_status = gr.Textbox(label="Test Status", interactive=False)
                    vl_save_status = gr.Textbox(label="Save Status", interactive=False)
                    vl_delete_status = gr.Textbox(label="Management Status", interactive=False)

    # Function to handle voice cloning method selection
    def handle_voice_clone_generation(
        transcript, uploaded_voice, temperature, max_new_tokens, seed, method,
        top_k, top_p, min_p, repetition_penalty, ras_win_len, ras_win_max_num_repeat, do_sample
    ):
        if method == "Official Method":
            return generate_voice_clone(transcript, uploaded_voice, temperature, max_new_tokens, seed,
                                      top_k, top_p, min_p, repetition_penalty, ras_win_len, ras_win_max_num_repeat, do_sample)
        else:
            return generate_voice_clone_alternative(transcript, uploaded_voice, temperature, max_new_tokens, seed,
                                                  top_k, top_p, min_p, repetition_penalty, ras_win_len, ras_win_max_num_repeat, do_sample)
    
    # Voice Library Event Handlers
    def handle_test_voice_with_params(audio_data, test_text, temperature, max_new_tokens, seed,
                                    top_k, top_p, min_p, repetition_penalty, ras_win_len, 
                                    ras_win_max_num_repeat, do_sample):
        """Test voice with custom generation parameters - isolated test that doesn't interfere with voice library"""
        if audio_data is None:
            return None, "❌ Please upload an audio sample first"
        
        if not test_text.strip():
            test_text = "This is a test of my voice with custom parameters."
        
        # Initialize model if not already done
        initialize_model()
        
        # Create unique temporary file paths to avoid conflicts
        import tempfile
        import uuid
        temp_audio_path = None
        temp_txt_path = None
        
        try:
            # Set seed for reproducibility
            if seed > 0:
                torch.manual_seed(seed)
                np.random.seed(seed)
            
            # Create a unique temporary audio file for testing (isolated from voice library)
            unique_id = uuid.uuid4().hex[:8]
            temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=f"_test_{unique_id}.wav").name
            
            # Copy the uploaded audio to our isolated temp location
            import shutil
            shutil.copy2(audio_data, temp_audio_path)
            
            # Create a corresponding text file for voice reference (isolated)
            temp_txt_path = temp_audio_path.replace('.wav', '.txt')
            
            # Auto-transcribe for voice reference
            try:
                transcription = transcribe_audio(temp_audio_path)
                with open(temp_txt_path, 'w', encoding='utf-8') as f:
                    f.write(transcription)
            except Exception as e:
                # Fallback to dummy text if transcription fails
                with open(temp_txt_path, 'w', encoding='utf-8') as f:
                    f.write("This is a voice sample for testing.")
                print(f"⚠️ Transcription failed, using fallback: {e}")
            
            # Create messages for voice generation
            system_content = "Generate audio following instruction."
            messages = [
                Message(role="system", content=system_content),
                Message(role="user", content="Please speak this text."),
                Message(role="assistant", content=AudioContent(audio_url=temp_audio_path)),
                Message(role="user", content=test_text)
            ]
            
            # Generate audio with custom parameters
            min_p_value = min_p if min_p > 0 else None
            output = optimized_generate_audio(
                messages, max_new_tokens, temperature, top_k, top_p, min_p_value, 
                repetition_penalty, ras_win_len, ras_win_max_num_repeat, do_sample, use_cache=False
            )
            
            # Save test output to a clearly marked test location
            test_output_path = get_output_path("voice_cloning", f"voice_test_{unique_id}")
            torchaudio.save(test_output_path, torch.from_numpy(output.audio)[None, :], output.sampling_rate)
            
            return test_output_path, "✅ Voice test completed successfully!"
            
        except Exception as e:
            return None, f"❌ Error testing voice: {str(e)}"
        
        finally:
            # Clean up temporary files
            for path in [temp_audio_path, temp_txt_path]:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except Exception as e:
                        print(f"Warning: Could not clean up temp file {path}: {e}")
    
    def handle_save_voice_with_config(audio_data, voice_name, description, temperature, max_new_tokens, seed,
                                     top_k, top_p, min_p, repetition_penalty, ras_win_len, 
                                     ras_win_max_num_repeat, do_sample):
        """Save voice to library with custom configuration"""
        if audio_data is None:
            return "❌ Please upload an audio sample first"
        
        if not voice_name or not voice_name.strip():
            return "❌ Please enter a voice name"
        
        try:
            # Load audio file and convert to array format
            import torchaudio
            waveform, sample_rate = torchaudio.load(audio_data)
            audio_array = waveform.numpy()[0]  # Convert to numpy array
            
            # Save the audio file first
            status = save_voice_to_library(audio_array, sample_rate, voice_name.strip())
            
            # If audio saved successfully, save custom config
            if status.startswith("✅"):
                config = {
                    "temperature": temperature,
                    "max_new_tokens": max_new_tokens,
                    "seed": seed,
                    "top_k": top_k,
                    "top_p": top_p,
                    "min_p": min_p,
                    "repetition_penalty": repetition_penalty,
                    "ras_win_len": ras_win_len,
                    "ras_win_max_num_repeat": ras_win_max_num_repeat,
                    "do_sample": do_sample,
                    "description": description.strip() if description else "",
                    "tags": []
                }
                
                if save_voice_config(voice_name.strip(), config):
                    return f"✅ Voice '{voice_name}' saved with custom parameters!"
                else:
                    return f"⚠️ Voice saved but failed to save parameters"
            else:
                return status
                
        except Exception as e:
            return f"❌ Error saving voice: {str(e)}"
    
    def handle_voice_selection(voice_name):
        """Handle voice selection and load its configuration"""
        if not voice_name or voice_name == "None":
            return ("*Select a voice to view details*", 
                   0.3, 1024, 12345, 50, 0.95, 0.0, 1.0, 7, 2, True, "")
        
        # Load voice configuration
        config = load_voice_config(voice_name)
        
        # Get voice info
        voice_path = os.path.join("voice_library", f"{voice_name}.wav")
        txt_path = os.path.join("voice_library", f"{voice_name}.txt")
        
        info_text = f"## 🎤 {voice_name}\n\n"
        
        # Add description if available
        if config.get("description"):
            info_text += f"**Description:** {config['description']}\n\n"
        
        # Add transcript preview
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    transcript = f.read().strip()
                    if len(transcript) > 200:
                        transcript = transcript[:200] + "..."
                    info_text += f"**Sample Text:** *{transcript}*\n\n"
            except:
                pass
        
        # Add parameter summary
        info_text += "**Current Parameters:**\n"
        info_text += f"- Temperature: {config['temperature']}\n"
        info_text += f"- Max Tokens: {config['max_new_tokens']}\n"
        info_text += f"- Top-K: {config['top_k']}, Top-P: {config['top_p']}\n"
        info_text += f"- RAS Window: {config['ras_win_len']}\n"
        
        return (info_text,
                config['temperature'], config['max_new_tokens'], config['seed'],
                config['top_k'], config['top_p'], config['min_p'], config['repetition_penalty'],
                config['ras_win_len'], config['ras_win_max_num_repeat'], config['do_sample'],
                config.get('description', ''))
    
    def handle_save_voice_changes(voice_name, description, temperature, max_new_tokens, seed,
                                 top_k, top_p, min_p, repetition_penalty, ras_win_len, 
                                 ras_win_max_num_repeat, do_sample):
        """Save changes to an existing voice's configuration"""
        if not voice_name or voice_name == "None":
            return "❌ Please select a voice first"
        
        try:
            config = {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "seed": seed,
                "top_k": top_k,
                "top_p": top_p,
                "min_p": min_p,
                "repetition_penalty": repetition_penalty,
                "ras_win_len": ras_win_len,
                "ras_win_max_num_repeat": ras_win_max_num_repeat,
                "do_sample": do_sample,
                "description": description.strip() if description else "",
                "tags": []
            }
            
            if save_voice_config(voice_name, config):
                return f"✅ Updated parameters for '{voice_name}'"
            else:
                return f"❌ Failed to save changes for '{voice_name}'"
                
        except Exception as e:
            return f"❌ Error saving changes: {str(e)}"
    
    def handle_delete_voice(voice_name):
        """Delete a voice and its associated files"""
        if not voice_name or voice_name == "None":
            return "❌ Please select a voice first"
        
        try:
            voice_path = os.path.join("voice_library", f"{voice_name}.wav")
            txt_path = os.path.join("voice_library", f"{voice_name}.txt")
            config_path = get_voice_config_path(voice_name)
            
            files_deleted = []
            
            # Delete audio file
            if os.path.exists(voice_path):
                os.remove(voice_path)
                files_deleted.append("audio")
            
            # Delete transcript file
            if os.path.exists(txt_path):
                os.remove(txt_path)
                files_deleted.append("transcript")
            
            # Delete config file
            if os.path.exists(config_path):
                os.remove(config_path)
                files_deleted.append("config")
            
            if files_deleted:
                return f"✅ Deleted '{voice_name}' ({', '.join(files_deleted)})"
            else:
                return f"❌ Voice '{voice_name}' not found"
                
        except Exception as e:
            return f"❌ Error deleting voice: {str(e)}"
    
    def handle_clear_test():
        return None, "Test cleared. Upload voice and try again."
    
    def process_multi_speaker_text(text):
        """Process multi-speaker text and create dynamic interface"""
        if not text or not text.strip():
            return (
                "*No text provided*",
                [],
                {},
                gr.update(interactive=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                "*No text to process*"
            )
        
        # Detect speakers
        speakers = detect_dynamic_speakers(text)
        
        if not speakers:
            return (
                "*❌ No speakers detected. Use format: [Speaker Name] dialogue*",
                [],
                {},
                gr.update(interactive=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                "*No speakers detected*"
            )
        
        # Create speaker mapping (Speaker Name -> SPEAKER0, SPEAKER1, etc.)
        speaker_mapping = {}
        for i, speaker in enumerate(speakers):
            speaker_mapping[speaker] = f"SPEAKER{i}"
        
        # Create info display
        info_text = f"**🎭 Detected {len(speakers)} speakers:**\n\n"
        for i, speaker in enumerate(speakers):
            info_text += f"• **{speaker}** → SPEAKER{i}\n"
        
        info_text += f"\n*Now select voice assignment method below and assign voices to each speaker*"
        
        return (
            info_text,
            speakers,
            speaker_mapping,
            gr.update(interactive=True),
            gr.update(visible=False),  # upload group
            gr.update(visible=False),  # library group  
            gr.update(visible=True),   # smart voice group
            gr.update(visible=True),   # speaker assignment
            create_speaker_assignment_interface(speakers, "Smart Voice")  # assignment content
        )
    
    def update_voice_method_visibility(voice_method, detected_speakers):
        """Update visibility of voice assignment sections based on method"""
        # Get updates for upload and library slots
        upload_updates = update_upload_slots(detected_speakers if voice_method == "Upload Voices" else [])
        library_updates = update_library_slots(detected_speakers if voice_method == "Voice Library" else [])
        
        # Base outputs
        outputs = [
            gr.update(visible=voice_method == "Smart Voice"),    # smart voice group
            gr.update(visible=voice_method == "Upload Voices"),  # upload group
            gr.update(visible=voice_method == "Voice Library"),  # library group
            create_speaker_assignment_interface(detected_speakers, voice_method),  # assignment content
        ]
        
        # Add upload slot updates (10 row visibility + 10 audio components)
        outputs.extend(upload_updates)
        
        # Add library slot updates (10 dropdown components)
        outputs.extend(library_updates)
        
        # Add refresh button
        outputs.append(gr.update(visible=(voice_method == "Voice Library" and len(detected_speakers) > 0)))
        
        return outputs
    
    def update_upload_slots(speakers):
        """Update upload slot visibility and labels based on detected speakers"""
        updates = []
        
        # Update row visibility (10 rows)
        for i in range(10):
            if i < len(speakers):
                updates.append(gr.update(visible=True))  # Row visible
            else:
                updates.append(gr.update(visible=False))  # Row hidden
        
        # Update audio component labels (10 audio components)
        for i in range(10):
            if i < len(speakers):
                speaker_name = speakers[i]
                updates.append(gr.update(label=f"{speaker_name} Voice Sample"))
            else:
                updates.append(gr.update(label=f"Speaker {i} Voice Sample"))
        
        return updates
    
    def update_library_slots(speakers):
        """Update library dropdown visibility and labels based on detected speakers"""
        updates = []
        
        # Update dropdown visibility and labels (10 dropdowns)
        for i in range(10):
            if i < len(speakers):
                speaker_name = speakers[i]
                updates.append(gr.update(
                    visible=True,
                    label=f"{speaker_name} Voice",
                    value="None (Smart Voice)"
                ))
            else:
                updates.append(gr.update(visible=False))
        
        return updates
    
    def create_speaker_assignment_interface(speakers, voice_method):
        """Create dynamic speaker assignment interface based on voice method"""
        if not speakers:
            return gr.update()
        
        if voice_method == "Smart Voice":
            # For Smart Voice, show info that AI will assign automatically
            content = ""
            for speaker in speakers:
                content += f"**{speaker}**: AI will automatically assign a distinct voice\n\n"
            content += "*No manual assignment needed - the AI will ensure each speaker has a unique voice*"
            return gr.update(value=content)
        
        elif voice_method == "Voice Library":
            # For Voice Library, show voice options for each speaker
            available_voices = get_current_available_voices()
            content = "**🎭 Voice Library Assignment:**\n\n"
            content += "*Use the dropdowns below to assign specific voices from your library to each character*\n\n"
            
            for i, speaker in enumerate(speakers):
                content += f"**{speaker}** → Use dropdown to select voice\n"
            
            content += f"\n**Available voices:** {', '.join(available_voices)}\n\n"
            content += "*💡 Select 'None (Smart Voice)' to let AI pick a voice for that character*\n"
            content += "*💡 Click 'Refresh Voice Library' if you've added new voices*"
            return gr.update(value=content)
        
        elif voice_method == "Upload Voices":
            # For Upload Voices, show upload areas for each speaker  
            content = "**Upload Voice Samples:**\n\n"
            for speaker in speakers:
                content += f"**{speaker}**: Upload a voice sample to clone this speaker's voice\n\n"
            content += "*💡 Upload functionality coming soon! For now, use Smart Voice mode.*"
            return gr.update(value=content)
        
        return gr.update()
    

    
    def generate_dynamic_multi_speaker(text, voice_method, speaker_mapping, temperature, max_new_tokens, seed, scene_description, 
                                     enable_normalization, normalization_method, target_volume, speaker_pause_duration, *voice_components):
        """Generate multi-speaker audio with dynamic speaker detection and volume normalization"""
        if not text or not speaker_mapping:
            return None
        
        try:
            # Initialize model
            initialize_model()
            
            # Set seed for reproducibility
            if seed > 0:
                torch.manual_seed(seed)
                np.random.seed(seed)
            
            # Convert text to SPEAKER format
            converted_text = convert_to_speaker_format(text, speaker_mapping)
            print(f"🎭 Converted text: {converted_text[:200]}...")
            
            # Generate the audio based on voice method
            output_audio_path = None
            
            if voice_method == "Smart Voice":
                # Use the existing generate_multi_speaker function with Smart Voice
                output_audio_path = generate_multi_speaker(
                    converted_text, "Smart Voice", [], [],
                    temperature, max_new_tokens, seed, scene_description, False, speaker_pause_duration
                )
            elif voice_method == "Upload Voices":
                # voice_components structure: [upload_audio_0, upload_audio_1, ..., upload_audio_9, library_dropdown_0, ..., library_dropdown_9]
                # Extract the first 10 components (upload audio components)
                uploaded_voices = []
                num_speakers = len(speaker_mapping)
                
                for i in range(min(num_speakers, 10)):
                    if i < len(voice_components) and voice_components[i] is not None:
                        uploaded_voices.append(voice_components[i])
                    else:
                        uploaded_voices.append(None)
                
                print(f"🎭 Using uploaded voices for {num_speakers} speakers")
                
                # Use the existing generate_multi_speaker function with Upload Voices
                output_audio_path = generate_multi_speaker(
                    converted_text, "Upload Voices", uploaded_voices, [],
                    temperature, max_new_tokens, seed, scene_description, False, speaker_pause_duration
                )
            elif voice_method == "Voice Library":
                # voice_components structure: [upload_audio_0, ..., upload_audio_9, library_dropdown_0, ..., library_dropdown_9]
                # Extract the last 10 components (library dropdown selections)
                library_selections = voice_components[10:20] if len(voice_components) >= 20 else voice_components[-10:]
                predefined_voices = []
                num_speakers = len(speaker_mapping)
                
                for i in range(max(3, num_speakers)):  # Ensure at least 3 for compatibility
                    if i < len(library_selections) and library_selections[i] and library_selections[i] != "None (Smart Voice)":
                        predefined_voices.append(library_selections[i])
                    else:
                        predefined_voices.append("None (Smart Voice)")
                
                print(f"🎭 Using library voices: {predefined_voices[:num_speakers]}")
                
                # Use the existing generate_multi_speaker function with Predefined Voices
                output_audio_path = generate_multi_speaker(
                    converted_text, "Predefined Voices", [], predefined_voices,
                    temperature, max_new_tokens, seed, scene_description, False, speaker_pause_duration
                )
            
            # Apply volume normalization if enabled and audio was generated
            if output_audio_path and enable_normalization:
                print(f"🔊 Applying {normalization_method} volume normalization...")
                
                # Load the generated audio
                audio_data, sample_rate = torchaudio.load(output_audio_path)
                
                # Apply normalization using our custom module
                normalized_audio = enhance_multi_speaker_audio(
                    audio_data.squeeze(),  # Remove batch dimension if present
                    sample_rate=sample_rate,
                    normalization_method=normalization_method,
                    target_rms=target_volume
                )
                
                # Create normalized output filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                normalized_filename = f"{timestamp}_normalized_multi_speaker_audio.wav"
                normalized_path = os.path.join("output", "multi_speaker", normalized_filename)
                
                # Ensure output directory exists
                os.makedirs(os.path.dirname(normalized_path), exist_ok=True)
                
                # Save normalized audio
                if isinstance(normalized_audio, torch.Tensor):
                    # Ensure audio is 2D (channels, samples)
                    if len(normalized_audio.shape) == 1:
                        normalized_audio = normalized_audio.unsqueeze(0)
                    torchaudio.save(normalized_path, normalized_audio, sample_rate)
                else:
                    # Convert numpy to tensor
                    normalized_tensor = torch.tensor(normalized_audio, dtype=torch.float32)
                    if len(normalized_tensor.shape) == 1:
                        normalized_tensor = normalized_tensor.unsqueeze(0)
                    torchaudio.save(normalized_path, normalized_tensor, sample_rate)
                
                print(f"🎵 Saved normalized audio to: {normalized_path}")
                return normalized_path
            
            return output_audio_path
                
        except Exception as e:
            print(f"❌ Error generating dynamic multi-speaker audio: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def auto_populate_voice_name(audio_data):
        """Automatically populate voice name from uploaded audio filename"""
        if audio_data is None:
            return ""
        
        try:
            # Check if this is a file path (when audio is uploaded as file)
            if isinstance(audio_data, str) and audio_data.strip():
                # Extract filename without extension
                import os
                filename = os.path.basename(audio_data)
                name_without_ext = os.path.splitext(filename)[0]
                
                # Clean up the name (remove special characters, keep alphanumeric, hyphens, underscores)
                clean_name = "".join(c for c in name_without_ext if c.isalnum() or c in "-_ ").strip()
                # Replace spaces with underscores and remove multiple consecutive underscores
                clean_name = "_".join(clean_name.split())
                clean_name = "_".join(filter(None, clean_name.split("_")))  # Remove empty parts
                
                # Limit length and ensure it's not empty
                if clean_name and len(clean_name) > 0:
                    return clean_name[:50]
                else:
                    return "uploaded_voice"
            
            # For other cases (microphone recordings), generate timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"voice_{timestamp}"
            
        except Exception as e:
            # Fallback if anything goes wrong
            print(f"Error auto-populating voice name: {e}")
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"voice_{timestamp}"
    
    def handle_refresh_library():
        new_choices = ["None"] + get_voice_library_voices()
        return gr.update(choices=new_choices, value="None")

    # Event handling for Basic Generation
    basic_generate_btn.click(
        fn=generate_basic,
        inputs=[basic_transcript, basic_voice_prompt, basic_temperature, basic_max_new_tokens, basic_seed, basic_scene_description, 
               basic_top_k, basic_top_p, basic_min_p, basic_repetition_penalty, basic_ras_win_len, basic_ras_win_max_num_repeat, basic_do_sample,
               basic_enable_normalization, basic_target_volume],
        outputs=basic_output_audio
    )
    
    basic_refresh_voices.click(
        fn=refresh_voice_list,
        outputs=basic_voice_prompt
    )
    
    # Event handling for Voice Cloning with method selection
    vc_generate_btn.click(
        fn=handle_voice_clone_generation,
        inputs=[vc_transcript, vc_uploaded_voice, vc_temperature, vc_max_new_tokens, vc_seed, vc_method,
               vc_top_k, vc_top_p, vc_min_p, vc_repetition_penalty, vc_ras_win_len, vc_ras_win_max_num_repeat, vc_do_sample],
        outputs=vc_output_audio
    )
    
    # Event handling for Long-form Generation
    lf_generate_btn.click(
        fn=generate_longform,
        inputs=[lf_transcript, lf_voice_choice, lf_uploaded_voice, lf_voice_prompt, lf_temperature, lf_max_new_tokens, lf_seed, lf_scene_description, lf_chunk_size],
        outputs=lf_output_audio
    )
    
    lf_refresh_voices.click(
        fn=refresh_voice_list,
        outputs=lf_voice_prompt
    )
    
    # Event handling for Dynamic Multi-Speaker Generation
    ms_process_btn.click(
        fn=process_multi_speaker_text,
        inputs=[ms_transcript],
        outputs=[
            ms_speaker_info, ms_detected_speakers, ms_speaker_mapping, ms_generate_btn,
            ms_upload_group, ms_library_group, ms_smart_voice_group, ms_speaker_assignment, ms_assignment_content
        ]
    )
    
    # Prepare outputs for voice method change handler
    voice_method_outputs = [
        ms_smart_voice_group, ms_upload_group, ms_library_group, ms_assignment_content
    ]
    
    # Add upload slot components (10 rows + 10 audio components)
    for upload_row, upload_audio in ms_upload_slots:
        voice_method_outputs.append(upload_row)  # Row visibility
    for upload_row, upload_audio in ms_upload_slots:
        voice_method_outputs.append(upload_audio)  # Audio component updates
    
    # Add library slot components (10 dropdowns)
    voice_method_outputs.extend(ms_library_slots)
    
    # Add refresh button
    voice_method_outputs.append(ms_refresh_library_voices)
    
    ms_voice_method.change(
        fn=update_voice_method_visibility,
        inputs=[ms_voice_method, ms_detected_speakers],
        outputs=voice_method_outputs
    )
    
    # Prepare inputs for generation button (includes all voice components)
    generation_inputs = [
        ms_transcript, ms_voice_method, ms_speaker_mapping, ms_temperature, ms_max_new_tokens, ms_seed, ms_scene_description,
        ms_enable_normalization, ms_normalization_method, ms_target_volume, ms_speaker_pause
    ]
    
    # Add all upload audio components
    for upload_row, upload_audio in ms_upload_slots:
        generation_inputs.append(upload_audio)
    
    # Add all library dropdown components  
    generation_inputs.extend(ms_library_slots)
    
    ms_generate_btn.click(
        fn=generate_dynamic_multi_speaker,
        inputs=generation_inputs,
        outputs=[ms_output_audio]
    )
    
    # Refresh voice library dropdown choices
    def refresh_multi_speaker_voices():
        """Refresh voice choices for multi-speaker dropdowns"""
        choices = get_current_available_voices()
        return [gr.update(choices=choices) for _ in range(10)]
    
    ms_refresh_library_voices.click(
        fn=refresh_multi_speaker_voices,
        outputs=ms_library_slots
    )
    
    # Event handling for Voice Library
    # Auto-populate voice name when audio is uploaded
    vl_new_voice_audio.upload(
        fn=auto_populate_voice_name,
        inputs=[vl_new_voice_audio],
        outputs=[vl_new_voice_name]
    )
    
    vl_test_btn.click(
        fn=handle_test_voice_with_params,
        inputs=[vl_new_voice_audio, vl_test_text, vl_temperature, vl_max_new_tokens, vl_seed,
               vl_top_k, vl_top_p, vl_min_p, vl_repetition_penalty, vl_ras_win_len, 
               vl_ras_win_max_num_repeat, vl_do_sample],
        outputs=[vl_test_audio, vl_test_status]
    )
    
    vl_clear_test_btn.click(
        fn=handle_clear_test,
        outputs=[vl_test_audio, vl_test_status]
    )
    
    vl_save_btn.click(
        fn=handle_save_voice_with_config,
        inputs=[vl_new_voice_audio, vl_new_voice_name, vl_voice_description, vl_temperature, vl_max_new_tokens, vl_seed,
               vl_top_k, vl_top_p, vl_min_p, vl_repetition_penalty, vl_ras_win_len, 
               vl_ras_win_max_num_repeat, vl_do_sample],
        outputs=[vl_save_status]
    )
    
    # Voice selector change handler
    vl_voice_selector.change(
        fn=handle_voice_selection,
        inputs=[vl_voice_selector],
        outputs=[vl_voice_info, vl_edit_temperature, vl_edit_max_new_tokens, vl_edit_seed,
                vl_edit_top_k, vl_edit_top_p, vl_edit_min_p, vl_edit_repetition_penalty,
                vl_edit_ras_win_len, vl_edit_ras_win_max_num_repeat, vl_edit_do_sample, vl_edit_description]
    )
    
    # Save changes to existing voice
    vl_save_changes_btn.click(
        fn=handle_save_voice_changes,
        inputs=[vl_voice_selector, vl_edit_description, vl_edit_temperature, vl_edit_max_new_tokens, vl_edit_seed,
               vl_edit_top_k, vl_edit_top_p, vl_edit_min_p, vl_edit_repetition_penalty,
               vl_edit_ras_win_len, vl_edit_ras_win_max_num_repeat, vl_edit_do_sample],
        outputs=[vl_delete_status]
    )
    
    vl_delete_btn.click(
        fn=handle_delete_voice,
        inputs=[vl_voice_selector],
        outputs=[vl_delete_status]
    )
    
    vl_refresh_btn.click(
        fn=handle_refresh_library,
        outputs=[vl_voice_selector]
    )

    # --- Place the GitHub link at the bottom of the app ---
    gr.HTML("""
    <div style='width:100%;text-align:center;margin-top:2em;margin-bottom:1em;'>
        <a href='https://github.com/Saganaki22/higgs-audio-WebUI' target='_blank' style='color:#fff;font-size:1.1em;text-decoration:underline;'>Github</a>
    </div>
    """)

# Launch the app
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Higgs Audio v2 Generator WebUI")
    parser.add_argument("--share", action="store_true", help="Create a public shareable link via Hugging Face")
    parser.add_argument("--server-name", type=str, default="127.0.0.1", help="Server host address")
    parser.add_argument("--server-port", type=int, default=7860, help="Server port")
    args = parser.parse_args()
    
    print("🚀 Starting Higgs Audio v2 Generator...")
    print("✨ Features: Voice Cloning, Multi-Speaker, Caching, Auto-Transcription, Enhanced Audio Processing")
    
    if args.share:
        print("🌐 Creating public shareable link via Hugging Face...")
        print("⚠️  Warning: Your interface will be publicly accessible to anyone with the link!")
    
    demo.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
        show_error=True
    )

