import os
import time
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from pipeline import VoiceTranslationPipeline, Language

# --- Page Config ---
st.set_page_config(
    page_title="Let's Talk - Voice Agent",
    page_icon="🎙️",
    layout="wide",
)

# --- Initialize Session State ---
if "pipeline" not in st.session_state:
    with st.spinner("Initializing Voice Translation Pipeline... (this may take a minute)"):
        st.session_state.pipeline = VoiceTranslationPipeline()
    st.success("Pipeline Initialized!")

if "processing" not in st.session_state:
    st.session_state.processing = False

# --- UI Layout ---
st.title("🎙️ Let's Talk: Real-Time Voice Agent")
st.markdown("""
Welcome to the Let's Talk Voice Agent! 
Speak into your microphone, and the agent will translate and generate a voiced response.
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Input Settings")
    
    src_lang = st.selectbox(
        "Source Language",
        options=[l.value for l in Language],
        index=0,
        help="The language you will be speaking."
    )
    
    tgt_lang = st.selectbox(
        "Target Language",
        options=[l.value for l in Language],
        index=1,
        help="The language the agent will translate into."
    )
    
    st.write("---")
    st.subheader("Record Audio")
    
    # Instructions to make stop condition clear
    st.markdown("""
    **How to use:**
    1. Click the **Blue Pulse** to start recording.
    2. Click it **again (Red Pulse)** to stop manually.
    3. *Or speak and stop; it will auto-stop after a short silence.*
    """)
    
    audio_bytes = audio_recorder(
        text="Click to Start/Stop",
        recording_color="#e74c3c",
        neutral_color="#3498db",
    )
    
    if audio_bytes:
        st.success("✅ Recording Captured!")
        st.audio(audio_bytes, format="audio/wav")
        st.info("Click 'Process' below to translate and generate a voiced response.")
    else:
        st.info("🎤 Ready to record...")

with col2:
    st.header("Agent Output")
    
    if audio_bytes:
        # Save audio to inputs/
        input_dir = os.path.join(os.getcwd(), "inputs")
        os.makedirs(input_dir, exist_ok=True)
        input_path = os.path.join(input_dir, "recorded_audio.wav")
        
        with open(input_path, "wb") as f:
            f.write(audio_bytes)
        
        st.info(f"Audio recorded and saved to `{input_path}`")
        
        if st.button("Process Voice Agent"):
            st.session_state.processing = True
            
            try:
                with st.status("Processing...", expanded=True) as status:
                    # 1. Transcribe
                    st.write("Transcribing...")
                    original_text = st.session_state.pipeline.transcribe_audio(input_path, src_lang)
                    st.write(f"**Original ({src_lang}):** {original_text}")
                    
                    # 2. Prepare voice clone (optional)
                    st.write("Preparing voice clone...")
                    outputs_dir = os.path.join(os.getcwd(), "outputs")
                    os.makedirs(outputs_dir, exist_ok=True)
                    ref_audio = st.session_state.pipeline.prepare_voice_clone(input_path, outputs_dir)
                    
                    # 3. Translate
                    st.write("Translating...")
                    translated_text = st.session_state.pipeline.translate_text(
                        original_text, src_lang, tgt_lang
                    )
                    st.write(f"**Translated ({tgt_lang}):** {translated_text}")
                    
                    # 4. TTS
                    st.write("Generating Speech...")
                    # We pass the output directory to text_to_speech
                    import asyncio
                    asyncio.run(st.session_state.pipeline.text_to_speech(
                        text=translated_text,
                        tgt_lang=tgt_lang,
                        output_path=outputs_dir,
                        ref_audio=ref_audio
                    ))
                    
                    status.update(label="Processing Complete!", state="complete", expanded=False)

                st.session_state.processing = False
                
                # Show results
                st.subheader("Results")
                st.success(f"**Transcription:** {original_text}")
                st.success(f"**Translation:** {translated_text}")
                
                # Play audio
                # Note: mlx-audio generate_audio saves files in the output_path. 
                # We need to find the specific file generated.
                # Usually it's in the directory we provided.
                output_files = [f for f in os.listdir(outputs_dir) if f.endswith(".wav") and f != "voice_ref.wav"]
                if output_files:
                    # Sort by mtime to get the latest
                    output_files.sort(key=lambda x: os.path.getmtime(os.path.join(outputs_dir, x)), reverse=True)
                    latest_audio = os.path.join(outputs_dir, output_files[0])
                    st.audio(latest_audio)
                else:
                    st.warning("No output audio file found in outputs/")

            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
                st.session_state.processing = False
    else:
        st.info("Waiting for audio input...")

# --- Footer ---
st.write("---")
if st.button("Clear Memory & Refresh"):
    if "pipeline" in st.session_state:
        st.session_state.pipeline.clear_memory()
        del st.session_state.pipeline
    st.rerun()
