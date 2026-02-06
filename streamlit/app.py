import streamlit as st
import os
import re
import torch

from huggingface_hub import login
from pyannote.audio import Pipeline
from pydub import AudioSegment
import whisper
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Meeting Minutes Generator",
    layout="wide"
)

st.title("🧠 AI Meeting Minutes Generator")
st.caption("Audio → Diarization → Transcription → Summary → Bullet Points → Action Items")

# ---------------- HUGGING FACE LOGIN ----------------
# 🔴 Replace with your token
login("REMOVED_TOKEN")

# ---------------- AUDIO UPLOAD ----------------
uploaded_audio = st.file_uploader(
    "Upload meeting audio (.wav)",
    type=["wav"]
)

if uploaded_audio is not None:

    # ✅ GENERALIZED AUDIO HANDLING
    AUDIO_FILE = "uploaded_audio.wav"

    with open(AUDIO_FILE, "wb") as f:
        f.write(uploaded_audio.read())

    st.success(f"Audio uploaded successfully")

    # ---------------- SPEAKER DIARIZATION ----------------
    st.subheader("🔊 Speaker Diarization")

    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=True
    )

    diarization = diarization_pipeline(AUDIO_FILE)

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        st.write(f"{speaker}: {turn.start:.2f}s → {turn.end:.2f}s")

    # ---------------- SPLIT AUDIO BY SPEAKER ----------------
    audio = AudioSegment.from_wav(AUDIO_FILE)
    os.makedirs("speaker_chunks", exist_ok=True)

    chunks = []
    for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
        start_ms = int(turn.start * 1000)
        end_ms = int(turn.end * 1000)
        chunk = audio[start_ms:end_ms]
        fname = f"speaker_chunks/{speaker}_{i}.wav"
        chunk.export(fname, format="wav")
        chunks.append((speaker, fname))

    # ---------------- SPEECH TO TEXT (WHISPER) ----------------
    st.subheader("📝 Speaker-wise Transcription")

    whisper_model = whisper.load_model("base")
    speaker_transcripts = []

    for speaker, path in chunks:
        result = whisper_model.transcribe(path)
        speaker_transcripts.append({
            "speaker": speaker,
            "text": result["text"].strip()
        })
        st.write(f"**{speaker}**: {result['text']}")

    # ---------------- MERGE TRANSCRIPT ----------------
    full_meeting_text = ""
    for item in speaker_transcripts:
        full_meeting_text += f"{item['speaker']}: {item['text']}\n"

    clean_text = re.sub(r"\s+", " ", full_meeting_text).strip()

    # ---------------- SUMMARIZATION (BART) ----------------
    st.subheader("📄 Meeting Summary")

    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn"
    )

    summary = summarizer(
        clean_text,
        max_length=80,
        min_length=40,
        do_sample=False
    )

    summary_text = summary[0]["summary_text"]
    st.success(summary_text)

    # ---------------- LOAD MISTRAL AGENT ----------------
    st.subheader("🤖 Bullet Points & Action Items")

    model_id = "mistralai/Mistral-7B-Instruct-v0.2"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    # =====================================================
    # 🔒 AGENT PROMPT (UNCHANGED)
    # =====================================================
    agent_prompt = f"""
You are an AI meeting assistant designed for structured meeting analysis.

You must perform TWO SEPARATE TASKS using DIFFERENT INPUTS.

=================================================
TASK 1: BULLET POINT GENERATION (FROM SUMMARY)
=================================================
Generate clear bullet points from the MEETING SUMMARY.

RULES FOR BULLET POINTS:
- Capture key discussion topics and decisions.
- Do NOT include implementation details.
- Do NOT include deadlines or task ownership.
- Each bullet point must be one concise sentence.
- Maximum 5 bullet points.

Meeting Summary:
{summary_text}

=================================================
TASK 2: ACTION ITEM EXTRACTION (FROM FULL TEXT)
=================================================
Extract ONLY explicit action items from the FULL MEETING TRANSCRIPT.

DEFINITION RULES:
1. A TASK is a clearly assigned activity.
2. An OWNER is the person explicitly named as responsible.
   - Use the exact name as written.
   - Do NOT infer from speaker labels.
3. A DEADLINE can be:
   - A date (e.g., 12 June)
   - A day name (e.g., Friday)
   - A relative time (e.g., next week)

STRICT RULES:
- Extract ONLY explicit action items (not discussions).
- Do NOT guess or infer missing details.
- If OWNER is missing, write "Not mentioned".
- If DEADLINE is missing, write "Not mentioned".
- Number each action item starting from 1.
- Use EXACT format:
  1. Task | Owner | Deadline

Full Meeting Transcript:
{full_meeting_text}

=================================================
FINAL OUTPUT FORMAT (DO NOT CHANGE)
=================================================

Bullet Points:
- Bullet point 1
- Bullet point 2
- Bullet point 3

Action Items:
1. Task | Owner | Deadline
2. Task | Owner | Deadline
"""

    # ---------------- TOKENIZATION ----------------
    inputs = tokenizer(
        agent_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
    )

    input_tokens = len(tokenizer.tokenize(agent_prompt))

    # ---------------- GENERATION ----------------
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=False
    )

    # ---------------- DECODE ----------------
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    final_output = full_output.replace(agent_prompt, "").strip()

    output_tokens = len(tokenizer.tokenize(final_output))

    st.text(final_output)

    # ---------------- TOKEN METRICS ----------------
    st.subheader("📊 Token Metrics")
    st.write(f"Input Tokens: {input_tokens}")
    st.write(f"Output Tokens: {output_tokens}")
