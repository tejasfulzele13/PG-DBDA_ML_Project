# 🧠 AI Meeting Minutes Generator

## 📌 Project Overview
AI Meeting Minutes Generator is an end-to-end machine learning application that automatically converts meeting audio into structured and actionable meeting 
minutes. The system identifies speakers, transcribes speech, summarizes discussions, and extracts bullet points and action items using transformer-based 
models.

---

## ❓ Problem Statement
Manual meeting minute preparation is time-consuming, error-prone, and inefficient, especially for long meetings involving multiple speakers. There is a need
for an automated system that can accurately capture discussions, identify speakers, and generate structured summaries and action items.

---

## 💡 Solution Overview
This project implements an AI-driven pipeline that:
- Identifies **who spoke when** using speaker diarization
- Converts speech to text using a speech recognition model
- Summarizes meeting discussions
- Extracts bullet points and structured action items using a goal-driven agentic AI approach

The application is deployed using **Streamlit** for an interactive user interface.

---

## 🏗️ Architecture / Workflow

Audio Input
↓
Speaker Diarization (PyAnnote)
↓
Speech-to-Text (Whisper)
↓
Text Cleaning
↓
Meeting Summarization (BART)
↓
Agentic AI (Mistral)
↓
Bullet Points & Action Items




-----Audio File-----
Sample audio files are not included to keep the repository lightweight.
