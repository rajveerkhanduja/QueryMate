import warnings
warnings.filterwarnings('ignore', message='SymbolDatabase.GetPrototype() is deprecated')

import cv2
import streamlit as st
import mediapipe as mp
import numpy as np
import time
from PyPDF2 import PdfReader
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pandas as pd
import altair as alt

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY environment variable not set")

genai.configure(api_key=api_key)

# Function to get text from PDF
def get_pdf_text(pdf):
    text = ""
    try:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    except Exception as e:
        st.error(f"Error reading {pdf.name}: {e}")
    return text

# Function to generate questions from resume text
def generate_questions(resume_text):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    prompt = (
        f"Generate technical interview questions based on the following resume text. "
        f"Make the questions challenging and logically sound. "
        f"Do not include topic names or question numbers or bullet point the questions in your response give only 10 shuffled questions related to the given text:\n\n{resume_text}\n\nQuestions:"
    )
    response = model.predict(prompt)
    questions = response.split('\n')
    filtered_questions = [
        q for q in questions if q.strip() and not q.startswith('Question ') and not q.endswith(':')
    ]
    filtered_questions = [
        q for q in filtered_questions if not q.lower().startswith("question ")
    ]
    return filtered_questions

# Function to evaluate answers
def evaluate_answers(questions_and_answers):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    score = 0
    total = len(questions_and_answers)
    results = []

    for q, a in questions_and_answers:
        if not a.strip():
            correct = "No"
        else:
            prompt = (
                f"Question: {q}\n"
                f"Answer provided: {a}\n"
                f"Is the provided answer correct, partially correct, or incorrect? Respond with 'Yes', 'Partially Correct', or 'No'."
            )
            response = model.predict(prompt)
            if "Yes" in response:
                correct = "Yes"
                score += 10
            elif "Partially Correct" in response:
                correct = "Partially Correct"
                score += 5
            else:
                correct = "No"
        results.append((q, correct))

    percentage = (score / (total * 10)) * 100
    return percentage, results

# Streamlit main function
def main():
    st.set_page_config(page_title="QueryMate🤖", layout="wide")
    st.title("QueryMate🤖 - Your Technical Interviewer")

    # Sidebar for PDF processing
    st.sidebar.header("Upload your resume")
    pdf_doc = st.sidebar.file_uploader("Upload your Resume (PDF)", type="pdf")
    if st.sidebar.button("Submit & Process"):
        if pdf_doc:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_doc)
                if raw_text:
                    questions = generate_questions(raw_text)
                    st.session_state['questions'] = questions
                    st.session_state.page = 'questions'
                    st.experimental_rerun()
                else:
                    st.error("Failed to extract text from PDF file. Make sure your resume is formatted correctly.")
        else:
            st.error("Please upload a PDF file.")

    # Create column layout
    left_column, right_column = st.columns([3, 1])

    # Left column for question answering and results
    with left_column:
        if 'page' not in st.session_state:
            st.session_state.page = 'home'

        if st.session_state.page == 'questions':
            st.subheader("Please answer the following questions:")
            answers = []
            for i, question in enumerate(st.session_state['questions']):
                st.markdown(f"**Question** {question}")
                answer = st.text_area(f"Answer", key=f"q{i+1}")
                answers.append((question, answer))
            
            if st.button("Submit Answers"):
                st.session_state['answers'] = answers
                st.session_state.page = 'result'
                st.experimental_rerun()

        elif st.session_state.page == 'result':
            with st.spinner("Evaluating your answers, please wait..."):
                percentage, results = evaluate_answers(st.session_state['answers'])
            
            st.success("Processing complete!")
            st.balloons()  # Displays a balloon animation
            st.write(f"Your score is: {percentage:.2f}%")
            st.markdown(
                """
                <style>
                table {
                    width: 100%;
                }
                th {
                    text-align: left !important;
                }
                td {
                    text-align: left !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            # Table
            results_df = pd.DataFrame(results, columns=["Questions", "Correct"])
            st.markdown(results_df.to_html(index=False), unsafe_allow_html=True)

            # Pie chart
            correct_count = results_df['Correct'].value_counts().to_dict()
            pie_data = pd.DataFrame({
                'Category': ['Yes', 'Partially Correct', 'No'],
                'Count': [
                    correct_count.get('Yes', 0),
                    correct_count.get('Partially Correct', 0),
                    correct_count.get('No', 0)
                ]
            })
            pie_chart = alt.Chart(pie_data).mark_arc().encode(
                theta=alt.Theta(field="Count", type="quantitative"),
                color=alt.Color(field="Category", type="nominal"),
                tooltip=['Category', 'Count']
            ).properties(
                width=600,
                height=600
            )
            st.altair_chart(pie_chart)

    # Right column for video feed and warnings
    with right_column:
        video_feed = st.empty()
        warning_text = st.empty()

        # Function to draw hand landmarks
        def draw_hand_landmarks(image, hand_landmarks, mp_hands):
            for landmarks in hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image, landmarks, mp_hands.HAND_CONNECTIONS)
            return image

        # Initialize MediaPipe Face Mesh and Hands
        mp_face_mesh = mp.solutions.face_mesh
        mp_hands = mp.solutions.hands

        # Initialize the webcam
        video = cv2.VideoCapture(0)

        # Initialize variables for warnings
        warnings_count = 0
        max_warnings = 3
        warning_issued_time = 0
        warning_cooldown = 5  # cooldown time in seconds between warnings

        def show_warning(message):
            warning_text.markdown(
                f"<div class='warning'>{message}</div>",
                unsafe_allow_html=True
            )

        with mp_face_mesh.FaceMesh(max_num_faces=2, min_detection_confidence=0.5) as face_mesh, mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            prevTime = 0

            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results_face = face_mesh.process(frame_rgb)
                results_hands = hands.process(frame_rgb)

                face_count = 0
                if results_face.multi_face_landmarks:
                    face_count = len(results_face.multi_face_landmarks)
                    for face_landmarks in results_face.multi_face_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(
                            frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

                current_time = time.time()
                if face_count == 0 and current_time - warning_issued_time > warning_cooldown:
                    warnings_count += 1
                    show_warning(f"{warnings_count} - User out of screen!")
                    warning_issued_time = current_time

                elif face_count > 2 and current_time - warning_issued_time > warning_cooldown:
                    warnings_count += 1
                    show_warning(f"More than 1 face detected!")
                    warning_issued_time = current_time

                if warnings_count >= max_warnings:
                    show_warning("Application terminated due to too many warnings.")
                    break

                if results_hands.multi_hand_landmarks:
                    frame = draw_hand_landmarks(frame, results_hands.multi_hand_landmarks, mp_hands)

                currTime = time.time()
                fps = 1 / (currTime - prevTime)
                prevTime = currTime

                video_feed.image(frame, channels='RGB', use_column_width=True)

                time.sleep(0.1)  # Add a short sleep to prevent high CPU usage

        video.release()

        # Add custom CSS to style the warning message
        st.markdown(
            """
            <style>
            .warning {
                background-color: #ffcc00;
                color: #000000;
                padding: 10px;
                border-radius: 5px;
                width: 100%; /* Make this width full */
                text-align: center;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()