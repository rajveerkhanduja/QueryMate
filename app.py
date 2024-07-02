import streamlit as st
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

def get_pdf_text(pdf):
    text = ""
    try:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    except Exception as e:
        st.error(f"Error reading {pdf.name}: {e}")
    return text

def generate_questions(resume_text):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    prompt = (
        f"Generate technical interview questions based on the following resume text. "
        f"Make the questions challenging and logically sound. "
        f"Do not include topic names or question numbers or bullet point the questions in your response give only 10 shuffled questions related to the given text:\n\n{resume_text}\n\nQuestions:"
    )
    response = model.predict(prompt)
    questions = response.split('\n')  # Assuming each question is separated by a newline

    # Filter out non-question items (headings)
    filtered_questions = [
        q for q in questions if q.strip() and not q.startswith('Question ') and not q.endswith(':')
    ]
    
    # Further filter to remove any lines starting with "Question"
    filtered_questions = [
        q for q in filtered_questions if not q.lower().startswith("question ")
    ]

    return filtered_questions

def evaluate_answers(questions_and_answers):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    score = 0
    total = len(questions_and_answers)
    results = []

    for q, a in questions_and_answers:
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

def main():
    st.set_page_config(page_title="QueryMate🤖", layout="wide")
    st.title("QueryMate🤖 - Your Technical Interviewer")

    if 'page' not in st.session_state:
        st.session_state.page = 'home'

    if st.session_state.page == 'home':
        with st.sidebar:
            st.header("Menu")
            st.write("Upload your resume to generate interview questions.")
            pdf_doc = st.file_uploader("Upload your Resume (PDF)", type="pdf")
            if st.button("Submit & Process"):
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

    elif st.session_state.page == 'questions':
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
        st.write("Evaluating your answers, please wait...")
        with st.spinner("Analyzing..."):
            percentage, results = evaluate_answers(st.session_state['answers'])
        
        st.success("Processing complete!")
        st.balloons()  # Displays a balloon animation
        st.write(f"Your score is: {percentage:.2f}%")

        # Display results in a table with only Question and Correct columns
        results_df = pd.DataFrame(results, columns=["Question", "Correct"])
        st.table(results_df)  # Hide the index

        # Display results in a pie chart
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
            width=400,
            height=400
        )
        st.altair_chart(pie_chart)

if __name__ == "__main__":
    main()