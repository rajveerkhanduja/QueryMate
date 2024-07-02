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
    # Use the Gemini model to generate precise and technical questions based on the resume text
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

def get_text_chunks(text):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. If the answer is not in
    the provided context, just say, "answer is not available in the context", and don't provide a wrong answer.\n\n
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def evaluate_answers(questions_and_answers):
    # Placeholder for evaluating answers
    # In a real implementation, you'd use the conversational chain to evaluate each answer
    score = 0
    total = len(questions_and_answers)
    
    # Dummy evaluation logic
    for q, a in questions_and_answers:
        if a:  # assuming non-empty answers are correct for demo purposes
            score += 1
    
    percentage = (score / total) * 100
    return percentage

def main():
    st.set_page_config(page_title="QueryMate🤖", layout="wide")
    st.title("QueryMate🤖 - Your Technical Interviewer")

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
                        st.success("Questions generated successfully!")
                    else:
                        st.error("Failed to extract text from PDF file. Make sure your resume is formatted correctly.")
            else:
                st.error("Please upload a PDF file.")
    
    if 'questions' in st.session_state:
        st.subheader("Please answer the following questions:")
        answers = []
        for i, question in enumerate(st.session_state['questions']):
            st.markdown(f"**Question** {question}")
            answer = st.text_area(f"Answer", key=f"q{i+1}")
            answers.append((question, answer))
        
        if st.button("Submit Answers"):
            percentage = evaluate_answers(answers)
            st.write(f"Your score is: {percentage:.2f}%")

if __name__ == "__main__":
    main()