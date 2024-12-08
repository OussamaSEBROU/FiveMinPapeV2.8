import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import io
import base64
import time
import uuid
import json
from datetime import datetime
import tempfile  # Add this import

# Ensure necessary libraries are installed
#!pip install -q streamlit PyPDF2 langchain google-generativeai python-dotenv

import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
# IMPORTANT: Replace with your actual Google API key or use os.getenv()
load_dotenv()
# Use environment variables for API keys
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

# Enhanced Configuration for Chat History using a temporary directory
CHAT_HISTORY_DIR = os.path.join(tempfile.gettempdir(), "5minpaper_chat_histories")
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

# Page Configuration with Mobile Responsiveness
st.set_page_config(
    page_title="5MinPaper",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="auto"
)

# Enhanced CSS Styling with Mobile Responsiveness
st.markdown("""
<style>
    :root {
        --primary-color: #3498db;
        --secondary-color: #2c3e50;
        --background-color: #f4f6f8;
        --text-color: #2c3e50;
        --accent-color: #2ecc71;
    }
    .pdf-viewer {
        width: 100%;
        height: 100vh;
        max-height: 800px;
        border: 2px solid var(--primary-color);
        border-radius: 10px;
        overflow: auto;
        background-color: white;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
        padding: 10px;
    }
    @media (max-width: 768px) {
        .pdf-viewer {
            height: 50vh;
            padding: 10px;
        }
        .stSidebar {
            min-width: 250px !important;
            max-width: 100% !important;
        }
    }
    .help-section {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .chat-history-item {
        margin-bottom: 10px;
        padding: 10px;
        background-color: #f4f6f8;
        border-radius: 5px;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .chat-history-item:hover {
        background-color: #e0e6eb;
    }
    iframe {
        width: 100%;
        height: 100vh;
        max-height: 800px;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# Utility Functions
def pdf_to_base64(pdf_file):
    return base64.b64encode(pdf_file.getvalue()).decode('utf-8')

def is_scanned_pdf(pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        text = "".join([page.extract_text() for page in pdf_reader.pages])
        return len(text.strip()) < 100
    except Exception as e:
        st.error(f"PDF Processing Error: {e}")
        return True

def process_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = "".join([page.extract_text() for page in pdf_reader.pages])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    text_chunks = text_splitter.split_text(text)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    file_id = str(uuid.uuid4())

    faiss_temp_dir = os.path.join(tempfile.gettempdir(), "faiss_indices")
    os.makedirs(faiss_temp_dir, exist_ok=True)
    full_path = os.path.join(faiss_temp_dir, f"faiss_index_{file_id}")
    vector_store.save_local(full_path)

    #vector_store.save_local(f"/content/faiss_index_{file_id}")

    return file_id

def save_chat_history(conversation_history):
    current_time = datetime.now().strftime("%Y-%m-%d_%Hh%Mmin")
    filename = f"{current_time}_chat_history.json"
    filepath = os.path.join(CHAT_HISTORY_DIR, filename)

    with open(filepath, 'w') as f:
        json.dump(conversation_history, f, indent=4)

    return filename

def load_chat_history(filename):
    filepath = os.path.join(CHAT_HISTORY_DIR, filename)
    with open(filepath, 'r') as f:
        return json.load(f)

def list_chat_histories():
    return sorted(
        [f for f in os.listdir(CHAT_HISTORY_DIR) if f.endswith('.json')],
        reverse=True
    )

def display_pdf(pdf_file):
    base64_pdf = base64.b64encode(pdf_file.getvalue()).decode('utf-8')
    pdf_display = f'''
    <div class="pdf-viewer">
        <iframe
            src="data:application/pdf;base64,{base64_pdf}"
            width="100%"
            height="100%"
            type="application/pdf"
        >
            Your browser does not support PDF viewing.
            Please download the PDF to view it.
        </iframe>
    </div>
    '''
    st.markdown(pdf_display, unsafe_allow_html=True)

def get_conversation_context(max_context_length=3):
    if not hasattr(st.session_state, 'conversation_history'):
        return ""

    context_history = st.session_state.conversation_history[-max_context_length:]
    context_str = "\n".join([
        f"Previous Query: {item['query']}\nPrevious Response: {item['response']}"
        for item in context_history
    ])

    return context_str

def render_sidebar():
    with st.sidebar:
        st.markdown("<div style='font-size: 1.6rem; font-weight: bold; color: #00b9c6;'>Interact with your Papers</div>", unsafe_allow_html=True)

        # New Chat Button
        if st.button("New Chat", key="new_chat_btn"):
            # Reset session state, preserving initial state
            keys_to_preserve = ['uploaded_pdf', 'pdf_processed', 'pdf_file_id', 'pdf_base64']
            for key in list(st.session_state.keys()):
                if key not in keys_to_preserve:
                    del st.session_state[key]

            # Initialize empty conversation history for new chat
            st.session_state.conversation_history = []
            st.rerun()

        # Document Upload Section
        with st.expander("📤 Document Upload", expanded=True):
            pdf_docs = st.file_uploader(
                "Choose PDF",
                type="pdf",
                help="Upload a text-based PDF for analysis"
            )

            if pdf_docs:
                if st.button("Process Document", key="process_doc"):
                    with st.spinner("Processing PDF..."):
                        if is_scanned_pdf(pdf_docs):
                            st.error("Scanned or low-text PDF detected")
                        else:
                            file_id = process_pdf(pdf_docs)
                            st.session_state.pdf_processed = True
                            st.session_state.pdf_file_id = file_id
                            st.session_state.pdf_base64 = pdf_to_base64(pdf_docs)
                            st.session_state.uploaded_pdf = pdf_docs
                            st.session_state.conversation_history = []
                            st.success("PDF processed successfully!")

        # PDF Viewing Option
        if hasattr(st.session_state, 'uploaded_pdf'):
            view_pdf = st.checkbox("View PDF", key="view_pdf_checkbox")

        # Chat History Management
        with st.expander("Chat History", expanded=False):
            chat_histories = list_chat_histories()

            if chat_histories:
                selected_history = st.selectbox(
                    "Select Previous Chat",
                    [""] + chat_histories,
                    format_func=lambda x: x.replace('_chat_history.json', '') if x else "Select a chat"
                )

                col1, col2 = st.columns(2)

                with col1:
                    if selected_history and st.button("Load Chat", key="load_chat_btn"):
                        loaded_history = json.load(open(os.path.join(CHAT_HISTORY_DIR, selected_history)))
                        st.session_state.conversation_history = loaded_history
                        st.rerun()

                with col2:
                    if selected_history and st.button("Delete Chat", key="delete_chat_btn"):
                        os.remove(os.path.join(CHAT_HISTORY_DIR, selected_history))
                        st.rerun()
            else:
                st.info("No chat histories available")

        # Help Section
        with st.expander("Help & Support", expanded=False):
            st.markdown("How to Use 5MinPaper")

            help_sections = {
                "Uploading Documents": [
                    "Click on 'Choose PDF' to upload your document",
                    "Ensure the PDF is text-based for best results",
                    "Click 'Process Document' to analyze the file"
                ],
                "Asking Questions": [
                    "Type your question in the input field",
                    "Click 'Get Insights' to receive AI-powered answers",
                    "Questions can range from summarization to specific details"
                ]
            }

            for section, tips in help_sections.items():
                st.markdown(f"#### {section}")
                for tip in tips:
                    st.markdown(f"- {tip}")

            st.markdown("Troubleshooting")
            st.markdown("""
            - **No Text Extraction**: Ensure PDF is not scanned
            - **Slow Response**: Large PDFs might take longer to process
            - **Error Messages**: Check document format and content
            """)

        # About Section
        with st.expander("About Us", expanded=False):
            st.markdown("#### 5MinPaper - Intelligent Scientific Papers Analysis")
            st.markdown("This app was developed by Mr. Oussama SEBROU")
            st.markdown("""
            **Version:** 2.5.1

            #### Features
            - AI-Powered PDF Analysis
            - Multi-Language Support
            - Context-Aware Question Answering
            - Fast and Precise Insights

            5MinPaper leverages advanced AI to transform how you interact with scientific papers and documents.

            © 2024 5MinPaper Team. All rights reserved.

            [Contact Us](mailto:oussama.sebrou@gmail.com?subject=5MinPaper%20Inquiry&body=Dear%205MinPaper%20Team,%0A%0AWe%20are%20writing%20to%20inquire%20about%20[your%20inquiry]%2C%20specifically%20[details%20of%20your%20inquiry].%0A%0A[Provide%20additional%20context%20and%20details%20here].%0A%0APlease%20let%20us%20know%20if%20you%20require%20any%20further%20information%20from%20our%20end.%0A%0ASincerely,%0A[Your%20Company%20Name]%0A[Your%20Name]%0A[Your%20Title]%0A[Your%20Phone%20Number]%0A[Your%20Email%20Address])
            """)

def main():
    # Initialize conversation history as empty list if not exists
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    st.markdown("<h1 style='font-size: 3.5rem;'>5MinPaper Chat</h1>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 1.5rem; color: #00b9c6;'>Unlock the Knowledg of your Scientific Paper with Large Language Models (LLMs) AI Technology. Ask insightful questions and receive precise, context-aware answers directly from your documents.</div>", unsafe_allow_html=True)
    st.write("")

    # Render Sidebar
    render_sidebar()

    # Display PDF if View PDF is checked
    if hasattr(st.session_state, 'uploaded_pdf'):
        if st.session_state.view_pdf_checkbox:
            display_pdf(st.session_state.uploaded_pdf)

    # Chat Interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # Display Conversation History
    if st.session_state.conversation_history:
        for interaction in st.session_state.conversation_history:
            st.markdown(f"<span style='font-size: 1.5rem; color: #00b9c6;'>**Q:** {interaction['query']}</span>", unsafe_allow_html=True)
            st.markdown(f"**A:** {interaction['response']}")

    # User Query Input
    user_query = st.text_input(
        "Your Question",
        placeholder="Ask something about your document...",
        key="user_query_input"
    )

    # Insights Generation
    if st.button("Get Insights", key="insights_btn"):
        if not hasattr(st.session_state, 'pdf_processed') or not st.session_state.pdf_processed:
            st.warning("Please upload and process a PDF first")
            return

        response_placeholder = st.empty()
        with st.spinner("Generating Insights..."):
            try:
                model = ChatGoogleGenerativeAI(model="gemini-exp-1206", temperature=0.7)

                prompt_template = """
                You are an advanced document analysis and interaction AI your name is forever 5minPaper AI, Your responses are based solely on the provided text.
                Context from Previous Interactions:
                {previous_context}

                Document Context:
                {context}

                User Request:
                {question}

                Advanced Instructions:
                1. Consider previous conversation context.
                2. Analyze deeply Document.
                3. Provide Precise, Contextual Response.
                4. Maintain Conversation Coherence.
                5. If the user has not asked for a translation yet, your answer should be in the same language as the question written.
                6. Use professional mathematical and scientific notation.
                7. Mathematical and Scientific Notation: For any mathematical formulas, scientific symbols, or code snippets, present them like professional LaTeX font formatting for professional and accurate representation.
                8. Output Length: Your response should ideally be around 1000 tokens or more. 
                """

                prompt = PromptTemplate(
                    template=prompt_template,
                    input_variables=["previous_context", "context", "question"]
                )

                chain = load_qa_chain(
                    model,
                    chain_type="stuff",
                    prompt=prompt
                )

                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

                file_id = st.session_state.get('pdf_file_id', 'default')
                faiss_temp_dir = os.path.join(tempfile.gettempdir(), "faiss_indices")
                full_path = os.path.join(faiss_temp_dir, f"faiss_index_{file_id}")
                new_db = FAISS.load_local(
                  full_path,
                  embeddings,
                  allow_dangerous_deserialization=True
                )



                #new_db = FAISS.load_local(
                #    f"/content/faiss_index_{file_id}",
                #    embeddings,
                #    allow_dangerous_deserialization=True
                #)

                docs = new_db.similarity_search(user_query)

                # Get conversation context
                previous_context = get_conversation_context()

                response = chain(
                    {
                        "previous_context": previous_context,
                        "input_documents": docs,
                        "question": user_query
                    },
                    return_only_outputs=True
                )

                # Simulate typing response
                displayed_text = ""
                for char in response["output_text"]:
                    displayed_text += char
                    response_placeholder.markdown(displayed_text)
                    time.sleep(0.006)

                # Update conversation history
                st.session_state.conversation_history.append({
                    'query': user_query,
                    'response': response["output_text"]
                })

            except Exception as e:
                st.error(f"Analysis Error: {e}")

    # Save Chat Button
    if st.session_state.conversation_history:
        if st.button("💾 Save Current Chat", key="save_chat_btn"):
            saved_filename = save_chat_history(st.session_state.conversation_history)
            st.success(f"Chat saved as {saved_filename}")

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
