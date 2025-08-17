import os
import json
import pandas as pd
import traceback
# from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.callbacks import get_openai_callback
import PyPDF2
import streamlit as st
import re

# load_dotenv()
# KEY = os.getenv("GROQ_API_KEY")
KEY = st.secrets["GROQ_API_KEY"]

# Using a more reliable model
llm = ChatGroq(
    groq_api_key=KEY,
    model_name="llama3-70b-8192",  # More stable and reliable model
    temperature=0
)

RESPONSE_JSON = {
    "1": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    "2": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    "3": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
}

TEMPLATE = """
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to 
create a quiz of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like RESPONSE_JSON below and use it as a guide. 
Ensure to make {number} MCQs. Please create MCQs which are provided in JSON format.
Return ONLY the JSON response, no additional text.

### RESPONSE_JSON
{RESPONSE_JSON}
"""

quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "RESPONSE_JSON"],
    template=TEMPLATE
)
quiz_chain = LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True)

generate_evaluate_chain = SequentialChain(
    chains=[quiz_chain], 
    input_variables=["text", "number", "subject", "tone", "RESPONSE_JSON"], 
    output_variables=["quiz"], 
    verbose=True
)

def extract_pdf_text(file):
    """Extract text from PDF file with better error handling"""
    try:
        # Use PdfReader instead of deprecated PdfFileReader
        reader = PyPDF2.PdfReader(file)
        text = ""
        
        # Check if PDF has pages
        if len(reader.pages) == 0:
            raise ValueError("PDF file appears to be empty or corrupted")
        
        # Extract text from all pages
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            page_text = page.extract_text()
            if page_text.strip():  # Only add non-empty text
                text += page_text + "\n"
        
        # Check if any text was extracted
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF")
        
        return text
    except Exception as e:
        raise ValueError(f"Error extracting text from PDF: {str(e)}")

def clean_json_response(response_text):
    """Clean and extract JSON from LLM response"""
    try:
        # Try to parse as is first
        return json.loads(response_text)
    except json.JSONDecodeError:
        # If direct parsing fails, try to extract JSON from response
        # Look for JSON-like content between braces
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # If all fails, raise error
        raise ValueError("Could not parse JSON from LLM response")

def validate_mcq_json(mcq_data, expected_count):
    """Validate the structure of MCQ JSON"""
    if not isinstance(mcq_data, dict):
        raise ValueError("MCQ data must be a dictionary")
    
    if len(mcq_data) != expected_count:
        raise ValueError(f"Expected {expected_count} MCQs, got {len(mcq_data)}")
    
    for key, mcq in mcq_data.items():
        if not isinstance(mcq, dict):
            raise ValueError(f"MCQ {key} must be a dictionary")
        
        required_keys = ['mcq', 'options', 'correct']
        for req_key in required_keys:
            if req_key not in mcq:
                raise ValueError(f"MCQ {key} missing required key: {req_key}")
        
        if not isinstance(mcq['options'], dict):
            raise ValueError(f"MCQ {key} options must be a dictionary")
        
        if len(mcq['options']) < 2:
            raise ValueError(f"MCQ {key} must have at least 2 options")

# Streamlit UI
st.title("MCQ Generator with PDF Support")
st.markdown("Upload a PDF or text file and generate multiple choice questions!")

# Input fields
col1, col2 = st.columns(2)
with col1:
    subject = st.text_input("Subject", placeholder="e.g., Mathematics, History, Science")
with col2:
    number = st.number_input("Number of MCQs", min_value=1, max_value=20, value=5, step=1)

tone = st.selectbox("Difficulty Level", ["simple", "medium", "hard"])
file = st.file_uploader("Upload a PDF or Text file", type=["pdf", "txt"])

# Show file info if uploaded
if file is not None:
    st.success(f"File uploaded: {file.name} ({file.type})")

generate_button = st.button("Generate MCQs", type="primary")

if generate_button:
    if not all([subject, number, tone, file]):
        st.error("Please fill in all fields and upload a file.")
    else:
        try:
            # Extract text from file
            with st.spinner("Extracting text from file..."):
                if file.type == "application/pdf":
                    TEXT = extract_pdf_text(file)
                elif file.type == "text/plain":
                    TEXT = file.read().decode("utf-8")
                else:
                    raise ValueError("Unsupported file type")
            
            # Validate extracted text
            if len(TEXT.strip()) < 100:
                st.warning("The extracted text is quite short. This might affect the quality of generated MCQs.")
            
            # Show text preview
            with st.expander("Preview of extracted text"):
                st.text_area("Extracted Text", TEXT[:500] + "..." if len(TEXT) > 500 else TEXT, height=150)
            
            # Generate MCQs
            with st.spinner("Generating MCQs..."):
                try:
                    response = generate_evaluate_chain(
                        {
                            "text": TEXT,
                            "number": number,
                            "subject": subject,
                            "tone": tone,
                            "RESPONSE_JSON": json.dumps(RESPONSE_JSON)
                        }
                    )
                    
                    # Clean and parse the response
                    quiz_text = response['quiz']
                    quiz_data = clean_json_response(quiz_text)
                    
                    # Validate the MCQ structure
                    validate_mcq_json(quiz_data, number)
                    
                    # Display MCQs
                    st.success("MCQs generated successfully!")
                    st.subheader("Generated MCQs:")
                    
                    # Create tabs for better organization
                    tab1, tab2 = st.tabs(["📝 Questions", "📊 Summary"])
                    
                    with tab1:
                        for idx, (key, mcq_data) in enumerate(quiz_data.items(), 1):
                            with st.container():
                                st.markdown(f"### Question {idx}")
                                st.markdown(f"**{mcq_data['mcq']}**")
                                
                                # Display options
                                for opt_key, option in mcq_data['options'].items():
                                    st.markdown(f"- **{opt_key.upper()}.** {option}")
                                
                                # Show correct answer with styling
                                st.markdown(f"**✅ Correct Answer:** {mcq_data['correct'].upper()}")
                                st.divider()
                    
                    with tab2:
                        st.markdown(f"**Subject:** {subject}")
                        st.markdown(f"**Difficulty:** {tone.capitalize()}")
                        st.markdown(f"**Total Questions:** {len(quiz_data)}")
                        st.markdown(f"**Text Length:** {len(TEXT)} characters")
                        
                        # Download option
                        st.download_button(
                            label="Download MCQs as JSON",
                            data=json.dumps(quiz_data, indent=2),
                            file_name=f"mcqs_{subject}_{tone}.json",
                            mime="application/json"
                        )
                
                except Exception as e:
                    st.error(f"Error generating MCQs: {str(e)}")
                    st.error("Please try again with a different file or check your API key.")
        
        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")
            st.error("Please ensure the file is not corrupted and contains readable text.")

# Sidebar with instructions
with st.sidebar:
    st.markdown("## Instructions")
    st.markdown("""
    1. **Subject**: Enter the subject area for your MCQs
    2. **Number of MCQs**: Choose how many questions to generate (1-20)
    3. **Difficulty**: Select the complexity level
    4. **File Upload**: Upload a PDF or text file with content
    5. **Generate**: Click to create your MCQs
    
    ### Supported Files
    - PDF files (.pdf)
    - Text files (.txt)
    
    ### Tips
    - Ensure your file contains enough text (at least 100 characters)
    - PDF files with images/scanned text may not work well
    - Use clear, educational content for better MCQs
    """)
    
    st.markdown("## About")
    st.markdown("This tool uses AI to generate multiple choice questions from your uploaded content.")