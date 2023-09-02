import streamlit as st

# Import PDF extraction modules
from pdf2image import convert_from_bytes
import pytesseract
from PyPDF2 import PdfReader
pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR\tesseract.exe'

# Import Langchain modules
from langchain.chat_models import ChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain

# Import text splitter packages to chunk data based on specified number of tokens
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter 
size, overlap = 4000, 50

# PDF extraction functions
def pdf_has_text(pdf):
    reader = PdfReader(pdf)
    for page in reader.pages:
        text = page.extract_text()
        if text.strip():
            return True
    return False

def extract(pdf):
    reader = PdfReader(pdf)
    text = ''
    for page in reader.pages:
        text += page.extract_text()
    return text

@st.cache_data
def extract_with_ocr(pdf):
    pdf_bytes = pdf.getvalue()
    images = convert_from_bytes(pdf_bytes)
    text = ''
    for image in images:
        text += pytesseract.image_to_string(image)
    return text

@st.cache_data
def summarizer(pdf_contents, api_key, chain_type='simple', chunk_number=size):
    '''Inputs one string literal, number of tokens (max of 4000 tokens), and chain type:
        (a) 'simple' refers to 'stuff' method
        (b) 'combined' combines individual summaries per chunk into one string
        (d) 'complex' refers to 'map_reduce' method
        (d) 'recursive' refers to refine method
    '''

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k", openai_api_key=api_key)

    if chunk_number > 4000:
        raise ValueError('The number of tokens per chunk cannot exceed beyond 4000.')

    # Split the text based on number of tokens
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
    chunks = text_splitter.create_documents([pdf_contents])

    if chain_type in ['simple', 'complex', 'recursive']:
        method_mapping = {'simple': 'stuff', 'complex': 'map_reduce', 'recursive': 'refine'}
        # Summarization model
        chain = load_summarize_chain(llm, chain_type = method_mapping.get(chain_type), verbose = False)
        summary = chain.run(chunks)
        return summary

    else:
        chain = load_summarize_chain(llm, chain_type = 'stuff', verbose = False)
        # Create new list to store chunked summaries
        summary_list = [chain.run([document]) for document in chunks]
        return '\n\n'.join(summary_list)

def explainer():
    st.write("Summarize PDF content using OpenAI GPT 3.5 Turbo-16K powered by LangChain framework, regardless if it's a digital or scanned PDF. To start, upload your PDF file then choose a summarization method. 💡")

    with st.expander("How does the summarization work?"):
            overview, technical = st.tabs(['General Overview', 'Dependencies'])
            with overview:
                st.write('**Text Extraction**: The initial step involves retrieving the text from the PDF. For digitally created PDFs, the text is directly extracted. For scanned documents or images, the app employs OCR (Optical Character Recognition) to identify and extract the text.')
                st.write("**AI Analysis**: After extraction, the text is processed by OpenAI's GPT-3.5 Turbo-16K. Based on the summarization method selected by the user, the AI analyzes the content. Note that each summary requires a certain number of tokens, which is the AI's method of quantifying its processing.")
                st.write("**Summary Output**: Finally, the app outputs the generated summary.")
            with technical:
                st.write('in progress')
                
    with st.expander("Which summarization method should I choose?"):
        help_captions = ['Use **Simple** for a quick, efficient summary of your text. The LLM will analyze the text as a whole and output a short paragraph.','Use **Complex** for a more in-depth summary. This works by dividing the text into chunks and creating an individual summary for each chunk. All of these individual summaries will be aggregated into one summary.','Use **Recursive** as another option for an in-depth summary. This works by first dividing the text into chunks. The first chunk will be summarized alone. Then this summary will be combined with the second chunk and summarized as one chunk. This is recursively done for all chunks.','Use **Combined** for a more detailed and longer summary which may take a longer time to process. This works by dividing the text into chunks and creating an individual summary for each chunk. Each summary will be combined as-is to produce one summary.']
        options = ['Simple', 'Complex', 'Recursive', 'Combined']

        tab1, tab2, tab3, tab4 = st.tabs(options)
        for i,tab in enumerate([tab1, tab2, tab3, tab4]):
            with tab:
                st.write(help_captions[i])

def main():
    st.set_page_config(page_title='PDF Analyzer')
    st.header('GPT 3.5-based Reading Summarizer 📖')

    explainer()
    st.divider()

    # User settings
    st.text_input('Type your OpenAI API key', key='openai_api_key')
    st.file_uploader('Upload your PDF', type='pdf', key='pdf')
    st.text_area('Or... directly paste your text', key='text')
    st.radio(label='Choose your summarization method', options=['Simple', 'Complex', 'Recursive', 'Combined'], horizontal=True, key='method')
    
    if len(st.session_state.openai_api_key) > 1:
        st.button('Summarize', disabled=False, key='button')
    else:
        st.button('Summarize', disabled=True, key='button')

    pdf_val, text_val = st.session_state.pdf is not None, (st.session_state.text is not None) or (st.session_state.text != '')

    text = ''
    if st.session_state.button:
        if pdf_val:
            # PDF extraction
            if pdf_has_text(st.session_state.pdf):
                text = extract(st.session_state.pdf)
            else:
                text = extract_with_ocr(st.session_state.pdf)
            
        else:
            text = st.session_state.text
        st.success('Done extracting text.')
        # Summarize
        summarized_text = summarizer(text, chain_type=st.session_state.method.lower(), chunk_number=size, api_key=st.session_state.openai_api_key)
        st.write(summarized_text)
        st.success('Done summarizing text.')
    
        st.download_button('Optional: download the extracted text as a TXT file.', text, file_name='extracted_text.txt')
        st.download_button('Optional: download the summarized text as a TXT file.', summarized_text, file_name='summarized_text.txt')

if __name__ == '__main__': 
    main()