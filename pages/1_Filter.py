import streamlit as st
import spacy
import re

# Import PDF extraction modules
import fitz
import io
from PIL import Image
import pytesseract
from PIL import Image
from PyPDF2 import PdfReader
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

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
    pdf_stream = io.BytesIO(pdf.getvalue())
    pdf_document = fitz.open("pdf", pdf_stream)
    text = ''
    for page_number in range(pdf_document.page_count):
        page = pdf_document.load_page(page_number)
        pix = page.get_pixmap()
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text += pytesseract.image_to_string(image)
    return text

def get_filtered_text(chunk, patterns, sentence_ender='. '):
    filtered_text = []
    for index, sentence in enumerate(chunk.split(sentence_ender)):
        sentence = sentence.strip() + '.'
        for pattern in patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                filtered_text.append(sentence)
                break
    return filtered_text

def get_patterns(filters_applied, file_path = r"regex_dict.txt"):
    filter_key_choices = ['By numbers', 'By citations with date', 'By citations without date', 'By alphabet bullets']
    filter_key_code = ['number', 'citation_with_year', 'citation_with_no_date', 'enumerate_alphabet']
    filters_coded = [filter_key_code[filter_key_choices.index(i)] for i in filters_applied]

    patterns = []
    with open(file_path, 'r') as file:
        for line in file:
            key, pattern = line.strip().split(': ', 1)
            if key in filters_coded:
                patterns.append(pattern)
    return patterns

def get_similar(chunk, keywords, similarity_level, sentence_ender='. '):
    nlp = spacy.load('en_core_web_sm')
    similar_text = []
    for index, sentence in enumerate(chunk.split(sentence_ender)):
        sentence = sentence.strip() + '.'
        for keyword in keywords:
            token1 = nlp(keyword)
            for word in sentence.split(' '):
                token2 = nlp(word)    
                similarity = token1.similarity(token2)
                if(similarity > similarity_level):
                    similar_text.append(sentence)
                    break
    return similar_text

def bullet_printer(s):
    s = s.split('\n')
    st.write(s[0])
    for i in s[1:]:
        st.markdown(i)
    st.write('\n')

def explainer():
    st.write("Filter sentences from your PDF using regex and similar words powered by spaCy, regardless if it's a digital or scanned PDF. To start, upload your PDF then choose a filtering method. ✍️")

    with st.expander('Filtering limitations'):
        tab1, tab2, tab3, tab4 = st.tabs(['Sentences', 'Citations', 'Alphabetical bullet list', 'Similarity'])
        
        with tab1:
            s = """**Sentences**
- Sentences are assumed to be written correctly or else there would be errors.
- The variable sentence_ender allows flexibility over how the chunk should be separated usually with the value of '.'"""
            bullet_printer(s)
        with tab2:
            s = """**Citations**
- Finds anything with a 4-digit number inside or "n.d." and is enclosed by parentheses
- Example for filter 'citation_with_year': (Firman, 1999), (Blunch, Canagarajah and Raju 2001), (World Bank 2006), (Portes and Hoffman 2003: 55)
- Example for filter 'citation_with_no_date': (Khalik, n.d.)"""
            bullet_printer(s)
        with tab3:
            s = """**Alphabetical List**
- Gets a format of: a single character + ')' + ends with a '.' or ','
- Unable to figure out a way for lists that uses '.' such as a. b. 1. 2. since they are usually recognized as the end of a sentence."""
            bullet_printer(s)
        with tab4:
            s = """**Spacy**
- Is able to get similar words from a keyword depending on the similarity level
- Context is not taken into consideration"""
            bullet_printer(s)          


def output_sentences(filtered_sentences):
    output_string = ''
    if filtered_sentences != []:
        st.write('**FILTERED SENTENCES:**')
        for i in filtered_sentences:
            st.write(i)
            st.write('\n')
            output_string += i + '\n'
    else:
        st.write('There are no filtered sentences based on your chosen method.')
    return output_string

def main():
    st.header('PDF Filtering Tool 🔍')

    explainer()
    st.divider()    

    # User settings
    st.file_uploader('Upload your PDF', type='pdf', key='pdf')
    st.text_area('Or... directly paste your text', key='text')
    st.radio(label='Choose your filter method', options=['By keyword', 'By filter keys'], horizontal=True, key='method')

    # Get keyword input
    if st.session_state.method == 'By keyword':
        st.text_area('Type your comma-separated keywords', help='Input example: macroeconomics, Marxism, free trade', key='keywords')
        st.radio('Similarity level', ['Low', 'Medium', 'High'], horizontal=True, key='similarity_level')
    else:
        st.multiselect('Choose your filter keys', ['By numbers', 'By citations with date', 'By citations without date', 'By alphabet bullets'], key='filter_keys')

    st.button('Filter', disabled=False, key='button')

    pdf_val, text_val = st.session_state.pdf is not None, (st.session_state.text is not None) or (st.session_state.text != '')
    text = ''

    if st.session_state.button:
        # Extract text
        st.write()
        if pdf_val:
            # PDF extraction
            if pdf_has_text(st.session_state.pdf):
                text = extract(st.session_state.pdf)
            else:
                text = extract_with_ocr(st.session_state.pdf)            
        else:
            text = st.session_state.text
        st.success('Done extracting text.')
    
        # Filter
        with st.spinner('Filtering text...'):
            if st.session_state.method == 'By keyword':
                keywords = [i.strip() for i in st.session_state.keywords.split(',')]
                similarity_level_dict = {'Low':0.5, 'Medium':0.8, 'High':0.95}
                similarity_level = similarity_level_dict[st.session_state.similarity_level]

                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=UserWarning)
                    filtered_sentences = get_similar(text, keywords, similarity_level)
            else:
                if st.session_state.filter_keys == []:
                    st.error('At least one filter key must be selected.')
                else:
                    patterns = get_patterns(st.session_state.filter_keys)
                    filtered_sentences = get_filtered_text(text, patterns)
            output_str = output_sentences(filtered_sentences)

        st.success('Done filtering sentences.')
        st.download_button('Optional: download the filtered sentences as a TXT file.', output_str, file_name='filtered_sentences.txt')

if __name__ == '__main__': 
    main()