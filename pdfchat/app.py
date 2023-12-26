import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import openai
import os
import pandas as pd 
import altair as alt
import random



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
            

def get_api_support_suggestions(stress_level):
    # Temporarily set the OpenAI API key for this function's scope
    openai.api_key = os.getenv("OPENAI_API_KEY")

    prompt = f"I am a caseworker feeling a stress level of {stress_level} out of 10. Can you provide me with some support suggestions?"
    
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  # You can choose the model you want to use here
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except Exception as e:
        # Handle API errors or any exception
        st.error(f"An error occurred: {str(e)}")
        return "We're unable to get suggestions at the moment. Please try again later."
    
def get_mock_activity_suggestions(interest, location):
    # Mock data to simulate activity suggestions
    activities = {
        "soccer": [
            "Join the local soccer league in [Location]",
            "Free play soccer event this weekend at the downtown field in [Location]",
            "Attend a soccer game near you in [Location]"
        ],
        "hiking": [
            "Explore the trails at a local park in [Location]",
            "Guided nature hike at a nearby state park in [Location]",
            "Group hike event this weekend in [Location]"
        ]
        # More mock interests and activities can be added here
    }
    
    # Replace [Location] with the provided location
    if interest in activities:
        suggestions = [activity.replace("[Location]", location) for activity in activities[interest]]
        return suggestions
    else:
        return ["No activities found for your interest. Try another!"]

def simulate_risk_analysis(document_text):
    """
    Simulate a risk analysis based on the content of a case document.
    Randomly assigns a risk level to each category and provides an explanation for demonstration purposes.
    """
    risk_categories = [
        "Incarceration",
        "Substance Abuse",
        "Homelessness",
        "Academic Failure",
        "Chronic Unemployment",
        "Mental Health Issues",
        "Social Isolation",
        "Physical Health Problems",
        "Victim of Crime",
        "Poverty",
    ]
    
    risk_explanations = {
        "High": "There are multiple strong indicators that suggest an elevated risk based on the analysis of behavioral patterns, historical factors, and current circumstances.",
        "Medium": "There are some concerns based on certain observed patterns and historical data, suggesting a moderate risk level.",
        "Low": "Current analysis does not indicate significant risk factors at this time, suggesting a low risk level."
    }

    # Assign a risk level and a matching explanation to each risk category
    risks = {risk: (level := random.choice(["High", "Medium", "Low"]), risk_explanations[level])
             for risk in risk_categories}

    return risks

        
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Review Cases :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
        #         st.success("Documents processed. You may now ask questions about your documents.")
        # else:
        #     st.error("Please upload at least one document to proceed with the analysis.")

        st.subheader("Your Well-being")
        st.write("Let's find ways to reduce your stress and enjoy your day.")
        
        stress_level = st.slider("How stressed are you feeling today?", 0, 10, 0)
        interests = st.text_input("What do you enjoy doing? (e.g., soccer, hiking)")
        location = st.text_input("Your city (e.g., Orlando)")

        if st.button("Get Suggestions"):
            if interests and location:
                # First, provide support suggestions based on stress level
                support_message = get_api_support_suggestions(stress_level)
                st.success(support_message)

                # Then, provide activity suggestions based on interests and location
                activity_suggestions = get_mock_activity_suggestions(interests.lower(), location)
                st.write("Here are some activity suggestions for you:")
                for suggestion in activity_suggestions:
                    st.write("- " + suggestion)
            else:
                st.error("Please enter your interests and location for personalized suggestions.")

 # Section for risk assessment of cases
    st.subheader("Risk Assessment of Cases")
    uploaded_file = st.file_uploader("Upload a case document", type=["txt", "pdf"])
    
    if uploaded_file is not None:
        # Check the file type and process accordingly
        if uploaded_file.type == "application/pdf":
            # Use PyPDF2 for PDF files
            try:
                text = get_pdf_text([uploaded_file])
            except Exception as e:
                st.error(f"An error occurred while reading the PDF: {e}")
                text = ""
        else:
            # Assume text for non-PDF files and decode
            try:
                text = uploaded_file.read().decode("utf-8")
            except UnicodeDecodeError as e:
                st.error(f"File decoding error: {e}")
                text = ""

        # If text was extracted, perform risk assessment
        if text:
            if st.button("Assess Risks"):
                risk_assessment = simulate_risk_analysis(text)
                for risk, (level, explanation) in risk_assessment.items():
                    with st.expander(f"{risk} - Risk Level: {level}"):
                        st.write(explanation)



if __name__ == '__main__':
    main()
