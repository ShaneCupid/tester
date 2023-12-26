# MultiPDF Chat App

> A Streamlit-based conversational AI tool for interacting with multiple PDF documents.

## Introduction

The MultiPDF Chat App leverages language models to provide quick and context-aware answers from PDF documents, enhancing productivity and decision-making for professionals.

## Features

### Review Cases
- **Functionality**: Searches historical case files using natural language queries.
- **Technical Insight**: Employs NLP for document indexing and retrieval, similar to a specialized search engine.
- **Benefit**: Immediate access to case histories and legal precedents, aiding informed decisions.

### Your Well-being
- **Functionality**: Recommends local events and activities for stress management.
- **Technical Insight**: Integrates with APIs from platforms like Meetup or Facebook to provide up-to-date local event information.
- **Benefit**: Promotes mental health and work-life balance, aiming to reduce burnout.

### Risk Assessment of Cases
- **Functionality**: Predicts potential risks in case documents.
- **Technical Insight**: Uses data analytics and AI models to assess and explain risks.
- **Benefit**: Offers foresight into case outcomes, facilitating preemptive action.


## How It Works
------------

![MultiPDF Chat App Diagram](./docs/PDF-LangChain.jpg)

The application follows these steps to provide responses to your questions:

1. PDF Loading: The app reads multiple PDF documents and extracts their text content.

2. Text Chunking: The extracted text is divided into smaller chunks that can be processed effectively.

3. Language Model: The application utilizes a language model to generate vector representations (embeddings) of the text chunks.

4. Similarity Matching: When you ask a question, the app compares it with the text chunks and identifies the most semantically similar ones.

5. Response Generation: The selected chunks are passed to the language model, which generates a response based on the relevant content of the PDFs.

## Dependencies and Installation
----------------------------
To install the MultiPDF Chat App, please follow these steps:

1. Clone the repository to your local machine.

2. Install the required dependencies by running the following command:
   ```
   pip install -r requirements.txt
   ```

3. Obtain an API key from OpenAI and add it to the `.env` file in the project directory.
```commandline
OPENAI_API_KEY=your_secrit_api_key
```

## Usage
-----
To use the MultiPDF Chat App, follow these steps:

1. Ensure that you have installed the required dependencies and added the OpenAI API key to the `.env` file.

2. Run the `main.py` file using the Streamlit CLI. Execute the following command:
   ```
   streamlit run app.py
   ```

3. The application will launch in your default web browser, displaying the user interface.

4. Load multiple PDF documents into the app by following the provided instructions.

5. Ask questions in natural language about the loaded PDFs using the chat interface.

## Contributing
------------
This repository is intended for educational purposes and does not accept further contributions. It serves as supporting material for a YouTube tutorial that demonstrates how to build this project. Feel free to utilize and enhance the app based on your own requirements.

## License
-------
The MultiPDF Chat App is released under the [MIT License](https://opensource.org/licenses/MIT).