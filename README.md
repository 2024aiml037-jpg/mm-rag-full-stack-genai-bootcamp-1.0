# Multimodal RAG Full-Stack GenAI Bootcamp

A full-stack Generative AI project for learning how to build a multimodal
Retrieval-Augmented Generation (RAG) application. The project will demonstrate
how documents and other supported content can be ingested, processed, indexed,
retrieved, and supplied to a large language model to produce grounded answers.

> [!NOTE]
> This repository is currently in its initial setup phase. Application features,
> dependencies, and detailed run commands should be added as they are implemented.

## Project goals

- Build an end-to-end RAG workflow.
- Process and retrieve information from multimodal source content.
- Generate responses grounded in retrieved context.
- Connect the AI pipeline to a user-facing application.
- Organize the backend, frontend, configuration, and supporting services as a
  maintainable full-stack project.

## Expected RAG workflow

1. Load source documents or other supported content.
2. Extract, clean, and divide the content into useful chunks.
3. Create embeddings that represent those chunks.
4. Store the embeddings and metadata in a vector store.
5. Retrieve the most relevant context for a user's question.
6. Send the question and retrieved context to a language model.
7. Return a grounded answer through the application interface.

## Prerequisites

Before setting up the project, install:

- [uv](https://docs.astral.sh/uv/)
- Python 3.12 (it can also be installed and managed through `uv`)
- Git

## Python version

This project uses **Python 3.12**.

List the Python versions available to `uv`:

```cmd
uv python list
```

If Python 3.12 is not available, install it with:

```cmd
uv python install 3.12
```

## Setup

### 1. Clone the repository

```cmd
git clone <repository-url>
cd mm-rag-full-stack-genai-bootcamp-1.0
```

Replace `<repository-url>` with the URL of this repository.

### 2. Create the virtual environment

Create a virtual environment named `env` using Python 3.12:

```cmd
uv venv env --python 3.12
```

The general form of the command is:

```cmd
uv venv env --python <python-version>
```

### 3. Activate the virtual environment

On Windows Command Prompt:

```cmd
env\Scripts\activate.bat
```

On macOS or Linux:

```bash
source env/bin/activate
```

After activation, confirm the selected Python version:

```cmd
python --version
```

The output should report Python 3.12.x.

### 4. Install dependencies

Install the dependencies listed in `requirements.txt`:

```cmd
uv pip install -r requirements.txt
```

The requirements file is currently empty and should be updated as project
packages are introduced.

### 5. Configure environment variables

Store local configuration, API keys, model settings, and service credentials in
the `.env` file. For example:

```dotenv
# Add only the variables required by the implemented services.
# LLM_API_KEY=your_api_key
# MODEL_NAME=your_model_name
```

Never commit real secrets or API keys. The `.env` file and `env` virtual
environment directory are excluded through `.gitignore`.

## Current project structure

```text
mm-rag-full-stack-genai-bootcamp-1.0/
|-- .env               # Local environment variables (not committed)
|-- .gitignore         # Files and directories excluded from Git
|-- README.md          # Project documentation
|-- requirements.txt   # Python dependencies
`-- env/               # Local Python virtual environment (not committed)
```

The structure will expand as the ingestion pipeline, retrieval layer, AI
services, backend API, frontend application, tests, and supporting configuration
are added.

## Common commands

```cmd
# List available Python installations
uv python list

# Create the Python 3.12 virtual environment
uv venv env --python 3.12

# Activate it in Command Prompt
env\Scripts\activate.bat

# Install project dependencies
uv pip install -r requirements.txt

# Leave the virtual environment
deactivate
```

## Development guidelines

- Use Python 3.12 for consistent local development.
- Activate the virtual environment before running Python commands.
- Add new Python dependencies to `requirements.txt`.
- Keep credentials and machine-specific settings in `.env`.
- Add tests as each application component is implemented.
- Update this README whenever setup or run commands change.

## Planned components

- Content ingestion and preprocessing
- Text and multimodal content extraction
- Chunking and metadata management
- Embedding generation
- Vector storage and semantic retrieval
- LLM-based response generation
- Backend API
- Frontend interface
- Evaluation, testing, logging, and observability

## Status

Initial project setup is complete. Feature implementation has not yet been added.
