# Intelligent Document Parser

This project provides a FastAPI-based service for intelligently parsing and extracting structured data from various financial documents like bank statements, pay stubs, loan statements, and brokerage statements. It leverages Google's Gemini Pro 1.5 Flash model and LangGraph to create a robust, multi-step processing pipeline that includes document classification, text extraction (OCR), data normalization, and confidence scoring.

## Features

- Handles PDF, PNG, JPG, and JPEG formats.
- Automatic document classification (bank, pay, loan, brokerage).
- Gemini-based text and table extraction.
- Structured JSON output based on predefined schemas.
- Field-level confidence scoring for extracted data.
- Post-processing and data cleaning for improved accuracy.
- API endpoint for easy integration.
- Containerized with Docker for simple deployment.

## Supported Document Types

- Bank Statements
- Pay Stubs
- Loan Statements
- Brokerage Statements

## Technology Stack

- Python
- FastAPI
- LangGraph
- Google Gemini API
- Docker
- Uvicorn

## Workflow

The document processing pipeline is built using LangGraph and consists of the following steps:

1.  **Fetch Document**: Receives the uploaded file.
2.  **Classify Document**: Determines the document type (bank, pay, loan, brokerage) using Gemini.
3.  **Extract Text**: Performs OCR and text transcription using Gemini.
4.  **Parse & Normalize**: Extracts structured data into a JSON object based on the document type's schema, again using Gemini.
5.  **Post-Process Data**: Cleans and refines the extracted data (e.g., date formatting, number parsing, data consolidation).
6.  **Evaluate Confidence**: Gemini evaluates the confidence of each extracted field.
7.  **Output JSON**: Returns the final JSON output with the overall confidence score.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd evonence-doc-parser
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set Environment Variable:**
    You must set your Gemini API key as an environment variable.
    ```bash
    export GEMINI_API_KEY='YOUR_API_KEY'
    ```

## Running the Application

### Using Docker

1.  **Build the Docker image:**
    ```bash
    docker build -t doc-parser .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run -p 8080:8080 -e GEMINI_API_KEY='YOUR_API_KEY' doc-parser
    ```

### Running Locally

```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

## API Endpoint

### POST /process

Upload one or more document files to be processed. The service logs the processed JSON output to the console and returns a status message upon completion.

**Request:**

```bash
curl -X POST http://localhost:8080/process \
  -F "files=@/path/to/your/file1.pdf" \
  -F "files=@/path/to/your/file2.png"
```

**Response:**

```json
{
  "status": "Processing complete"
}
```
The detailed JSON output for each file will be printed in the server logs.

## Colab Notebook

The file `colab_workflow_with_confidence_score.py` provides a demonstration of the core document processing logic in a Google Colab environment. It is useful for testing, experimentation, and running the workflow step-by-step without setting up the full FastAPI service.

```
