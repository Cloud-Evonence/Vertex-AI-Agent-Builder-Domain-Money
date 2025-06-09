# Vertex-AI-Agent-Builder-Domain-Money
This project automates the extraction of structured data from PDF documents stored in Google Cloud Storage (GCS). It uses Gemini (generative AI) for classification and extraction, and LangGraph for managing the workflow.


Step 1: Features

- Classifies PDF documents (bank statement, payslip, loan, brokerage)
- Extracts and normalizes text using Gemini
- Evaluates field-level confidence
- Stores structured JSON output in GCS


Step 2: Environment Variables

INPUT_BUCKET: The name of the GCS bucket where input PDF files are stored. Default is domain-evonance-storage.

INPUT_PREFIX: The folder path within the input bucket that contains the PDF documents. Default is source_documents/.

OUTPUT_BUCKET: The GCS bucket where the structured JSON output will be saved. Default is outputjson4pdf.

MAX_PDFS: The maximum number of PDF files to process in a single run. Default is 10.