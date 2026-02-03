# Persian NLP Service

## Overview
This repository contains an academic-oriented, modular AI-based service for **Persian document understanding**.  
The system focuses on extracting, cleaning, segmenting, and summarizing Persian text from heterogeneous document formats, including scanned documents.

The project is designed as a **pipeline-based NLP service**, integrating OCR, text normalization, chunking, and abstractive summarization, with optional LLM-based enhancements.

---

## Objectives
The main objectives of this project are:

- Extract text from Persian documents (scanned and digital)
- Improve OCR output quality through post-processing
- Support long-document processing via chunking
- Generate meaningful summaries of Persian texts
- Provide a clean and extensible API-based architecture
- Maintain academic clarity, modularity, and reproducibility

---

## Supported Input Formats
- Images: PNG, JPG, JPEG, BMP, TIFF
- PDF documents (scanned)
- Text files (TXT)

> Support for DOCX files and MIME-based file type detection is planned as future work.

---

## System Architecture

### High-Level Components
The system is composed of the following layers:

- **API Layer**
  - FastAPI-based endpoints for document upload and processing
- **Core Layer**
  - Configuration management and shared utilities
- **Service Layer**
  - OCR Service
  - Text Cleaning Service
  - Chunking Module
  - Summarization Service
- **External Services**
  - OCR engine (Tesseract)
  - Large Language Models (via API)
- **Persistence Layer (Planned)**
  - Storage of OCR and summarization results

---

## Processing Pipeline
The overall processing flow is as follows:

1. File upload via API
2. Temporary file storage
3. File type detection
4. Text extraction:
   - OCR for images and scanned PDFs
   - Direct extraction for text files
5. Optional text cleaning using LLM
6. Text segmentation (chunking)
7. Text summarization
8. Structured API response

---

## OCR Service

### OCR Engine
The OCR component is implemented using **Tesseract OCR** with the following configuration:

- OCR Engine Mode: `--oem 3`
- Page Segmentation Mode: `--psm 6`
- Preserved inter-word spacing
- Language support: Persian (`fas`) and English (`eng`)

### PDF Processing Strategy
- PDF pages are converted to images using `pdf2image`
- OCR is applied page-by-page
- For large documents, a configurable page limit is applied to prevent resource exhaustion
- Each page is labeled in the extracted output for traceability

---

## Text Cleaning and Normalization

An optional post-OCR cleaning step is implemented using a Large Language Model (LLM).  
This step aims to:

- Correct OCR-induced spelling errors
- Normalize spacing and Persian half-spaces (ZWNJ)
- Remove OCR artifacts and noise
- Preserve the original semantic meaning

This step can be enabled or disabled per request to control latency and cost.

---

## Chunking Module
To handle long documents effectively, extracted text is divided into logical chunks based on:

- Word count constraints
- Sentence boundary preservation
- Configurable chunk size limits

This design prevents sentence truncation and improves summarization quality.

---

## Summarization Service

The summarization module supports multiple approaches:

- Rule-based summarization
- Statistical summarization (e.g., TextRank)
- Transformer-based abstractive summarization

### Model Selection
The project uses **mT5 Multilingual XLSum** for abstractive summarization due to:

- Multilingual capability with good Persian performance
- Lower computational requirements compared to large gated models
- Public availability without authentication requirements
- Optimization for summarization tasks

---

## API Example

### OCR Endpoint

POST /ocr


#### Parameters
- `file`: input document
- `start_page` (optional): starting page for PDF
- `end_page` (optional): ending page for PDF
- `clean_text`: enable LLM-based text cleaning

#### Response Example
```json
{
  "type": "pdf",
  "filename": "document.pdf",
  "detail": "extracted and processed text",
  "cleaned": true
}

Project Structure

app/
├── api/
│   └── ocr_routes.py
├── core/
│   └── config.py
├── services/
│   ├── ocr_service.py
│   └── summarizer_service.py
├── models/
├── db/
└── main.py

Limitations

    File type detection currently relies on file extensions

    DOCX files are not yet supported

    OCR accuracy depends on input scan quality

    LLM-based cleaning introduces latency and external dependency

    No asynchronous processing for large files in the current version

Future Work

    MIME-based file type detection

    DOCX support with embedded image OCR

    Detection of digital vs scanned PDFs

    Advanced image preprocessing (deskewing, denoising)

    Asynchronous processing using task queues

    Database persistence for OCR and summarization results

    Docker-based deployment

    Evaluation metrics for OCR and summarization quality

Academic Context

This project is developed as part of an academic effort in the field of:

    Natural Language Processing (NLP)

    Information Retrieval (IR)

    Document Understanding

    Persian Language Processing

The architecture and design decisions prioritize clarity, extensibility, and reproducibility.
Author

Zeynab Arianmanesh
License

This project is intended for academic and research purposes
