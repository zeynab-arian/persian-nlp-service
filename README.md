# Technical Report  
## Design and Implementation of an AI-based Persian NLP Service

**Course / Project:** NLP / Information Retrieval  
**Student:** Zeynab Arianmanesh  
**Supervisor:** Dr. Ahad Herati  
**Date:** 1404/09/16  

---

## 1. Introduction

With the increasing volume of unstructured textual data, especially in Persian language documents, there is a growing need for intelligent systems capable of extracting, cleaning, and analyzing text from heterogeneous sources. Many real-world documents exist in non-editable formats such as scanned PDFs or images, making them inaccessible to traditional NLP pipelines.

This project focuses on the design and implementation of a **modular AI-based Persian NLP service**, referred to as **NLP-AI Service**, which integrates OCR, text correction, information extraction, classification, and summarization into a unified and extensible API-based system.

The primary objective of this project is to provide a reusable and scalable infrastructure for Persian document understanding in academic and applied contexts.

---

## 2. System Overview

The NLP-AI service is designed as a **pipeline-oriented and modular system**, where each NLP capability is implemented as an independent service. This design allows flexibility in selecting processing strategies, models, and configurations based on user requirements.

### 2.1 Core Modules

The system currently consists of the following main modules:

1. Text Correction Module  
2. Text Classification Module  
3. Information Extraction Module  
4. OCR Module  
5. Text Summarization Module  

In addition, several optional parameters are provided to control processing behavior, such as token limits and summarization ratio.

---

## 3. Overall Architecture

### 3.1 Architectural Layers

The system architecture is organized into the following layers:

- **API Layer**  
  Responsible for handling HTTP requests, file uploads, and response formatting using FastAPI.

- **Core Layer**  
  Manages configuration, environment variables, and shared utilities using `pydantic_settings`.

- **Service Layer**  
  Contains independent services for OCR, text cleaning, chunking, and summarization.

- **Model Layer**  
  Includes local NLP models and interfaces to external LLM providers.

- **External Services**  
  OCR engines (Tesseract), HuggingFace models, and LLM APIs (via OpenRouter).

- **Persistence Layer (Planned)**  
  Storage for OCR outputs, summaries, and metadata.

---

## 4. OCR Module

### 4.1 Purpose

The OCR module is responsible for converting non-editable documents into plain text suitable for downstream NLP processing. It serves as the entry point for scanned documents and images.

### 4.2 Supported Input Formats

- Image files (PNG, JPG, JPEG, BMP, TIFF)
- PDF documents (scanned)
- Text files (TXT)

### 4.3 OCR Pipeline

The OCR processing flow is implemented as follows:

1. Temporary storage of uploaded file
2. File type detection (based on file extension in current version)
3. Text extraction:
   - Images: grayscale conversion followed by OCR
   - PDFs: page-by-page image conversion and OCR
   - TXT: direct text reading
4. Optional post-OCR text cleaning using an LLM
5. Structured API response

### 4.4 OCR Engine Configuration

The OCR engine is implemented using **Tesseract OCR** with the following settings:

- OCR Engine Mode: `--oem 3`
- Page Segmentation Mode: `--psm 6`
- Inter-word spacing preservation enabled
- Language models: Persian (`fas`) and English (`eng`)

These settings are selected to optimize OCR performance for Persian documents and mixed-language content.

### 4.5 PDF Processing Strategy

For PDF files, the system:
- Extracts total page count
- Converts selected pages to images using `pdf2image`
- Applies OCR per page
- Adds page markers to the extracted text for traceability

To prevent excessive resource usage, a configurable page preview limit is applied when page ranges are not explicitly specified.

---

## 5. Post-OCR Text Cleaning

OCR outputs often contain noise, incorrect spacing, and spelling errors, especially for Persian text. To address this issue, an optional **LLM-based text cleaning service** is implemented.

### 5.1 Cleaning Objectives

- Correction of OCR-induced spelling errors
- Normalization of Persian spacing and ZWNJ
- Removal of OCR artifacts
- Preservation of original semantic content

### 5.2 Design Considerations

- Cleaning is optional and controlled via API parameters
- External LLM usage introduces latency and cost
- Fallback mechanisms return raw OCR output in case of failure

---

## 6. Text Chunking Strategy

Long documents are divided into logical chunks before further processing. Unlike naive token-based segmentation, the system:

- Splits text based on word count
- Preserves sentence boundaries
- Avoids cutting sentences mid-structure

This approach improves summarization quality and ensures coherent outputs.

---

## 7. Summarization Module

### 7.1 Supported Methods

The summarization service supports multiple approaches:

- Rule-based summarization
- Statistical summarization (TextRank)
- Transformer-based abstractive summarization

### 7.2 Model Selection and Justification

Initially, the Mistral-7B model was considered; however, it was replaced with **mT5 Multilingual XLSum** due to the following reasons:

- Mistral-7B is a gated model requiring authentication
- High computational resource requirements
- mT5 XLSum is publicly available
- Optimized for multilingual summarization, including Persian
- More suitable for deployment on limited hardware

---

## 8. Additional NLP Modules

### 8.1 Text Correction Module

Provides two correction strategies:
- LLM-based correction
- Local Persian-optimized correction model

### 8.2 Text Classification Module

Supports:
- Training classification models on labeled data
- Model persistence
- Independent prediction routes for new data

### 8.3 Information Extraction Module

Extracts domain-specific keywords and key information from input texts based on user-defined domains.

---

## 9. Implementation Details

### 9.1 Project Structure

app/
├── api/
├── core/
├── services/
├── models/
├── db/


### 9.2 Configuration Management

- Environment variables stored in `.env`
- Centralized configuration using `pydantic_settings`
- Default database configuration using SQLite

---

## 10. Development Timeline and Progress

The development process includes multiple coding phases covering OCR, summarization, and service integration. Key completed actions include:

- Implementation of multi-format OCR
- Integration of LLM-based text cleaning
- Modular summarization strategies
- API design for document-to-summary processing

Ongoing and planned tasks include database persistence, asynchronous task queues, testing, and containerization.

---

## 11. Limitations

- File type detection relies on file extensions
- DOCX files are not yet supported
- OCR accuracy is sensitive to scan quality
- LLM-based processing introduces external dependencies
- No asynchronous processing for large files in the current version

---

## 12. Future Work

Planned improvements include:

- MIME-based file type detection
- DOCX support with embedded image OCR
- Digital vs scanned PDF detection
- Advanced image preprocessing
- Task queue integration (Celery / RQ)
- Persistent storage for OCR and summaries
- Docker-based deployment
- Formal evaluation metrics for OCR and summarization quality

---

## 13. Conclusion

This project presents a comprehensive and extensible AI-based service for Persian NLP tasks, integrating OCR, text correction, and summarization into a unified architecture. The modular design enables future extensions and supports both academic research and practical applications in document understanding.

---
