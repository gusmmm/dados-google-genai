# Project Brief: Google GenAI Data Processing

## Core Purpose
This project is a full stack solution to the management of the patients from the critical care burns unit data. Build a system for extracting, processing, and analyzing data from PDF documents using Google's Generative AI capabilities. The system will convert PDFs to markdown, clean the data, and utilize AI for advanced data processing.

## Core Requirements
1. PDF to Markdown conversion pipeline. The patient’s data comes originally in PDF files. The patient’s data PDF files are first converted to .md documents and formatted to be LLM ready. 
All data will be extracted using pydantic ai models and classes to maintain a specific data structure
All the extracted data will initially be stored in json files
2. Data cleaning and standardization
3. Integration with Google GenAI
4. Structured data extraction using Pydantic models
5. Batch processing capabilities
6. Backend setup is a mongodb database with fastapi, uvicorn, motor, pydantic, pydantic-ai. Create endpoints for CRUD operations.
7. Frontend setup: is a React project using Vite with routing and basic components, services for API communication and a patient data management interface

## Project Goals 
1. Create a reliable PDF processing pipeline
2. Implement efficient data extraction patterns
3. Ensure clean, standardized data output
4. Leverage AI capabilities for intelligent data processing
5. Maintain modular and extensible architecture
6. Create a fully functional full stack solution for data management in a mongodb database
7. To have the ability to add new data, modify existing data, delete data
8. To Have the capacity to extract tables and data in other hformats to be analyzed - statistical analysis, data visualization, data mining

