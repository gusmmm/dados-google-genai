"""PDF Manager package for handling medical document conversions.

This package provides tools for converting medical PDF documents to markdown format
and cleaning the generated markdown files, with support for different types of medical
notes and beautiful terminal output.
"""

from .pdf_to_md import (
    process_medical_files,
    extract_text_from_pdf,
    PDFProcessingError,
    print_status,
    parse_filename,
    create_markdown_content,
    NOTE_TYPES
)

from .md_clean import (
    MDCleaner,
    MarkdownCleaningError,
    process_directory
)

__version__ = "0.1.0"
__all__ = [
    # PDF to Markdown conversion
    "process_medical_files",
    "extract_text_from_pdf",
    "PDFProcessingError",
    "print_status",
    "parse_filename",
    "create_markdown_content",
    "NOTE_TYPES",
    
    # Markdown cleaning
    "MDCleaner",
    "MarkdownCleaningError",
    "process_directory"
]