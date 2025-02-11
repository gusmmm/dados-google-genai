"""
PDF to Markdown Converter for Medical Notes
----------------------------------------

This script processes medical PDF files and converts them into organized markdown documents.
It handles different types of medical notes:
- Admission notes (E)
- Release notes (A)
- Provisory death reports (BIC)
- Final death reports (O)

Dependencies:
- PyMuPDF (fitz): For PDF text extraction
- pathlib: For cross-platform path handling
- logging: For error tracking and debugging

Author: [Your Name]
Version: 1.0
"""

import fitz  # PyMuPDF
import os
from collections import defaultdict
from pathlib import Path
import traceback
import logging
from typing import Dict, Tuple, Optional
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
import sys

# Initialize rich console for beautiful terminal output
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_processing.log'),
        logging.StreamHandler()
    ]
)

# Constants
NOTE_TYPES = {
    'E': ('admission', 0),
    'A': ('release', 1),
    'BIC': ('provisory - death report', 2),
    'O': ('final - death report', 3)
}

class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors."""
    pass

def print_status(message: str, style: str = "info") -> None:
    """Print beautifully formatted status messages to the terminal.
    
    Args:
        message (str): The message to display
        style (str): The style of the message (info, error, success)
    """
    styles = {
        "info": "blue",
        "error": "red",
        "success": "green"
    }
    console.print(Panel(message, style=styles.get(style, "white")))

def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """Extracts text from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        Optional[str]: Extracted text or None if extraction fails
        
    Raises:
        PDFProcessingError: If there's an error processing the PDF
    """
    try:
        logging.info(f"Opening PDF file: {pdf_path}")
        with fitz.open(pdf_path) as doc:
            text = ""
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task(f"Processing {os.path.basename(pdf_path)}", total=len(doc))
                for page_num, page in enumerate(doc, 1):
                    text += page.get_text()
                    progress.update(task, advance=1)
            return text
    except Exception as e:
        error_msg = f"Error processing {pdf_path}: {str(e)}"
        logging.error(error_msg)
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise PDFProcessingError(error_msg)

def parse_filename(filename: str) -> Tuple[str, str]:
    """Parse PDF filename to extract patient number and note type.
    
    Args:
        filename (str): The filename without extension
        
    Returns:
        Tuple[str, str]: Patient number and note type
    """
    base_name = filename[:-4]  # Remove .pdf
    if base_name.endswith('BIC'):
        return base_name[:-3], 'BIC'
    return base_name[:-1], base_name[-1]

def create_markdown_content(patient_num: str, docs: Dict) -> str:
    """Create markdown content for a patient's documents.
    
    Args:
        patient_num (str): Patient identifier
        docs (Dict): Dictionary containing patient documents
        
    Returns:
        str: Formatted markdown content
    """
    markdown_content = [f"# Patient {patient_num} Medical Notes\n"]
    
    for order in range(4):
        if order in docs:
            note_name, content = docs[order]
            markdown_content.extend([
                f"## {note_name.title()} Note",
                "---",
                "```",
                content.strip(),
                "```",
                "\n"
            ])
    
    return "\n".join(markdown_content)

def process_medical_files(pdf_folder: str, output_folder: str) -> None:
    """Process medical PDFs and organize them by patient.
    
    Args:
        pdf_folder (str): Path to folder containing PDF files
        output_folder (str): Path to output folder for markdown files
    """
    try:
        # Create output folder
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        print_status(f"✓ Output folder ready: {output_folder}", "success")
        
        # Get PDF files
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
        print_status(f"Found {len(pdf_files)} PDF files to process", "info")
        
        # Process files
        patient_docs = defaultdict(dict)
        with Progress() as progress:
            task = progress.add_task("Processing PDF files...", total=len(pdf_files))
            
            for filename in pdf_files:
                try:
                    patient_num, note_type = parse_filename(filename)
                    
                    if note_type not in NOTE_TYPES:
                        print_status(f"Skipping unknown note type: {filename}", "error")
                        continue
                    
                    pdf_path = os.path.join(pdf_folder, filename)
                    text = extract_text_from_pdf(pdf_path)
                    
                    if text:
                        note_name, order = NOTE_TYPES[note_type]
                        patient_docs[patient_num][order] = (note_name, text)
                    
                    progress.update(task, advance=1)
                    
                except PDFProcessingError as e:
                    print_status(f"Error processing {filename}: {str(e)}", "error")
                    continue
        
        # Generate markdown files
        print_status(f"Generating markdown files for {len(patient_docs)} patients", "info")
        for patient_num, docs in patient_docs.items():
            try:
                markdown_content = create_markdown_content(patient_num, docs)
                output_path = os.path.join(output_folder, f"{patient_num}.md")
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                
                print_status(f"✓ Created: {patient_num}.md", "success")
                
            except Exception as e:
                print_status(f"Failed to create markdown for patient {patient_num}: {str(e)}", "error")
                logging.error(f"Traceback: {traceback.format_exc()}")
    
    except Exception as e:
        error_msg = f"Critical error: {str(e)}"
        logging.error(f"{error_msg}\n{traceback.format_exc()}")
        print_status(error_msg, "error")
        raise

if __name__ == "__main__":
    console.print("\n[bold blue]PDF to Markdown Converter[/bold blue]")
    console.print("=" * 50 + "\n")
    
    pdf_folder = "./dados/pdf_originals"
    output_folder = "./dados/md_files"
    
    try:
        process_medical_files(pdf_folder, output_folder)
        print_status("Conversion completed successfully!", "success")
    except Exception:
        print_status("Conversion failed! Check the logs for details.", "error")
        sys.exit(1)
