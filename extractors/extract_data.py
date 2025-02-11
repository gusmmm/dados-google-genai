"""
Medical Notes Data Extractor
---------------------------

This script uses Pydantic models and Google's Generative AI to extract structured
data from medical notes markdown files. It processes the files and saves the 
extracted data as JSON.

Dependencies:
- pydantic: For data validation and settings management
- google-genai: For text analysis and information extraction
- rich: For beautiful terminal output
- logging: For error tracking and debugging
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from genai.genai_config import initialize_genai, GenAIConfigError, print_status
from config import *
from extractors.classes_pydantic import PatientData, ExtractionError

# Initialize rich console for beautiful terminal output
console = Console()

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

def load_glossary() -> str:
    """Load and format the Portuguese medical terms glossary."""
    glossary_path = project_root / GLOSSARY_PATH
    with open(glossary_path, 'r', encoding='utf-8') as f:
        return f.read()

def extract_patient_data(file_path: str, genai_client) -> PatientData:
    """Extract patient data from a markdown file using GenAI."""
    try:
        # Read markdown file and glossary
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        glossary = load_glossary()
        
        # Prepare prompt for GenAI with explicit location instructions
        prompt = f"""You are a medical data extraction assistant specialized in structured data extraction. 
You understand Portuguese medical terminology. Use the following glossary to interpret Portuguese medical terms and abbreviations: {glossary}

Your task is to extract specific patient information from the medical notes below and output using the PatientData model.
Medical notes: {content}

Pay special attention to abbreviations like "UQ" which means "unidade de queimados" (burns unit).

Search for these fields in these specific locations:
1. id_patient: Number after "DOCUMENT ID:" in the metadata header
2. gender: In admission note section, "Feminino" → "F" or "Masculino" → "M"
3. full_name: In admission note section, full name after birth date
4. date_of_birth: In admission note section, date in format yyyy-mm-dd → convert to dd-mm-yyyy
5. process_number: In admission note section, the process number us in the line before the line containing "Nº Processo:."
6. address: In admission note section, complete street address and postal code before "Data Nasc:"
7. admission_date: In admission note section, date after "Data de admissão na UQ:" or similar text indicating admission to burns unit → format as dd-mm-yyyy
8. admission_time: In admission note section, time after "Hora de admissão na UQ:" or similar text indicating admission time → format as hh:mm
9. origin: In admission note section, look for previous location before UQ transfer. Use glossary to understand hospital unit abbreviations
10. discharge_date: In discharge note ("nota de alta") section, look for date before "Dr(a)." → format as dd-mm-yyyy
11. destination: In discharge note section, look for transfer location after UQ. Use glossary to understand hospital unit abbreviations

Output this exact JSON structure, using null for missing values:
{{
    "id_patient": <integer>,
    "gender": <"M" or "F">,
    "full_name": <string>,
    "date_of_birth": <"dd-mm-yyyy">,
    "process_number": <integer>,
    "address": <string>,
    "admission_date": <"dd-mm-yyyy">,
    "admission_time": <"hh:mm">,
    "origin": <string>,
    "discharge_date": <"dd-mm-yyyy">,
    "destination": <string>
}}"""
        
        console.print(f"\n[bold cyan]Using model:[/bold cyan] {MODEL_NAME}")
        
        # Generate content using the correct API methods
        response = genai_client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={'temperature': MODEL_TEMPERATURE}
        )
        
        # Log the raw response for debugging
        logging.debug(f"Raw GenAI response: {response.text}")
        
        # Extract JSON from response
        json_text = response.text.strip()
        
        # Try to find JSON in the response if it's not properly formatted
        start_idx = json_text.find('{')
        end_idx = json_text.rfind('}')
        
        if (start_idx == -1 or end_idx == -1):
            raise ExtractionError("No valid JSON object found in response")
            
        json_text = json_text[start_idx:end_idx + 1]
        
        try:
            # Parse JSON response
            data = json.loads(json_text)
            
            # Convert date strings to proper format if they exist
            for field in ['date_of_birth', 'admission_date', 'discharge_date']:
                if data.get(field):
                    # Ensure date is in dd-mm-yyyy format
                    parts = data[field].split('-')
                    if len(parts) == 3:
                        if len(parts[2]) == 4:  # If year is last
                            data[field] = f"{parts[0]:0>2}-{parts[1]:0>2}-{parts[2]}"
                        elif len(parts[0]) == 4:  # If year is first
                            data[field] = f"{parts[2]:0>2}-{parts[1]:0>2}-{parts[0]}"
            
            # Convert time to proper format if it exists
            if data.get('admission_time'):
                # Ensure time is in hh:mm format
                if ':' not in data['admission_time']:
                    data['admission_time'] = f"{data['admission_time'][:2]}:{data['admission_time'][2:]}"
            
            # Validate with Pydantic model
            patient_data = PatientData(**data)
            
            # Log successful extraction
            logging.info(f"Successfully extracted data from {file_path}")
            
            return patient_data
            
        except json.JSONDecodeError as e:
            raise ExtractionError(f"Failed to parse GenAI response as JSON: {str(e)}\nResponse was: {json_text}")
        except Exception as e:
            raise ExtractionError(f"Error processing extracted data: {str(e)}")
            
    except Exception as e:
        error_msg = f"Failed to extract data from {file_path}: {str(e)}"
        logging.error(error_msg)
        raise ExtractionError(error_msg)

def process_markdown_file(input_path: str, output_dir: str, genai_client) -> None:
    """Process a single markdown file and save extracted data as JSON.
    
    Args:
        input_path (str): Path to input markdown file
        output_dir (str): Directory to save JSON output
        genai_client: Initialized GenAI client
    """
    try:
        print_status(f"Processing: {Path(input_path).name}", "info")
        
        # Extract data
        patient_data = extract_patient_data(input_path, genai_client)
        
        # Create output path
        output_path = Path(output_dir) / f"{Path(input_path).stem}.json"
        
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(patient_data.model_dump(), f, indent=2, ensure_ascii=False)
            
        # Display extracted data
        console.print("\n[bold green]Extracted Data:[/bold green]")
        console.print(patient_data.model_dump_json(indent=2))
        
        print_status(f"✓ Saved to: {output_path}", "success")
        
    except Exception as e:
        print_status(f"Error processing {input_path}: {str(e)}", "error")
        raise

def main():
    """Main execution function."""
    console.print("\n[bold blue]Medical Notes Data Extractor[/bold blue]")
    console.print("=" * 50 + "\n")
    
    try:
        # Initialize GenAI client
        genai_client = initialize_genai()
        
        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print_status(f"✓ Output directory ready: {OUTPUT_DIR}", "success")
        
        # Process file
        process_markdown_file(DEFAULT_INPUT_FILE, OUTPUT_DIR, genai_client)
        
        print_status("Extraction completed successfully!", "success")
        
    except (GenAIConfigError, ExtractionError) as e:
        print_status(f"Failed to extract data: {str(e)}", "error")
        exit(1)
    except Exception as e:
        print_status(f"Unexpected error: {str(e)}", "error")
        logging.error(f"Unexpected error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()