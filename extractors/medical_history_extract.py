"""
Medical History Data Extractor
---------------------------
This script uses Pydantic models and Google's Generative AI to extract medical history data
from patient medical notes markdown files. It processes the files and extracts structured data
about previous medical conditions, surgical history, and medications.

Dependencies:
- pydantic: For data validation and settings management
- google-genai: For text analysis and information extraction
- rich: For beautiful terminal output
- logging: For error tracking and debugging
"""
import json
import logging
import sys
from typing import List, Optional
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from pydantic import BaseModel, Field

from genai.genai_config import initialize_genai, GenAIConfigError, print_status
from config import *

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

class MedicalCondition(BaseModel):
    """Model representing a medical condition."""
    name: str = Field(description="Name of the medical condition")
    details: Optional[str] = Field(default=None, description="Additional details about the condition")
    
class Surgery(BaseModel):
    """Model representing a previous surgery."""
    procedure: str = Field(description="Name of the surgical procedure")
    date: Optional[str] = Field(default=None, description="Date of the procedure (dd-mm-yyyy)")
    details: Optional[str] = Field(default=None, description="Additional details about the surgery")
    
class Medication(BaseModel):
    """Model representing a medication."""
    name: str = Field(description="Name of the medication")
    dosage: Optional[str] = Field(default=None, description="Dosage of the medication")
    frequency: Optional[str] = Field(default=None, description="Frequency of intake (e.g., daily, twice daily)")
    details: Optional[str] = Field(default=None, description="Additional details about the medication")

class MedicalHistoryData(BaseModel):
    """Model for representing a patient's medical history."""
    medical_conditions: List[MedicalCondition] = Field(default_factory=list, description="List of previous medical conditions")
    surgical_history: List[Surgery] = Field(default_factory=list, description="List of previous surgeries")
    medications: List[Medication] = Field(default_factory=list, description="List of medications the patient was taking")
    allergies: Optional[str] = Field(default=None, description="Known allergies")
    other_notes: Optional[str] = Field(default=None, description="Other relevant medical history notes")

class ExtractionError(Exception):
    """Custom exception for data extraction errors."""
    pass

def load_glossary() -> str:
    """Load and format the Portuguese medical terms glossary."""
    glossary_path = project_root / GLOSSARY_PATH
    try:
        with open(glossary_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Failed to load glossary from {glossary_path}: {str(e)}")
        raise ExtractionError(f"Failed to load glossary: {str(e)}")

def extract_medical_history(file_path: str, genai_client) -> MedicalHistoryData:
    """Extract medical history data from a markdown file using GenAI.
    
    Args:
        file_path (str): Path to the markdown file
        genai_client: Initialized GenAI client
        
    Returns:
        MedicalHistoryData: Structured medical history data
    """
    try:
        # Read markdown file and glossary
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        glossary = load_glossary()
        
        # Prepare prompt for GenAI with explicit instructions
        system_prompt = f"""
            You are a medical data extraction assistant specialized in extracting medical history from Portuguese clinical notes.
            Using this Portuguese medical glossary for reference:
            
            {glossary}
            
            Extract the following information from the patient's medical notes:
            
            1. Previous Medical Conditions (before the current admission):
               - Look for sections labeled "AP:", "Antecedentes Pessoais:", "Antecedentes patológicos", or similar
               - Include full names of conditions and any details about them
               - Common abbreviations: HTA (hipertensão/hypertension), DM (diabetes mellitus), DLP (dislipidemia/dyslipidemia)
            
            2. Previous Surgical History (before the current admission):
               - Look for mentions of "cirurgias prévias", "intervenções cirúrgicas anteriores", etc.
               - Include dates if available
            
            3. Prior Medications (medications the patient was taking before admission):
               - Look for "medicação habitual", "medicação em ambulatório", "medicação prévia" or similar
               - Include dosages and frequency if available
            
            4. Allergies:
               - Look for "alergias", "alergias medicamentosas", etc.
               - Note whether allergies are present, absent, or unknown
            
            Medical notes: {content}
            
            Extract exact terms as they appear in the text, then translate condition names to English.
            Output a valid JSON with the following structure:
            {{
                "medical_conditions": [
                    {{
                        "name": "<condition name in English>",
                        "details": "<additional details or null>"
                    }},
                    ...
                ],
                "surgical_history": [
                    {{
                        "procedure": "<procedure name>",
                        "date": "<date in dd-mm-yyyy format or null>",
                        "details": "<additional details or null>"
                    }},
                    ...
                ],
                "medications": [
                    {{
                        "name": "<medication name>",
                        "dosage": "<dosage or null>",
                        "frequency": "<frequency or null>",
                        "details": "<additional details or null>"
                    }},
                    ...
                ],
                "allergies": "<allergies information or null>",
                "other_notes": "<other relevant medical history notes or null>"
            }}
            
            If any section has no information, return an empty array []. If absolutely no medical history information is found, still return the structure with empty arrays.
        """
        
        console.print(f"\n[bold cyan]Using model:[/bold cyan] {MODEL_NAME}")
        
        # Generate content using the correct API methods
        response = genai_client.models.generate_content(
            model=MODEL_NAME,
            contents=system_prompt,
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
            
            # Format dates if present in surgical history
            if data.get('surgical_history'):
                for surgery in data['surgical_history']:
                    if surgery.get('date'):
                        parts = surgery['date'].split('-')
                        if len(parts) == 3:
                            if len(parts[2]) == 4:  # If year is last (dd-mm-yyyy)
                                surgery['date'] = f"{parts[0]:0>2}-{parts[1]:0>2}-{parts[2]}"
                            elif len(parts[0]) == 4:  # If year is first (yyyy-mm-dd)
                                surgery['date'] = f"{parts[2]:0>2}-{parts[1]:0>2}-{parts[0]}"
            
            # Validate with Pydantic model
            medical_history_data = MedicalHistoryData(**data)
            
            # Log successful extraction
            logging.info(f"Successfully extracted medical history data from {file_path}")
            
            return medical_history_data
            
        except json.JSONDecodeError as e:
            raise ExtractionError(f"Failed to parse GenAI response as JSON: {str(e)}\nResponse was: {json_text}")
        except Exception as e:
            raise ExtractionError(f"Error processing extracted data: {str(e)}")
            
    except Exception as e:
        error_msg = f"Failed to extract medical history from {file_path}: {str(e)}"
        logging.error(error_msg)
        raise ExtractionError(error_msg)

def update_patient_json(patient_id: int, medical_history_data: MedicalHistoryData) -> None:
    """Update patient JSON file with medical history data.
    
    Args:
        patient_id (int): The ID of the patient
        medical_history_data (MedicalHistoryData): The extracted medical history data
        
    This function reads the existing patient JSON file, adds the medical history data
    as a new section, and saves the updated data back to the file.
    """
    try:
        # Construct the JSON file path
        json_path = project_root / f"dados/json/{patient_id}.json"
        
        # Read existing patient data
        with open(json_path, 'r', encoding='utf-8') as f:
            patient_data = json.load(f)
            
        # Convert medical history data to dict and remove None values
        history_dict = {k: v for k, v in medical_history_data.model_dump().items() if v is not None}
        
        # Add medical history data as a nested document
        patient_data['medical_history'] = history_dict
        
        # Write updated data back to file
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(patient_data, f, indent=2, ensure_ascii=False)
            
        print_status(f"✓ Updated patient {patient_id} JSON with medical history data", "success")
        
    except FileNotFoundError:
        error_msg = f"Patient JSON file not found: {json_path}"
        logging.error(error_msg)
        raise ExtractionError(error_msg)
    except Exception as e:
        error_msg = f"Failed to update patient JSON: {str(e)}"
        logging.error(error_msg)
        raise ExtractionError(error_msg)

def process_markdown_file(input_path: str, genai_client) -> MedicalHistoryData:
    """Process a single markdown file and extract medical history data.
    
    Args:
        input_path (str): Path to input markdown file
        genai_client: Initialized GenAI client
    
    Returns:
        MedicalHistoryData: Extracted medical history data
    """
    try:
        print_status(f"Processing: {Path(input_path).name}", "info")
        
        # Extract data
        medical_history_data = extract_medical_history(input_path, genai_client)
        
        # Display extracted data
        console.print("\n[bold green]Extracted Medical History Data:[/bold green]")
        formatted_json = json.dumps(medical_history_data.model_dump(), indent=2, ensure_ascii=False)
        console.print(formatted_json)
        
        # Get patient ID from filename
        patient_id = int(Path(input_path).stem)
        
        # Update patient JSON file
        update_patient_json(patient_id, medical_history_data)
        
        print_status(f"✓ Successfully extracted medical history data from {Path(input_path).name}", "success")
        
        return medical_history_data
        
    except Exception as e:
        print_status(f"Error processing {input_path}: {str(e)}", "error")
        raise

def main():
    """Main execution function."""
    console.print("\n[bold blue]Medical History Data Extractor[/bold blue]")
    console.print("=" * 50 + "\n")
    
    try:
        # Initialize GenAI client
        genai_client = initialize_genai()
        print_status(f"✓ GenAI client initialized successfully", "success")
        
        # Define input file
        input_file = project_root / "dados/md_clean_files/2301.md"
        
        # Process file
        medical_history_data = process_markdown_file(input_file, genai_client)
        
        # Output the result as JSON to terminal
        console.print("\n[bold blue]Medical History JSON Output:[/bold blue]")
        formatted_json = json.dumps(medical_history_data.model_dump(), indent=4, ensure_ascii=False)
        console.print(Panel(formatted_json, title="Medical History Data", expand=False))
        
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