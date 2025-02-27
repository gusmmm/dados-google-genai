"""
Burns Injury Data Extractor
---------------------------
This script uses Pydantic models and Google's Generative AI to extract burn injury data
from medical notes markdown files. It processes the files and extracts structured burn data.
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
from enum import Enum
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

class BurnDepth(str, Enum):
    FIRST_DEGREE = "1st degree"
    SECOND_DEGREE_SUPERFICIAL = "2nd degree superficial"
    SECOND_DEGREE_DEEP = "2nd degree deep"
    THIRD_DEGREE = "3rd degree"
    FOURTH_DEGREE = "4th degree"

class BurnLocation(BaseModel):
    location: str = Field(description="Body part affected")
    degree: BurnDepth = Field(description="Burn depth for this location")
    laterality: Optional[str] = Field(description="Left, right, or bilateral if applicable")
    is_circumferential: Optional[bool] = Field(description="Whether the burn is circumferential")

class FluidAdministration(BaseModel):
    type: str = Field(description="Type of fluid administered")
    volume: str = Field(description="Volume of fluid administered")
    
class Intervention(BaseModel):
    date: str = Field(description="Date of intervention (dd-mm-yyyy)")
    procedure: str = Field(description="Type of procedure")
    details: Optional[str] = Field(description="Additional details about the procedure")

class BurnData(BaseModel):
    injury_date: Optional[str] = Field(default=None)  # dd-mm-yyyy
    injury_time: Optional[str] = Field(default=None)  # HH:MM
    injury_cause: Optional[str] = Field(default=None)
    injury_location: List[str] = Field(default_factory=list)
    burn_degree: List[BurnLocation] = Field(default_factory=list)
    tbsa: Optional[float] = Field(default=None)
    inhalation_injury: bool = Field(default=False)
    pre_hospital_intubation: bool = Field(default=False)
    pre_hospital_fluid: List[FluidAdministration] = Field(default_factory=list)
    pre_hospital_other: Optional[str] = Field(default=None)
    mechanical_ventilation: bool = Field(default=False)
    parkland_formula: Optional[dict] = Field(default=None)
    consultations: List[str] = Field(default_factory=list)
    interventions: List[Intervention] = Field(default_factory=list)

class ExtractionError(Exception):
    """Custom exception for data extraction errors."""
    pass

def load_instructions(file_path: str) -> str:
    """Load instructions from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Failed to load instructions from {file_path}: {str(e)}")
        raise ExtractionError(f"Failed to load instructions: {str(e)}")

def load_glossary() -> str:
    """Load and format the Portuguese medical terms glossary."""
    glossary_path = project_root / GLOSSARY_PATH
    try:
        with open(glossary_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Failed to load glossary from {glossary_path}: {str(e)}")
        raise ExtractionError(f"Failed to load glossary: {str(e)}")

def normalize_burn_degree(degree: str) -> str:
    """Normalize burn degree to match the expected enum values."""
    degree_mapping = {
        "first-degree": "1st degree",
        "first degree": "1st degree",
        "1st-degree": "1st degree",
        "second-degree-superficial": "2nd degree superficial",
        "second degree superficial": "2nd degree superficial",
        "2nd-degree-superficial": "2nd degree superficial",
        "second-degree-deep": "2nd degree deep",
        "second degree deep": "2nd degree deep",
        "2nd-degree-deep": "2nd degree deep",
        "third-degree": "3rd degree",
        "third degree": "3rd degree",
        "3rd-degree": "3rd degree",
        "fourth-degree": "4th degree",
        "fourth degree": "4th degree",
        "4th-degree": "4th degree"
    }
    return degree_mapping.get(degree.lower(), degree)

def extract_burn_data(file_path: str, genai_client) -> BurnData:
    """Extract burn injury data from a markdown file using GenAI."""
    try:
        # Read markdown file and glossary
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        glossary = load_glossary()
        
        # Prepare prompt for GenAI with explicit instructions for handling special values
        system_prompt = f"""
            You are a medical data extraction assistant specialized in burn injuries.
            Using this Portuguese medical glossary for reference:
            
            {glossary}
            
            Extract burn injury information from these Portuguese medical notes, following these strict rules:

            1. Boolean Fields (must be exactly true or false, never text):
               - inhalation_injury: true if explicitly mentioned, false if explicitly denied or not mentioned
               - pre_hospital_intubation: true if mentioned, false if not mentioned
               - mechanical_ventilation: true if mentioned, false if not mentioned
               - is_circumferential: true if explicitly described as circular/circumferential, false otherwise
            
            2. Burn Degree Classification (must use EXACTLY these terms):
               - "1st degree"
               - "2nd degree superficial"
               - "2nd degree deep"
               - "3rd degree"
               - "4th degree"
               For any burn described only as "2nd degree", default to "2nd degree deep" if context suggests depth
               or significant injury, otherwise use "2nd degree superficial"
            
            3. Dates must always be in dd-mm-yyyy format. If no date is available, omit the field entirely
               rather than using null.

            4. For burn locations: if location is bilateral, create separate entries for left and right sides
               with appropriate laterality values.
            
            Medical notes: {content}
            
            Output a valid JSON with the following structure:
            {{
                "injury_date": "<dd-mm-yyyy>",
                "injury_time": "<HH:MM>",
                "injury_cause": "<cause>",
                "injury_location": ["<location1>", "<location2>", ...],
                "burn_degree": [
                    {{
                        "location": "<body part>",
                        "degree": "<EXACTLY one of: '1st degree', '2nd degree superficial', '2nd degree deep', '3rd degree', '4th degree'>",
                        "laterality": "<left/right or null>",
                        "is_circumferential": true/false
                    }},
                    ...
                ],
                "tbsa": <percentage as float>,
                "inhalation_injury": true/false,
                "pre_hospital_intubation": true/false,
                "pre_hospital_fluid": [
                    {{
                        "type": "<fluid type>",
                        "volume": "<volume with units>"
                    }},
                    ...
                ],
                "pre_hospital_other": "<other pre-hospital info or null>",
                "mechanical_ventilation": true/false,
                "parkland_formula": {{
                    "weight": <weight in kg>,
                    "percentage": <burn percentage>,
                    "total_volume": "<total volume in ml>",
                    "first_8h": "<volume for first 8 hours in ml>",
                    "next_16h": "<volume for next 16 hours in ml>"
                }},
                "consultations": ["<specialty1>", "<specialty2>", ...],
                "interventions": [
                    {{
                        "date": "<dd-mm-yyyy>",
                        "procedure": "<procedure name>",
                        "details": "<details or null>"
                    }},
                    ...
                ]
            }}
            
            Remember:
            1. ALL boolean fields must be exactly true or false, never strings
            2. Burn degrees must match EXACTLY one of the specified values
            3. Dates must be in dd-mm-yyyy format or omitted entirely
            4. Never use 'unknown' or 'unspecified' as values
            5. For missing or unclear data, omit the field rather than using null/unknown
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
            
            # Additional validation and normalization
            if data.get('burn_degree'):
                for burn in data['burn_degree']:
                    # Ensure is_circumferential is boolean
                    if 'is_circumferential' in burn:
                        if isinstance(burn['is_circumferential'], str):
                            burn['is_circumferential'] = burn['is_circumferential'].lower() == 'true'
                    else:
                        burn['is_circumferential'] = False
                        
                    # Handle basic 2nd degree classifications
                    if burn.get('degree') == '2nd degree':
                        burn['degree'] = '2nd degree deep'  # Default to deep as safer option
            
            # Ensure boolean fields are actually boolean
            boolean_fields = ['inhalation_injury', 'pre_hospital_intubation', 'mechanical_ventilation']
            for field in boolean_fields:
                if field in data:
                    if isinstance(data[field], str):
                        data[field] = data[field].lower() == 'true'
                else:
                    data[field] = False
            
            # Format dates
            if data.get('injury_date'):
                parts = data['injury_date'].split('-')
                if len(parts) == 3:
                    if len(parts[2]) == 4:  # If year is last (dd-mm-yyyy)
                        data['injury_date'] = f"{parts[0]:0>2}-{parts[1]:0>2}-{parts[2]}"
                    elif len(parts[0]) == 4:  # If year is first (yyyy-mm-dd)
                        data['injury_date'] = f"{parts[2]:0>2}-{parts[1]:0>2}-{parts[0]}"
            
            # Format intervention dates
            if data.get('interventions'):
                filtered_interventions = []
                for intervention in data['interventions']:
                    if intervention.get('date'):
                        parts = intervention['date'].split('-')
                        if len(parts) == 3:
                            if len(parts[2]) == 4:  # If year is last
                                intervention['date'] = f"{parts[0]:0>2}-{parts[1]:0>2}-{parts[2]}"
                            elif len(parts[0]) == 4:  # If year is first
                                intervention['date'] = f"{parts[2]:0>2}-{parts[1]:0>2}-{parts[0]}"
                        filtered_interventions.append(intervention)
                    else:
                        # Skip interventions without dates instead of including null dates
                        continue
                data['interventions'] = filtered_interventions
            
            # Validate with Pydantic model
            burn_data = BurnData(**data)
            
            # Log successful extraction
            logging.info(f"Successfully extracted burn data from {file_path}")
            
            return burn_data
            
        except json.JSONDecodeError as e:
            raise ExtractionError(f"Failed to parse GenAI response as JSON: {str(e)}\nResponse was: {json_text}")
        except Exception as e:
            raise ExtractionError(f"Error processing extracted data: {str(e)}")
            
    except Exception as e:
        error_msg = f"Failed to extract data from {file_path}: {str(e)}"
        logging.error(error_msg)
        raise ExtractionError(error_msg)

def update_patient_json(patient_id: int, burn_data: BurnData) -> None:
    """Update patient JSON file with burn data.
    
    Args:
        patient_id (int): The ID of the patient
        burn_data (BurnData): The extracted burn data to add
        
    This function reads the existing patient JSON file, adds the burn data
    as a new section, and saves the updated data back to the file.
    """
    try:
        # Construct the JSON file path
        json_path = project_root / f"dados/json/{patient_id}.json"
        
        # Read existing patient data
        with open(json_path, 'r', encoding='utf-8') as f:
            patient_data = json.load(f)
            
        # Convert burn data to dict and remove None values
        burn_dict = {k: v for k, v in burn_data.model_dump().items() if v is not None}
        
        # Add burn data as a nested document
        patient_data['burns_data'] = burn_dict
        
        # Write updated data back to file
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(patient_data, f, indent=2, ensure_ascii=False)
            
        print_status(f"✓ Updated patient {patient_id} JSON with burn data", "success")
        
    except FileNotFoundError:
        error_msg = f"Patient JSON file not found: {json_path}"
        logging.error(error_msg)
        raise ExtractionError(error_msg)
    except Exception as e:
        error_msg = f"Failed to update patient JSON: {str(e)}"
        logging.error(error_msg)
        raise ExtractionError(error_msg)

def process_markdown_file(input_path: str, genai_client) -> BurnData:
    """Process a single markdown file and extract burn data.
    
    Args:
        input_path (str): Path to input markdown file
        genai_client: Initialized GenAI client
    
    Returns:
        BurnData: Extracted burn data
    """
    try:
        print_status(f"Processing: {Path(input_path).name}", "info")
        
        # Extract data
        burn_data = extract_burn_data(input_path, genai_client)
        
        # Display extracted data
        console.print("\n[bold green]Extracted Burn Data:[/bold green]")
        formatted_json = json.dumps(burn_data.model_dump(), indent=2, ensure_ascii=False)
        console.print(formatted_json)
        
        # Get patient ID from filename
        patient_id = int(Path(input_path).stem)
        
        # Update patient JSON file
        update_patient_json(patient_id, burn_data)
        
        print_status(f"✓ Successfully extracted burn data from {Path(input_path).name}", "success")
        
        return burn_data
        
    except Exception as e:
        print_status(f"Error processing {input_path}: {str(e)}", "error")
        raise

def main():
    """Main execution function."""
    console.print("\n[bold blue]Burns Data Extractor[/bold blue]")
    console.print("=" * 50 + "\n")
    
    try:
        # Initialize GenAI client
        genai_client = initialize_genai()
        
        # Define input file
        input_file = project_root / "dados/md_clean_files/2301.md"
        
        # Process file
        burn_data = process_markdown_file(input_file, genai_client)
        
        # Output the result as JSON to terminal
        console.print("\n[bold blue]Burns Data JSON Output:[/bold blue]")
        formatted_json = json.dumps(burn_data.model_dump(), indent=4, ensure_ascii=False)
        console.print(Panel(formatted_json, title="Burns JSON Data", expand=False))
        
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