"""
Enhanced Batch Data Extractor
--------------------------------
This script processes multiple medical note files and extracts structured data
using a sequence of three extractors:

1. Patient Data Extractor: Basic patient demographics and information
2. Burns Data Extractor: Burn injury specific information
3. Medical History Extractor: Previous conditions, surgeries, and medications

Features:
- Sequential processing with dependency checks
- Skip data that's already been extracted
- Process all files or a specific range
- Parallel processing with progress tracking
- Rich terminal output and detailed logging
"""
import sys
import json
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum, auto

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.panel import Panel
from rich.table import Table

from genai.genai_config import initialize_genai, GenAIConfigError, print_status
from config import *

# Import extractors
from extractors.extract_data import extract_patient_data, ExtractionError as PatientExtractionError
from extractors.burns_extract import extract_burn_data
from extractors.medical_history_extract import extract_medical_history

# Initialize rich console
console = Console()

class ExtractorType(Enum):
    """Enum for different extractor types"""
    PATIENT = auto()
    BURNS = auto()
    MEDICAL_HISTORY = auto()
    ALL = auto()
    
    @classmethod
    def from_string(cls, value: str) -> 'ExtractorType':
        """Convert string to extractor type"""
        mapping = {
            "patient": cls.PATIENT,
            "burns": cls.BURNS,
            "medical_history": cls.MEDICAL_HISTORY,
            "all": cls.ALL
        }
        
        if value.lower() not in mapping:
            raise ValueError(f"Invalid extractor type: {value}. Valid options are: {list(mapping.keys())}")
            
        return mapping[value.lower()]

def setup_logging() -> None:
    """Configure logging with both file and console handlers."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler('batch_extraction.log'),
            logging.StreamHandler()
        ]
    )

def get_markdown_files(file_range: Optional[Tuple[int, int]] = None) -> List[Path]:
    """Get markdown files from the input directory within a specific range.
    
    Args:
        file_range: Optional tuple of (start_id, end_id) to limit files by ID range
        
    Returns:
        List of Path objects for markdown files
    """
    files = list(INPUT_DIR.glob('*.md'))
    
    if file_range:
        start_id, end_id = file_range
        filtered_files = []
        
        for file_path in files:
            try:
                file_id = int(file_path.stem)
                if start_id <= file_id <= end_id:
                    filtered_files.append(file_path)
            except ValueError:
                # Skip files that don't have numeric names
                pass
                
        return filtered_files
    
    return files

def field_exists_in_json(json_path: Path, field_name: str) -> bool:
    """Check if a specific field exists in a JSON file.
    
    Args:
        json_path: Path to the JSON file
        field_name: Name of the field to check
        
    Returns:
        True if field exists, False otherwise
    """
    import json
    
    if not json_path.exists():
        return False
        
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return field_name in data
    except Exception as e:
        logging.warning(f"Error checking JSON file {json_path}: {str(e)}")
        return False

def should_extract(file_path: Path, extractor_type: ExtractorType) -> bool:
    """Check if extraction should be performed for a file.
    
    Args:
        file_path: Path to the markdown file
        extractor_type: Type of extractor to check
        
    Returns:
        True if extraction should be performed, False otherwise
    """
    json_path = project_root / "dados/json" / f"{file_path.stem}.json"
    
    if not json_path.exists():
        return True
        
    # Check for field based on extractor type
    if extractor_type == ExtractorType.PATIENT:
        # For patient data, basic fields should exist
        return not field_exists_in_json(json_path, "id_patient")
    elif extractor_type == ExtractorType.BURNS:
        return not field_exists_in_json(json_path, "burns_data")
    elif extractor_type == ExtractorType.MEDICAL_HISTORY:
        return not field_exists_in_json(json_path, "medical_history")
    elif extractor_type == ExtractorType.ALL:
        # For ALL, check if any extractor should run
        return (not field_exists_in_json(json_path, "id_patient") or
                not field_exists_in_json(json_path, "burns_data") or
                not field_exists_in_json(json_path, "medical_history"))
                
    return True

def update_patient_json(patient_id: int, data: Dict[str, Any], field_name: str) -> None:
    """Update patient JSON file with extracted data.
    
    Args:
        patient_id: Patient ID
        data: Data to add to JSON
        field_name: Field name to use in the JSON
    """
    import json
    
    json_path = project_root / f"dados/json/{patient_id}.json"
    
    try:
        if json_path.exists():
            # Read existing data
            with open(json_path, 'r', encoding='utf-8') as f:
                patient_data = json.load(f)
        else:
            # Create new data structure
            patient_data = {}
        
        # Add or update the specified field
        patient_data[field_name] = data
        
        # Write updated data back to file
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(patient_data, f, indent=2, ensure_ascii=False)
            
        logging.info(f"Updated patient {patient_id} JSON with {field_name} data")
        
    except Exception as e:
        error_msg = f"Failed to update patient JSON: {str(e)}"
        logging.error(error_msg)
        raise Exception(error_msg)

def extract_and_update(file_path: Path, extractor_type: ExtractorType, genai_client: Any) -> Dict[str, Any]:
    """Extract data using the specified extractor and update the JSON file.
    The extractors are run in a specific sequence: patient data -> burns data -> medical history.
    Each extractor checks if its data already exists before running.
    
    Args:
        file_path: Path to the markdown file
        extractor_type: Type of extractor to use
        genai_client: Initialized GenAI client
        
    Returns:
        Dictionary with extraction results
    """
    patient_id = int(file_path.stem)
    results = {}
    json_path = project_root / f"dados/json/{patient_id}.json"
    
    try:
        # First check if JSON file exists for patient data extraction
        if extractor_type in [ExtractorType.PATIENT, ExtractorType.ALL]:
            if not json_path.exists():
                # Extract and create initial patient data
                patient_data = extract_patient_data(str(file_path), genai_client)
                json_dict = patient_data.model_dump()
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_dict, f, indent=2, ensure_ascii=False)
                results["patient"] = "extracted"
                logging.info(f"Created new patient JSON for {patient_id}")
            else:
                results["patient"] = "skipped"
                logging.info(f"Patient JSON already exists for {patient_id}")
                
        # Then check and extract burns data if needed
        if extractor_type in [ExtractorType.BURNS, ExtractorType.ALL]:
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    patient_data = json.load(f)
                    
                if 'burns_data' not in patient_data:
                    burns_data = extract_burn_data(str(file_path), genai_client)
                    patient_data['burns_data'] = burns_data.model_dump()
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(patient_data, f, indent=2, ensure_ascii=False)
                    results["burns"] = "extracted"
                    logging.info(f"Added burns data to patient {patient_id}")
                else:
                    results["burns"] = "skipped"
                    logging.info(f"Burns data already exists for patient {patient_id}")
                    
        # Finally check and extract medical history if needed
        if extractor_type in [ExtractorType.MEDICAL_HISTORY, ExtractorType.ALL]:
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    patient_data = json.load(f)
                    
                if 'medical_history' not in patient_data:
                    history_data = extract_medical_history(str(file_path), genai_client)
                    patient_data['medical_history'] = history_data.model_dump()
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump(patient_data, f, indent=2, ensure_ascii=False)
                    results["medical_history"] = "extracted"
                    logging.info(f"Added medical history to patient {patient_id}")
                else:
                    results["medical_history"] = "skipped"
                    logging.info(f"Medical history already exists for patient {patient_id}")
                    
        results["status"] = "success"
        results["error"] = None
        
    except Exception as e:
        logging.error(f"Error processing {file_path.name}: {str(e)}")
        results["status"] = "failed"
        results["error"] = str(e)
        
    return {
        "file": file_path.name,
        "patient_id": patient_id,
        "results": results
    }

def process_file_with_progress(
    file_path: Path,
    extractor_type: ExtractorType,
    genai_client: Any,
    progress: Progress,
    task_id: TaskID
) -> Dict[str, Any]:
    """Process a single file with progress tracking."""
    try:
        print_status(f"Processing: {file_path.name}", "info")
        result = extract_and_update(file_path, extractor_type, genai_client)
        return result
    finally:
        progress.update(task_id, advance=1)

def process_files_in_parallel(
    files: List[Path],
    extractor_type: ExtractorType,
    max_workers: int = MAX_WORKERS
) -> List[Dict[str, Any]]:
    """Process multiple files in parallel with progress tracking.
    
    The extractors run in a specific sequence for each file:
    1. Patient Data (if needed)
    2. Burns Data (if patient data exists)
    3. Medical History (if patient data exists)
    
    Files are processed in parallel, but the extractors for each file
    run sequentially to maintain data consistency.
    """
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("[cyan]Processing files...", total=len(files))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(
                    process_file_with_progress, 
                    file_path, 
                    extractor_type,
                    initialize_genai(), 
                    progress,
                    task
                ): file_path 
                for file_path in files
            }
            
            for future in as_completed(future_to_file):
                result = future.result()
                results.append(result)
    
    return results

def display_results(results: List[Dict[str, Any]], extractor_type: ExtractorType) -> None:
    """Display processing results in a formatted table."""
    table = Table(title="Processing Results")
    table.add_column("File", style="cyan")
    table.add_column("Status", style="green")
    
    if extractor_type == ExtractorType.PATIENT or extractor_type == ExtractorType.ALL:
        table.add_column("Patient Data", style="blue")
    if extractor_type == ExtractorType.BURNS or extractor_type == ExtractorType.ALL:
        table.add_column("Burns Data", style="yellow")
    if extractor_type == ExtractorType.MEDICAL_HISTORY or extractor_type == ExtractorType.ALL:
        table.add_column("Medical History", style="magenta")
    
    table.add_column("Error", style="red")
    
    successful = 0
    failed = 0
    
    for result in results:
        status_style = "green" if result.get("results", {}).get("status") == "success" else "red"
        
        row = [
            result["file"],
            f"[{status_style}]{result.get('results', {}).get('status', 'unknown')}[/{status_style}]",
        ]
        
        # Add extractor-specific columns
        if extractor_type == ExtractorType.PATIENT or extractor_type == ExtractorType.ALL:
            patient_result = result.get("results", {}).get("patient", "not processed")
            row.append(patient_result)
            
        if extractor_type == ExtractorType.BURNS or extractor_type == ExtractorType.ALL:
            burns_result = result.get("results", {}).get("burns", "not processed")
            row.append(burns_result)
            
        if extractor_type == ExtractorType.MEDICAL_HISTORY or extractor_type == ExtractorType.ALL:
            history_result = result.get("results", {}).get("medical_history", "not processed")
            row.append(history_result)
        
        # Add error column
        row.append(str(result.get("results", {}).get("error", "")) or "")
        
        table.add_row(*row)
        
        if result.get("results", {}).get("status") == "success":
            successful += 1
        else:
            failed += 1
    
    console.print(table)
    console.print(f"\nSummary: {successful} successful, {failed} failed")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enhanced Batch Data Extractor")
    
    parser.add_argument(
        "--extractor",
        type=str,
        choices=["patient", "burns", "medical_history", "all"],
        default="all",
        help="Type of extractor to use (patient, burns, medical_history, or all)"
    )
    
    parser.add_argument(
        "--start",
        type=int,
        help="Starting patient ID"
    )
    
    parser.add_argument(
        "--end",
        type=int,
        help="Ending patient ID"
    )
    
    parser.add_argument(
        "--file",
        type=str,
        help="Process a specific file by name or ID"
    )
    
    args = parser.parse_args()
    
    # Handle file range logic
    file_range = None
    if args.start is not None and args.end is not None:
        file_range = (args.start, args.end)
    elif args.start is not None:
        file_range = (args.start, args.start)
    elif args.end is not None:
        file_range = (2301, args.end)  # Assuming 2301 is the first patient ID
        
    # Handle specific file
    specific_file = None
    if args.file:
        try:
            # If numeric, interpret as patient ID
            specific_file = int(args.file)
            file_range = (specific_file, specific_file)
        except ValueError:
            # If not numeric, interpret as filename
            specific_file = args.file
    
    return args.extractor, file_range, specific_file

def main():
    """Main execution function."""
    console.print("\n[bold blue]Enhanced Batch Data Extractor[/bold blue]")
    console.print("=" * 50 + "\n")
    
    try:
        # Parse arguments
        extractor_name, file_range, specific_file = parse_arguments()
        extractor_type = ExtractorType.from_string(extractor_name)
        
        # Setup logging
        setup_logging()
        
        # Show configuration
        if file_range:
            console.print(f"[cyan]Processing patient IDs {file_range[0]} to {file_range[1]}[/cyan]\n")
        elif specific_file:
            console.print(f"[cyan]Processing specific file: {specific_file}[/cyan]\n")
        else:
            console.print("[cyan]Processing all files[/cyan]\n")
            
        console.print(f"[cyan]Using extractor: {extractor_name}[/cyan]\n")
        
        # Ensure output directory exists
        (project_root / "dados/json").mkdir(parents=True, exist_ok=True)
        
        # Get markdown files
        if not specific_file or isinstance(specific_file, int):
            files = get_markdown_files(file_range)
        else:
            # Handle specific filename
            specific_path = INPUT_DIR / specific_file
            files = [specific_path] if specific_path.exists() else []
            
        if not files:
            console.print("[yellow]No markdown files found matching criteria[/yellow]")
            return
        
        console.print(f"Found {len(files)} files to process\n")
        
        # Process files
        results = process_files_in_parallel(files, extractor_type)
        
        # Display results
        display_results(results, extractor_type)
        
    except Exception as e:
        console.print(f"[bold red]Fatal error: {str(e)}[/bold red]")
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()