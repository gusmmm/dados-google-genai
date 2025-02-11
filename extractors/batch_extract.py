"""
Batch Medical Notes Data Extractor
--------------------------------

This script processes multiple medical note files in parallel and extracts
structured data using the Google GenAI API.
"""

import sys
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.table import Table

from genai.genai_config import initialize_genai, GenAIConfigError
from config import *
from extractors.extract_data import process_markdown_file, ExtractionError

# Initialize rich console
console = Console()

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

def get_markdown_files(max_files: Optional[int] = None) -> List[Path]:
    """Get markdown files from the input directory.
    
    Args:
        max_files: Optional maximum number of files to process
        
    Returns:
        List of Path objects for markdown files
    """
    files = list(INPUT_DIR.glob('*.md'))
    if max_files is not None:
        files = files[:max_files]
    return files

def file_already_processed(file_path: Path) -> bool:
    """Check if a markdown file has already been processed.
    
    Args:
        file_path: Path to the markdown file
        
    Returns:
        True if corresponding JSON exists, False otherwise
    """
    json_path = OUTPUT_DIR / f"{file_path.stem}.json"
    return json_path.exists()

def process_file_with_progress(
    file_path: Path,
    genai_client: Any,
    progress: Progress,
    task_id: TaskID
) -> Dict[str, Any]:
    """Process a single file with progress tracking."""
    try:
        if file_already_processed(file_path):
            console.print(f"[yellow]Skipping {file_path.name} - JSON already exists[/yellow]")
            return {"file": file_path.name, "status": "skipped", "error": None}
            
        process_markdown_file(str(file_path), str(OUTPUT_DIR), genai_client)
        return {"file": file_path.name, "status": "success", "error": None}
    except Exception as e:
        error_msg = f"Error processing {file_path.name}: {str(e)}"
        logging.error(error_msg)
        return {"file": file_path.name, "status": "failed", "error": str(e)}
    finally:
        progress.update(task_id, advance=1)

def process_files_in_parallel(files: List[Path], max_workers: int = MAX_WORKERS) -> List[Dict[str, Any]]:
    """Process multiple files in parallel with progress tracking."""
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

def display_results(results: List[Dict[str, Any]]) -> None:
    """Display processing results in a formatted table."""
    table = Table(title="Processing Results")
    table.add_column("File", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Error", style="red")
    
    successful = 0
    failed = 0
    skipped = 0
    
    for result in results:
        status_style = {
            "success": "green",
            "failed": "red",
            "skipped": "yellow"
        }.get(result["status"], "white")
        
        table.add_row(
            result["file"],
            f"[{status_style}]{result['status']}[/{status_style}]",
            str(result["error"]) if result["error"] else ""
        )
        
        if result["status"] == "success":
            successful += 1
        elif result["status"] == "failed":
            failed += 1
        else:
            skipped += 1
    
    console.print(table)
    console.print(f"\nSummary: {successful} successful, {failed} failed, {skipped} skipped")

def main():
    """Main execution function."""
    console.print("\n[bold blue]Batch Medical Notes Data Extractor[/bold blue]")
    console.print("=" * 50 + "\n")
    
    try:
        # Setup logging
        setup_logging()
        
        # Show max files configuration
        if MAX_FILES:
            console.print(f"[cyan]Processing up to {MAX_FILES} files[/cyan]\n")
        else:
            console.print("[cyan]Processing all files[/cyan]\n")
        
        # Ensure output directory exists
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Get markdown files
        files = get_markdown_files(MAX_FILES)
        if not files:
            console.print("[yellow]No markdown files found in input directory[/yellow]")
            return
        
        console.print(f"Found {len(files)} files to process\n")
        
        # Process files
        results = process_files_in_parallel(files)
        
        # Display results
        display_results(results)
        
    except Exception as e:
        console.print(f"[red]Fatal error: {str(e)}[/red]")
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
