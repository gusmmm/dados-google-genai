"""
Markdown Cleaner for Medical Notes
--------------------------------

This script cleans and improves markdown files generated from PDF conversions.
It handles:
- Removing duplicate lines
- Cleaning whitespace
- Preserving section markers
- Maintaining document structure

Dependencies:
- rich: For beautiful terminal output
- pathlib: For cross-platform path handling
- logging: For error tracking and debugging

Author: [Your Name]
Version: 1.0
"""

import os
from pathlib import Path
import re
import logging
from typing import List, Dict, Pattern
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Initialize rich console for beautiful terminal output
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('md_improvement.log'),
        logging.StreamHandler()
    ]
)

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

class MarkdownCleaningError(Exception):
    """Custom exception for markdown cleaning errors."""
    pass

class MDCleaner:
    def __init__(self):
        """Initialize cleaner with section marker patterns"""
        self.section_markers: Dict[str, Pattern] = {
            "start": re.compile(r"^(>>|═+)\s*(?:START|start)\s+.*\s*(?:NOTE|note)\s*(<<|═+)\s*$"),
            "end": re.compile(r"^(>>|═+)\s*(?:END|end)\s+.*\s*(?:NOTE|note)\s*(<<|═+)\s*$")
        }
        
        self.section_styles = {
            "admission": {
                "top": "╔═══════════════════ ADMISSION NOTE START ═══════════════════╗",
                "bottom": "╚════════════════════ ADMISSION NOTE END ════════════════════╝"
            },
            "release": {
                "top": "╔═══════════════════ RELEASE NOTE START ════════════════════╗",
                "bottom": "╚════════════════════ RELEASE NOTE END ═════════════════════╝"
            },
            "provisory - death report": {
                "top": "╔════════════ PROVISORY DEATH REPORT START ═════════════╗",
                "bottom": "╚════════════ PROVISORY DEATH REPORT END ═══════════════╝"
            },
            "final - death report": {
                "top": "╔═════════════ FINAL DEATH REPORT START ══════════════╗",
                "bottom": "╚═════════════ FINAL DEATH REPORT END ════════════════╝"
            }
        }
        
        self.header_marker = "╔══════════════════════════════════════════════════════════════╗"
        self.footer_marker = "╚══════════════════════════════════════════════════════════════╝"
        self.metadata_marker = "║"

    def _format_section_header(self, section_name: str) -> str:
        """Format a section header with clear boundaries.
        
        Args:
            section_name (str): Name of the section
            
        Returns:
            str: Formatted section header
        """
        return f"{self.header_marker}\n║ SECTION: {section_name.upper()} ║\n{self.footer_marker}"

    def _is_section_marker(self, line: str) -> bool:
        """Check if line is a section marker."""
        return any(pattern.match(line.strip())
                for pattern in self.section_markers.values())

    def _clean_line(self, line: str) -> str:
        """Clean individual line and improve section markers."""
        line = line.strip()
        
        # Handle section headers (## style markers)
        if line.startswith("## "):
            section_type = line.lower().replace("## ", "").replace(" note", "").strip()
            if section_type in self.section_styles:
                return f"\n{self.section_styles[section_type]['top']}"
            
        # Handle section content
        if line.startswith("---"):
            return ""  # Remove old separators
        if line == "```":
            return ""  # Remove old code block markers
            
        return ' '.join(line.split())

    def _remove_duplicates(self, lines: List[str]) -> List[str]:
        """Remove duplicate lines while preserving section markers."""
        seen_lines = set()
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            # Always keep section markers and formatted headers/footers
            if (self._is_section_marker(line) or 
                line.startswith("╔") or 
                line.startswith("╚") or 
                line.startswith("║")):
                cleaned_lines.append(line)
                continue
                
            if line not in seen_lines:
                seen_lines.add(line)
                cleaned_lines.append(line)
        
        return cleaned_lines

    def clean_file(self, input_path: str, output_path: str) -> None:
        """Clean and improve a single markdown file."""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Add file metadata header
            file_name = os.path.basename(input_path)
            metadata_header = [
                self.header_marker,
                f"{self.metadata_marker} DOCUMENT ID: {file_name[:-3]}",
                f"{self.metadata_marker} DOCUMENT TYPE: MEDICAL NOTES",
                f"{self.metadata_marker} FORMAT: MARKDOWN",
                f"{self.metadata_marker} STATUS: PROCESSED AND STRUCTURED",
                self.footer_marker,
                "",
                "# Patient Medical Record",
                ""
            ]

            # Process content
            lines = content.split('\n')
            current_section = None
            cleaned_lines = []
            
            for line in lines:
                clean_line = self._clean_line(line)
                if not clean_line:
                    continue
                    
                # Check for section transitions
                if clean_line.startswith("╔═══") and "START" in clean_line:
                    current_section = next(
                        (k for k, v in self.section_styles.items() 
                         if v["top"] == clean_line), None)
                    cleaned_lines.append(clean_line)
                elif current_section and clean_line.strip() and not clean_line.startswith("╔") and not clean_line.startswith("╚"):
                    cleaned_lines.append(clean_line)
                elif current_section and (clean_line.startswith("## ") or clean_line.startswith("#")):
                    # Add section end marker before starting new section
                    cleaned_lines.append(self.section_styles[current_section]["bottom"])
                    current_section = None
                    if not clean_line.startswith("# Patient"):  # Skip main title
                        cleaned_lines.append(clean_line)
                else:
                    cleaned_lines.append(clean_line)
            
            # Add final section end marker if needed
            if current_section:
                cleaned_lines.append(self.section_styles[current_section]["bottom"])
            
            # Remove duplicates while preserving markers
            final_lines = self._remove_duplicates(cleaned_lines)
            
            # Combine metadata and content
            full_content = metadata_header + final_lines
            
            # Write improved content
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(full_content))
            
            logging.info(f"Successfully cleaned {input_path}")
            print_status(f"✓ Cleaned: {os.path.basename(input_path)}", "success")
            
        except Exception as e:
            error_msg = f"Error processing {input_path}: {str(e)}"
            logging.error(error_msg)
            raise MarkdownCleaningError(error_msg)

def process_directory(input_dir: str, output_dir: str) -> None:
    """Process all markdown files in a directory.
    
    Args:
        input_dir (str): Directory containing markdown files to process
        output_dir (str): Directory to save cleaned markdown files
    """
    try:
        cleaner = MDCleaner()
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print_status(f"✓ Output folder ready: {output_dir}", "success")
        
        # Get markdown files
        md_files = [f for f in os.listdir(input_dir) if f.endswith('.md')]
        print_status(f"Found {len(md_files)} markdown files to process", "info")
        
        # Process each markdown file with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Cleaning markdown files...", total=len(md_files))
            
            for filename in md_files:
                try:
                    input_path = os.path.join(input_dir, filename)
                    # Use same filename in output directory since they are in different folders
                    output_path = os.path.join(output_dir, filename)
                    cleaner.clean_file(input_path, output_path)
                    progress.update(task, advance=1)
                    
                except MarkdownCleaningError as e:
                    print_status(str(e), "error")
                    continue
        
        print_status("Markdown cleaning completed successfully!", "success")
        
    except Exception as e:
        error_msg = f"Critical error: {str(e)}"
        logging.error(error_msg)
        print_status(error_msg, "error")
        raise

if __name__ == "__main__":
    console.print("\n[bold blue]Markdown Cleaner for Medical Notes[/bold blue]")
    console.print("=" * 50 + "\n")
    
    input_dir = "./dados/md_files"
    output_dir = "./dados/md_clean_files"
    
    try:
        process_directory(input_dir, output_dir)
    except Exception:
        print_status("Cleaning failed! Check the logs for details.", "error")
        exit(1)


