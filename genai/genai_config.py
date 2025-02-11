"""
Google Generative AI Configuration
--------------------------------

This module handles the configuration and initialization of the Google Generative AI client.
It loads the API key from environment variables and provides error handling and logging.

Dependencies:
- google-genai: For accessing Google's Generative AI models
- python-dotenv: For environment variable management
- rich: For beautiful terminal output
- logging: For error tracking and debugging
"""

import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from google import genai
from google.genai import types

# Initialize rich console for beautiful terminal output
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('genai.log'),
        logging.StreamHandler()
    ]
)

class GenAIConfigError(Exception):
    """Custom exception for GenAI configuration errors."""
    pass

def print_status(message: str, style: str = "info") -> None:
    """Print beautifully formatted status messages to the terminal."""
    styles = {
        "info": "blue",
        "error": "red",
        "success": "green"
    }
    console.print(Panel(message, style=styles.get(style, "white")))

def initialize_genai() -> genai.Client:
    """Initialize the Google Generative AI client with API key from environment.
    
    Returns:
        genai.Client: Configured GenAI client
        
    Raises:
        GenAIConfigError: If API key is missing or configuration fails
    """
    try:
        # Load environment variables from .env file
        env_path = Path(__file__).parent.parent / '.env'
        if not env_path.exists():
            raise GenAIConfigError(f"Environment file not found at {env_path}")
        
        load_dotenv(env_path)
        
        # Get API key from environment
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise GenAIConfigError("GEMINI_API_KEY not found in environment variables")
            
        logging.info("Initializing Google Generative AI client")
        client = genai.Client(api_key=api_key)
        print_status("✓ GenAI client initialized successfully", "success")
        return client
        
    except Exception as e:
        error_msg = f"Failed to initialize GenAI client: {str(e)}"
        logging.error(error_msg)
        print_status(error_msg, "error")
        raise GenAIConfigError(error_msg)

def test_connection(client: genai.Client) -> bool:
    """Test the GenAI connection with a simple query.
    
    Args:
        client (genai.Client): The initialized GenAI client
        
    Returns:
        bool: True if test succeeded, False otherwise
    """
    try:
        logging.info("Testing GenAI connection")
        response = client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents='Test connection'
        )
        print_status("✓ GenAI connection test successful", "success")
        return True
        
    except Exception as e:
        error_msg = f"GenAI connection test failed: {str(e)}"
        logging.error(error_msg)
        print_status(error_msg, "error")
        return False

if __name__ == "__main__":
    console.print("\n[bold blue]Google Generative AI Configuration[/bold blue]")
    console.print("=" * 50 + "\n")
    
    try:
        client = initialize_genai()
        if not test_connection(client):
            sys.exit(1)
    except GenAIConfigError:
        sys.exit(1)
