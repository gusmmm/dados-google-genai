# dados-google-genai
Gestão de dados - versão com google genai + mongoDB + fastAPI e React

"""
Medical Notes Processing with Google Generative AI
-----------------------------------------------

This project processes medical notes using Google's Generative AI to extract and structure information.

## Setup

1. Install dependencies:
```bash
uv pip install -r requirements.txt
```

2. Environment Configuration:
- Copy `.env.example` to `.env`
- Add your Gemini API key to `.env`:
```
GEMINI_API_KEY=your_api_key_here
```

## Project Structure

- `dados/`: Contains original PDFs and processed markdown files
- `extractors/`: Pydantic models for data extraction
- `genai/`: Google Generative AI configuration and helpers
- `pdf_manager/`: PDF to Markdown conversion utilities

## Usage

1. Convert PDFs to Markdown:
```bash
python -m pdf_manager.pdf_to_md
```

2. Clean and structure markdown files:
```bash
python -m pdf_manager.md_clean
```

3. Extract structured data:
```bash
python -m extractors.classes_pydantic
```

## Error Handling

All errors are logged to:
- `pdf_processing.log`: PDF conversion errors
- `md_improvement.log`: Markdown cleaning errors
- `genai.log`: Google AI API errors

## License

[Your License]
"""
