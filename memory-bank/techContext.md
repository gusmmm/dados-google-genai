# Technical Context

## Technology Stack

### Backend
- Python
- FastAPI (web framework)
- Uvicorn (ASGI server)
- Motor (async MongoDB driver)
- Pydantic (data validation)
- Google GenAI
- PDF processing libraries

### Frontend
- React
- Vite (build tool)
- React Router
- Axios/Fetch (API communication)
- Data visualization libraries (TBD)

### Database
- MongoDB

## Project Structure
```
├── pdf_manager/          # PDF processing
│   ├── pdf_to_md.py     # PDF to markdown conversion
│   └── md_clean.py      # Markdown cleaning
├── extractors/           # Data extraction
│   ├── extract_data.py  # Core extraction logic
│   ├── batch_extract.py # Batch processing
│   └── classes_pydantic.py # Data models
├── genai/               # AI integration
│   └── genai_config.py  # AI configuration
├── backend/             # FastAPI application
│   ├── models/         # Pydantic models
│   ├── routes/         # API endpoints
│   ├── services/       # Business logic
│   └── database.py     # MongoDB connection
├── frontend/           # React application
│   ├── src/
│   │   ├── components/ # React components
│   │   ├── services/   # API services
│   │   ├── pages/      # Route pages
│   │   └── App.tsx     # Root component
│   └── package.json    # Frontend dependencies
└── config.py           # Global configuration
```

## Dependencies

### Backend Dependencies
- fastapi
- uvicorn
- motor
- pydantic
- google-generativeai
- python-multipart
- pymongo
- python-jose[cryptography]
- passlib[bcrypt]

### Frontend Dependencies
- react
- react-dom
- react-router-dom
- axios
- (data visualization libraries TBD)

### Development Tools
- uv (dependency management)
- Python version managed via .python-version
- Node.js and npm/yarn for frontend
- MongoDB Community Edition

## Development Setup

### Backend Setup
1. Python environment configuration
2. MongoDB installation and setup
3. Google GenAI credentials
4. Environment variables configuration
   - MongoDB URI
   - API keys
   - JWT secrets

### Frontend Setup
1. Node.js installation
2. Package installation via npm/yarn
3. Environment configuration
   - API endpoint URLs
   - Development ports

### Database Setup
1. MongoDB installation
2. Database creation
3. User setup and authentication
4. Collections initialization

## Technical Constraints
1. PDF document formatting compatibility
2. Google GenAI API limitations
3. Processing performance considerations
4. Data validation requirements
5. MongoDB document size limits
6. Frontend browser compatibility
7. API rate limiting
8. Security requirements
   - Authentication
   - Authorization
   - Data encryption
