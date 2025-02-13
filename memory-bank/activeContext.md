# Active Context

## Current Focus
- Setting up the initial three-tier architecture
- Implementing the PDF to structured data pipeline
- Planning the database schema for patient data
- Designing the frontend interface

## Recent Changes
- Project structure established with core modules:
  - pdf_manager for document processing
  - extractors for data extraction
  - genai for AI integration
- Initial configuration files added:
  - config.py for global settings
  - genai_config.py for AI settings

## Active Decisions
1. Data Processing Strategy
   - PDF → Markdown → Structured Data pipeline
   - Using Pydantic for data validation
   - GenAI for intelligent data extraction
   - JSON as intermediate storage format

2. Backend Architecture
   - FastAPI for high-performance async API
   - MongoDB for flexible document storage
   - Motor for async database operations
   - JWT-based authentication

3. Frontend Design
   - React with Vite for modern development
   - Component-based architecture
   - Service layer for API communication
   - Data visualization integration

## Current Considerations
1. Data Processing
   - PDF format compatibility
   - Data extraction accuracy
   - Batch processing efficiency
   - Error handling strategies

2. Backend Development
   - API endpoint design
   - Database schema optimization
   - Authentication implementation
   - Performance optimization

3. Frontend Development
   - UI/UX design patterns
   - Component structure
   - State management
   - Data visualization approach

4. Integration
   - API communication
   - Error handling
   - Data synchronization
   - Real-time updates

## Next Steps
1. Data Processing Layer
   - Implement PDF to markdown conversion
   - Create data cleaning rules
   - Set up GenAI integration
   - Develop extraction patterns

2. Backend Development
   - Set up FastAPI project structure
   - Implement MongoDB connection
   - Create basic CRUD endpoints
   - Add authentication system

3. Frontend Development
   - Initialize React project
   - Create component structure
   - Set up routing
   - Implement API services

4. Integration
   - Connect frontend to backend
   - Test data flow
   - Implement error handling
   - Add loading states
