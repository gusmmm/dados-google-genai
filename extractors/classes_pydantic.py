from typing import Optional
from pydantic import BaseModel, Field

class ExtractionError(Exception):
    """Custom exception for data extraction errors."""
    pass

class PatientData(BaseModel):
    id_patient: int = Field(description="It is the DOCUMENT ID.")
    gender: Optional[str] = Field(default=None, description="M for male or F for female.")
    full_name: Optional[str] = Field(default=None, description="Full name of the patient.")
    date_of_birth: Optional[str] = Field(default=None, description="Date of birth of the patient. Return format dd-mm-yyyy")  # dd-mm-yyyy
    process_number: Optional[int] = Field(default=None, description="Patient's process number.") 
    address: Optional[str] = Field(default=None, description="Patient's date of birth.")
    admission_date: Optional[str] = Field(default=None, description="Patient's admission date in the burns unit (UQ). Output format dd-mm-yyyy.")  # dd-mm-yyyy
    admission_time: Optional[str] = Field(default=None, description="If present, hour of admission in the burns unit (UQ). Output format hh:mm.")  # hh:mm
    origin: Optional[str] = Field(default=None, description="Location of where the patient was before being transfered to the burns unit.")
    discharge_date: Optional[str] = Field(default=None, description="Date of discharge from the burns unit. If there is no discharge note, look for date of deathg in death note. Output format dd-mm-yyyy.")  # dd-mm-yyyy
    destination: Optional[str] = Field(default=None, description="Where the patient was transfered to, after leaving the burns unit.")