


class PatientData(BaseModel):
    id_patient: int = Field(description="It is the DOCUMENT ID. Located in the beginning of the document.")
    gender: Optional[str] = Field(default=None, description="M for male or F for female.")
    full_name: Optional[str] = Field(default=None, description="In the admission note. Full name of the patient.")
    date_of_birth: Optional[str] = Field(default=None, description="In the admission note. In the line after the gender. In the line befor the name of the patient. Return format dd-mm-yyyy")  # dd-mm-yyyy
    process_number: Optional[int] = Field(default=None, description="In the admission note. In the line before the line that contains 'Nº Processo:'") 
    address: Optional[str] = Field(default=None, description="In the admission note. In the lines before the line that contains 'Data Nasc:'.")
    admission_date: Optional[str] = Field(default=None, description="In the admission note, look for date of 'admissão' or 'internamento' in the burns unit (UQ). Output format dd-mm-yyyy.")  # dd-mm-yyyy
    admission_time: Optional[str] = Field(default=None, description="In the admission note, look for time of 'admissão' or 'internamento' in the burns unit (UQ). Output format hh:mm.")  # hh:mm
    origin: Optional[str] = Field(default=None, description="In the admission note, look where the patient was before being transfered to the burns unit.")
    discharge_date: Optional[str] = Field(default=None, description="In the discharge note in the line before the line that contains 'Dr(a).'. If there is no discharge note, look for date of deathg in death note. Output format dd-mm-yyyy.")  # dd-mm-yyyy
    destination: Optional[str] = Field(default=None, description="In the discharge note, look where the patient was transfered after leaving the burns unit.")