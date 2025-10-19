from enum import Enum
from pydantic import BaseModel, Field

class Home(str, Enum):
    other = "other"
    rent = "rent"
    mortgage = "mortgage"
    own = "own"


class Intent(str, Enum):
    debtconsolidation = "debtconsolidation"
    personal = "personal"
    education = "education"
    medical = "medical"
    venture = "venture"
    homeimprovement = "homeimprovement"


class ClientData(BaseModel):
    age: int = Field(ge=18, le=120)
    income: int = Field(ge=0, le=100000000)
    home_ownership: Home
    employment_length: float = Field(ge=0, le=110)
    loan_amount: int = Field(ge=1, le=1000000000)
    def_on_file: float = Field(ge=0, le=1)
    loan_intent: Intent