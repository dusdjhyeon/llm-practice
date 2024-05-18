from pydantic import BaseModel

class RequestQuery(BaseModel):
    query: str

class ResponseAnswer(BaseModel):
    answer: str