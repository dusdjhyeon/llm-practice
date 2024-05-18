from typing import Any
from fastapi import APIRouter

from app.agent.agent import ExecutorAgent
from app.schemas.chat import ResponseAnswer, RequestQuery
# from app.generation import Generator
# from app.schemas.generation import Generation, RequestText, Assessment, RequestAssessment
# from app.utils import PreProcessor, parsing_generation_output
# from app.utils.decorators import live_mode, validate_content

router = APIRouter()
agent = ExecutorAgent()

@router.post("/completion", response_model = ResponseAnswer)
def make_chat(req : RequestQuery):
    """
    입력받은 텍스트로부터 응답을 생성.\n
    텍스트(string)은 request body에서 필요.
    """
    answer = agent.run(req.query)
    return {"answer":answer}