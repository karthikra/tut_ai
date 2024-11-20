from pydantic import BaseModel
from typing import List, Optional

# Model for Prompt Feedback
class PromptFeedback(BaseModel):
    prompt: str
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
    counter_examples: List[str]

# Model for Career Advice
class CareerAdvice(BaseModel):
    career_path: str
    current_role: str
    skills_to_develop: List[str]
    resources: List[str]

# Model for Job Advice
class JobAdvice(BaseModel):
    job_roles: List[str]
    industries_to_explore: List[str]
    tips: List[str]

class PromptScore(BaseModel):
    prompt: str
    score: float


class PromptRewrite(BaseModel):
    prompt: str
    rewritten_prompt: List[str]

class Badge(BaseModel):
    badge_name: str
    badge_description: str
    badge_image: str


# Define a Pydantic model for the input
class TaskInputs(BaseModel):
    prompt: str
    current_role: str
    career_path: str
    field: str