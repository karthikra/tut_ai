from fastapi import FastAPI, Form
from fastapi import FastAPI, Depends,Request


from pydantic import BaseModel

from src.models.response_models import TaskInputs

from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles


from src.ai_tut.crew import AIPromptingTutorCrew
import uvicorn
from dotenv import load_dotenv
import os
load_dotenv('/Users/karthikramesh/env/.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Create the FastAPI app
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize the AIPromptingTutorCrew
prompt_crew = AIPromptingTutorCrew()


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("/home/index.html", {"request": request})

@app.post("/run-tasks/")
async def run_tasks(
    prompt: str = Form(...),
    current_role: str = Form(...),
    career_path: str = Form(...),
    field: str = Form(...)

):
# Create a dictionary from the form data
    input_dict = {
    "prompt": prompt,
    "current_role": current_role,
    "career_path": career_path,
    "field": field
}
# Use the Pydantic model to validate and parse the data
    inputs = TaskInputs(**input_dict)
    input_dict = inputs.model_dump()

    # Use the default crew kickoff to run all tasks with the provided inputs
    crew_instance = prompt_crew.crew()
    response = crew_instance.kickoff(inputs=input_dict)  # Use kickoff() to execute tasks

    # Extract results from response.tasks_output
    structured_response = {}
    for task_output in response.tasks_output:
        task_name = task_output.name
        pydantic_output = task_output.pydantic

        # Convert the Pydantic model to a dictionary
        if pydantic_output:
            structured_response[task_name] = pydantic_output.dict()

    # Return the structured response dictionary
    return structured_response

def main():
    uvicorn.run(app, host='127.0.0.1', port=8000)
    pass


if __name__ == '__main__':
    main()
