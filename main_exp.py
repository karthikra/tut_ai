from numba.cuda.printimpl import print_item

from src.ai_tut.crew import AIPromptingTutorCrew
from typing import List
from dotenv import load_dotenv
import os
load_dotenv('/Users/karthikramesh/env/.env')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


def run():
    # Import your AIPromptingTutorCrew class
    from src.ai_tut.crew import AIPromptingTutorCrew  # Replace 'your_module' with the actual module name

    # Initialize the AIPromptingTutorCrew
    prompt_crew = AIPromptingTutorCrew()

    # Set custom inputs for the agent
    inputs = {
        "prompt": "How can I improve my project proposals?",
        "current_role": "Data Analyst",
        "career_path": "Data Scientist",
        "field": "Marketing",
    }

    # Use the default crew kickoff to run all tasks sequentially
    crew_instance = prompt_crew.crew()
    response = crew_instance.kickoff(inputs=inputs)  # Use kickoff() to execute tasks with the provided inputs

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
    result = run()
    print(result)
    pass


if __name__ == '__main__':
    main()
