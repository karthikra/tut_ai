from typing import Dict
import yaml
from crewai import Agent, Crew, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import DallETool
from crewai.flow.flow import listen, start, and_, or_, router

from src.models.response_models import PromptFeedback, CareerAdvice, JobAdvice, PromptScore, PromptRewrite, Badge


@CrewBase
class AIPromptingTutorCrew():
    def __init__(self):
        # Load the agents.yaml configuration file
        with open("src/ai_tut/config/agents.yaml", "r") as file:
            self.agents_config = yaml.safe_load(file)

        # Load the tasks.yaml configuration file
        with open("src/ai_tut/config/tasks.yaml", "r") as file:
            self.tasks_config = yaml.safe_load(file)

        # Initialize inputs as placeholders
        self.prompt_input = "Default prompt"
        self.career_input = {"desired_path": "Data Scientist"}
        self.field_input = {"interest": "Technology"}

    @agent
    def ai_tutor(self) -> Agent:
        # Initialize the agent using the configuration
        return Agent(
            config=self.agents_config['ai_tutor'],
            verbose=True
        )
    def ai_prompt_scorer(self) -> Agent:
        # Initialize the agent using the configuration
        return Agent(
            config=self.agents_config['ai_prompt_scorer'],
            verbose=True
        )
    def ai_prompt_rewrite(self) -> Agent:
        # Initialize the agent using the configuration
        return Agent(
            config=self.agents_config['ai_prompt_rewriter'],
            verbose=True
        )
    def ai_badge_maker(self) -> Agent:
        # Initialize the agent using the configuration
        return Agent(
            config=self.agents_config['ai_badge_maker'],
            tools = [DallETool()],
            allow_delegation=False,
            verbose=True
        )

    @task
    def prompt_refinement_task(self) -> Task:
        # Initialize the prompt refinement task
        return Task(
            config=self.tasks_config['tasks']['prompt_refinement'],
            agent=self.ai_tutor(),
            output_pydantic=PromptFeedback
        )

    @task
    def career_advice_task(self) -> Task:
        # Initialize the career advice task
        return Task(
            config=self.tasks_config['tasks']['career_advice'],
            agent=self.ai_tutor(),
            output_pydantic=CareerAdvice
        )

    @task
    def field_suggestions_task(self) -> Task:
        # Initialize the field-specific suggestions task
        return Task(
            config=self.tasks_config['tasks']['field_suggestions'],
            agent=self.ai_tutor(),
            output_pydantic = JobAdvice
        )

    @task
    def prompt_scoring_task(self) -> Task:
        # Initialize the field-specific suggestions task
        return Task(
            config=self.tasks_config['tasks']['prompt_scoring'],
            agent=self.ai_prompt_scorer(),
            output_pydantic= PromptScore
        )
    @task
    def prompt_rewrite_task(self) -> Task:
        # Initialize the field-specific suggestions task
        return Task(
            config=self.tasks_config['tasks']['prompt_rewrite'],
            agent=self.ai_prompt_rewrite(),
            output_pydantic= PromptRewrite
        )
    @task
    @listen(prompt_scoring_task)
    def badge_task(self) -> Task:
        # Initialize the field-specific suggestions task
        return Task(
            config=self.tasks_config['tasks']['badge_making'],
            agent=self.ai_badge_maker(),
            context = [self.prompt_scoring_task()],
            output_pydantic=Badge,
        )

    @crew
    def crew(self) -> Crew:
        # Set up the crew with the agent and tasks
        return Crew(
            agents=[
                self.ai_tutor(),
                self.ai_prompt_scorer(),
                self.ai_badge_maker(),
                self.ai_prompt_rewrite()
                    ],
            tasks=[
                self.prompt_refinement_task(),
                self.career_advice_task(),
                self.field_suggestions_task(),
                self.prompt_scoring_task(),
                self.badge_task(),
                self.prompt_rewrite_task()
            ],
            process="sequential",
            verbose=True
        )

