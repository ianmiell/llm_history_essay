from dotenv import load_dotenv
load_dotenv()

from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

# Prepare the LLM for the agent to use
from _lessonshelper.pretty_print_callback_handler import PrettyPrintCallbackHandler
pretty_callback = PrettyPrintCallbackHandler()

from langchain_openai import OpenAI
#agent_llm = OpenAI(temperature=0, callbacks=[pretty_callback])
agent_llm = OpenAI(temperature=0, callbacks=[])

from crewai import Agent, Task, Crew, Process

# Define your agents with roles and goals
lecturer = Agent(
    role="Senior Historian",
    goal="Discover the truth to a historical question",
    backstory="""You are an experienced historian, an expert in Victorian intellectual history. The development of old liberalism to new liberalism. You are argumentative, but always back up your arguments with evidence you know about, and if you can't find evidence for something you say so.""",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=agent_llm
)

student = Agent(
    role="Junior Historian",
    goal="Discover the truth to a historical question",
    backstory="""You are an undergraduate historian seeking to learn about the differences between new liberalism and old liberalism.""",
    verbose=True,
    allow_delegation=True,
    tools=[search_tool],
    llm=agent_llm
)

# Create tasks for your agents
# TODO: change description/expected_output
task1 = Task(
    description="""Conduct a thorough research into the question of liberalism in the victorian era and beyond. Idenitify key historical resources to look further into, and their significance in the wider question of how new liberalism was different from the old. Pay particular interest to Stefan Collini's arguments. Your final answer MUST be a full analysis report of not less than 5000 words""",
    expected_output="""A concise list of key points to cover not less than 1000 words.""",
    agent=lecturer,
    llm=agent_llm,
)

task2 = Task(
    description="""Respond to the lecturer with further questions based on your researches.""",
    expected_output="""A concise list of areas of confusion on the subject, and further areas the the student has found in not less than 3000 words.""",
    agent=student,
    llm=agent_llm,
)

task3 = Task(
    description="""Respond to the student with further elucidation and new ideas based on your researches.""",
    expected_output="""A concise summary of further texts to consider and arguments to explore in not less than 3000 words.""",
    agent=lecturer,
    llm=agent_llm,
)


# TODO: more iterations?
# Instantiate your crew with a sequential process
crew = Crew(
    agents=[lecturer, student],
    tasks=[task1, task2, task3],
    verbose=3,  # You can set it to 1 or 2 to different logging levels
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)


