from dotenv import load_dotenv
load_dotenv()

from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

# Prepare the LLM for the agent to use
from _lessonshelper.pretty_print_callback_handler import PrettyPrintCallbackHandler
pretty_callback = PrettyPrintCallbackHandler()

from langchain_openai import OpenAI
#agent_llm = OpenAI(temperature=0, callbacks=[pretty_callback])
agent_llm = OpenAI(temperature=1, callbacks=[])

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
    allow_delegation=False,
    tools=[search_tool],
    llm=agent_llm
)

# Create tasks for your agents
task1 = Task(
    description="""Conduct a thorough research into the question of liberalism in the victorian era and beyond. Idenitify key historical resources to look further into, and their significance in the wider question of how new liberalism was different from the old.""",
    expected_output="""A bullet point list of topics to cover in an undergraduate essay.""",
    agent=lecturer,
    llm=agent_llm,
)

task2 = Task(
    description="""Respond to the lecturer with further questions based on your researches.""",
    expected_output="""A bullet point list of your areas of confusion on the subject, and further areas of study you have found not mentioned by the lecturer that may be worth looking into.""",
    agent=student,
    llm=agent_llm,
)

task3 = Task(
    description="""Respond to the student with further elucidation of their areas of confusion.""",
    expected_output="""A concise summary of further texts to consider and arguments to explore.""",
    agent=lecturer,
    llm=agent_llm,
)


# Instantiate your crew with a sequential process
crew = Crew(
    agents=[lecturer, student],
    tasks=[task1, task2, task3, task2, task3],
    verbose=99,  # You can set it to 1 or 2 to different logging levels
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)


