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
    goal="Help a Junior Historian Stefan Collini's views on Victorian liberalism",
    backstory="""You are an experienced historian, an expert in Victorian intellectual, economic, and political history.""",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=agent_llm
)

student = Agent(
    role="Junior Historian",
    goal="Perform research to answer the question: 'What are Stefan Collini's views on Victorian liberalism?",
    backstory="""You are an undergraduate historian tasked with writing an undergraduate essay on the topic: 'How did new liberalism differ from old liberalism?""",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=agent_llm
)

# Create tasks for your agents
task1 = Task(
    description="""Idenitify key historical resources to look further into in answering the question.""",
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

tasks=[]
tasks.append(task1)
for i in range(10):
	tasks.append(task2)
	tasks.append(task3)

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[lecturer, student],
    tasks=tasks,
    verbose=99,  # You can set it to 1 or 2 to different logging levels
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)
print(crew.usage_metrics)

