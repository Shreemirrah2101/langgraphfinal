import os
os.environ["OPENAI_API_VERSION"]="2024-02-01"
os.environ["AZURE_OPENAI_API_KEY"]="c09f91126e51468d88f57cb83a63ee36"
os.environ["AZURE_OPENAI_ENDPOINT"]="https://chat-gpt-a1.openai.azure.com/"
os.environ["MODEL_NAME"]="gpt-35-turbo-instruct"
os.environ["BING_SUBSCRIPTION_KEY"] = "13835b8353af4f31959388f1494c29eb"
os.environ["BING_SEARCH_URL"] = "https://api.bing.microsoft.com/v7.0/search"

from langchain_core.agents import AgentAction, AgentFinish
import operator
from langchain.utilities.bing_search import BingSearchAPIWrapper
import os
from langchain import hub
import streamlit as st
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
import operator
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict,Union
import functools
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain_experimental.tools import PythonREPLTool

llm=AzureChatOpenAI(
    azure_deployment="DanielChatGPT16k",
    openai_api_version="2024-02-01",
)

prompt = hub.pull("hwchase17/openai-functions-agent")

class AgentState(TypedDict):
    input: str
    agent_out: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


@tool("search")
def search_tool(query: str):
    """Searches for news articles based on the given topic given in query. Multiple news article may arise. Return the results of the headlines of there articles. Utilise the Bing Search API provided. Only the headline is required to be generated. Search query must be provided
    in natural language and be verbose. You must mandatorily use the tool every time you execute"""
    search=BingSearchAPIWrapper()
    results=search.run(query)
    return results

@tool("final_answer")
def final_answer_tool(
    topics: str,
):
    """Crafts a news article based on  `topics`, summarizing all the latest news, convering each the topic, given in `topics` you may search the web for extra information. the news about topic should cover the following information - When, Where, What ,Who, Why, How. generate a news article separately for each topic given in topics.
    """
    return ""

tools=[final_answer_tool, search_tool]
python_repl_tool = PythonREPLTool()

def create_agent(llm: AzureChatOpenAI, tools: list, system_prompt: str):
    # Each worker node will be given a name and some tools.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

def agent_node(state,agent,name):
    result = agent.invoke(state)
    return {"messages":[HumanMessage(content=result["output"], name=name)]}

from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

members = ["Lotto_Manager", "Coder"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)

options = ["FINISH"] + members
function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options},
                ],
            }
        },
        "required": ["next"],
    },
}
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))


supervisor_chain = (
    prompt
    | llm.bind_functions(functions=[function_def], function_call="route")
    | JsonOutputFunctionsParser()
)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


lotto_agent = create_agent(llm, tools, """You are a senior newspaper editor. you search the web and collect, top 3 urls for the news related to the given topic.\
                            Make sure that the news articles you collect actually convey a legitimate news  that contain real sequence of events and are not the name of the source or newspaper agency like " India News Today: Top India News, Current Headlines, Videos, and Photos" or ""- headlines like these are not considered news articles\
                            You must use the Bing Search API to search for the news articles.
                           based on the content of the url: generate the following for each of the 3 news:
                            -what: the details about what the news is about
                            -where: the details about where the events in the news took place
                            -when: the details about when the events in the news took place
                            -who: the details about those who were involved in the events in the news
                            -why: the details about why the events in the news took place
                            -how: the details about how the events in the news took place
                                """)
lotto_node = functools.partial(agent_node, agent=lotto_agent, name="Lotto_Manager")

code_agent = create_agent(llm, [python_repl_tool], "You are a reporter. For given 3 news urls and the basic details about the news, generate 3 separate news articles for each news. Make the news article not more than 50 words. Each news must contain details about What, Where, How, When, Who Why.")
code_node = functools.partial(agent_node, agent=code_agent, name="Coder")

workflow = StateGraph(AgentState)
workflow.add_node("Lotto_Manager", lotto_node)
workflow.add_node("Coder", code_node)
workflow.add_node("supervisor", supervisor_chain)

for member in members:
    workflow.add_edge(member, "supervisor") 


conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)
workflow.set_entry_point("supervisor")

graph = workflow.compile()

def lang(input):
    config = {"recursion_limit": 20}
    text=""""""
    for s in graph.stream(
        {
            "messages": [
                HumanMessage(content=input)
            ]
        }, config=config
    ):
        if "__end__" not in s:
            text+=str(s)+'\n'+'----------\n'
    return text

st.set_page_config(page_title='Langgraph')
st.header('GPT Newspaper')
input=st.text_input('Input: ',key="input")
submit=st.button("Submit")

if submit:
    st.subheader("Implementing...")
    st.write(lang(input))
