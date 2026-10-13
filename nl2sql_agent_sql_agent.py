import os
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.base import create_sql_agent

# Verify credentials
if not os.environ.get("OPENAI_API_KEY"):
    print("⚠️ Error: Please set your OPENAI_API_KEY environment variable first!")
    exit(1)

# 1. Connect to our local database
db = SQLDatabase.from_uri("sqlite:///ecommerce.db")

# 2. Initialize the Language Model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 3. Construct the official LangChain SQL Agent 
agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="openai-tools",
    verbose=True
)

# 4. Ask the Agent questions in plain English
user_query = "What is our best selling product category by revenue, and what was the total revenue?"

print(f"🤔 User Asked: {user_query}\n")
response = agent_executor.invoke({"input": user_query})

print("\n🤖 Agent Final Answer:")
print(response["output"])
