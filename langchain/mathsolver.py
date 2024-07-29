import streamlit as st
import time
import openai
from langchain import LLMChain, PromptTemplate
from langchain.agents import initialize_agent, Tool, load_tools
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv, find_dotenv

# Loading OpenAI API Key
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']


# Define the LLM and the math tool
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125")  # or use "text-davinci-003"
tools = load_tools(["llm-math"], llm=llm)

# Initialize the agent with the math tool
agent = initialize_agent(tools=tools, llm=llm, agent_type="zero-shot-react-description", verbose=True)

def solve_math_question(question):
    try:
        start_time = time.time()
        # Use the agent to solve the math question
        result = agent({"input": question})
        end_time = time.time()
        inference_time = end_time - start_time
        return result['output'], inference_time
    except ValueError as e:
        return "I'm sorry, but I cannot solve this problem. Please ask another math question.", None


def main():
    st.title("MathBOT, Ask me anything!")

    question = st.text_input("Enter a math question:")

    if st.button("Get Solution"):
        with st.spinner("Thinking..."):
            solution, inference_time = solve_math_question(question)
            st.success(f"Solution: {solution}")
            # st.text_area(label='', value=solution, height=200)
            if inference_time is not None:
                st.write(f"Inference Time: {inference_time:.2f} seconds")

if __name__ == "__main__":
    main()
