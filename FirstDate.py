from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool
import os
import google.generativeai as genai

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]
    
gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key = "AIzaSyDe8_ru7Y-503iBt8HMqf4yyGKRyoTOubA")
llm = genai.GenerativeModel("gemini-2.5-pro")


def gemini_agent(user_input):
    response = llm.generate_content(user_input)
    return response.text
if __name__ == "__main__":
    print("Hello! welcome to FirstDateAI we are here to help you on your first date")
    print("Would you like help with fashion first? (y/n)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        elif user_input == "y":
            print("We got you")
            print(llm.generate_content("What should I wear on a first date?").text + "\n" +"Anything else you would like to ask?")
            if user_input == "y":
                continue
            elif user_input == "n":
                break
        elif user_input == "n":
            print("No worries, how about help with conversation topics? (y/n)")
            user_input = input("You: ")
            if user_input.lower() == "y":
                print("No worries we got you covered")
                print(llm.generate_content("What are some good conversation topics for a first date?").text + "\n" +"Anything else you would like to ask?")
                if user_input == "y":
                    continue
                elif user_input == "n":
                    break
            elif user_input == "n":
                print("Alright, how about help with date ideas? (y/n)")
                user_input = input("You: ")
                if user_input == "y":
                    print("No need to worry")
                    print(llm.generate_content("What are some good first date ideas?").text + "\n" +"Anything else you would like to ask?")
                    if user_input == "y":
                        continue
                    elif user_input == "n":
                        break
                    

                elif user_input == "n":
                    print("Alright, if you need anything else, just ask!")
                    continue
               
        else:
            reply = gemini_agent(user_input)
            print("Gemini:", reply)
