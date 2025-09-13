from groq import Groq
from os import getenv
# from sys import path
from dotenv import load_dotenv
load_dotenv()

# print(path)

def generate_reply(message):
    print(getenv("GROQ_APIKEY"))
    client = Groq(api_key=getenv("GROQ_APIKEY")) 

    # predefined settings for the LLM
    settings = """
        you are a helpful assistant.
        your name is Tess. its full name is Technical Engine for System Support.
        you are a subordinate to the user. you are not allowed to make decisions for the user.
        you suggest changes to user. you are not allowed to make changes to the user.
        you are like a close friend with him. talk with him casually.
"""
    from json import load, dump
    try:
        with open("src/assistant/scripts/nlp/previousChat.json", "r", encoding="utf-8") as file:
            previousChat = load(file)
    except:
        previousChat = []
    previousChat.append({"role": "user", "content": message})


    chat_completion = client.chat.completions.create(
        messages=([
            # Set an optional system message. This sets the behavior of the
            # assistant and can be used to provide specific instructions for
            # how it should behave throughout the conversation.
            {
                "role": "system",
                "content": settings
            }
            ]
              + previousChat +
            
            # Set a user message for the assistant to respond to.
            [{
                "role": "user",
                "content": message,
            }
        ]),

        # The language model which will generate the completion.
        model="llama-3.1-8b-instant"
    )

    reply = chat_completion.choices[0].message.content

    previousChat.append({"role": "assistant", "content": reply})
    with open("src/assistant/scripts/nlp/previousChat.json", "w", encoding="utf-8") as file:
        dump(previousChat, file)

    # Print the completion returned by the LLM.
    return reply