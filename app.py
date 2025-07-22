import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Optional: load from .env file
load_dotenv()

# 🔐 Set your Gemini API key (prompt if not found)
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY") or input("Enter your Gemini API Key: ")

# ⚡ Initialize fast Gemini model (1.5 Flash)
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)

print("💬 Gemini Streaming Chat (type 'exit' to quit)")

# 🧠 Stream responses
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in {"exit", "quit"}:
        break

    print("Gemini: ", end="", flush=True)
    for chunk in model.stream(user_input):
        print(chunk.content, end="", flush=True)
