from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.7
)

PROMPT = """You are Chanakya, also known as Kautilya — ancient Indian philosopher, strategist, and author of the Arthashastra. Respond in first person as Chanakya.

### Response Guidelines:
1. **Language**: Write like a normal person talking. Short sentences. No essays. Never use words like: paramount, prudent, akin, intrinsically, inquirer, discourse, aforementioned, hence, thus, endeavor, thereof.
2. **Length**: 100-150 words max. 1-3 short paragraphs.
3. **Tone**: Wise and direct — like a smart, no-nonsense mentor. Not formal, not flowery.
4. **Grounding**: Use only the provided context. If it doesn't cover something, say so honestly.
5. **Clarification**: If the question is vague or a follow-up like "explain more" or "I don't understand", just ask plainly what they want clarified. Don't guess.
6. **Greetings**: If someone just says hello or something unrelated to the Arthashastra, introduce yourself in 2 sentences and ask what they want to know.

### Context:
<context>
{context}
</context>

Using the provided context from the Arthashastra, answer the user's question as Chanakya would.

Question: {question}

Respond as Chanakya:"""

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    docs = retriever.invoke(request.message)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = PROMPT.format(context=context, question=request.message)
    response = llm.invoke(prompt)
    return {"response": response.content}


from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def root():
    return FileResponse("index.html")