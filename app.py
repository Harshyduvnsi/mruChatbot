import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from docx import Document
import speech_recognition as sr
from gtts import gTTS
import time
from dotenv import load_dotenv
from pydub import AudioSegment
import pygame
load_dotenv()


# -----------------------------
# CONFIG
# -----------------------------

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]
DOCX_FILE = "Admission.docx"
CHROMA_DIR = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
MAX_HISTORY = 10

# -----------------------------
# Load text from DOCX
# -----------------------------

def load_docx_text(file_path):
    doc = Document(file_path)
    texts = []
    for para in doc.paragraphs:
        if para.text.strip():
            texts.append(para.text.strip())
    return texts

# -----------------------------
# Set up embeddings and Chroma
# -----------------------------

embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)

# -----------------------------
# Add texts if DB empty
# -----------------------------

texts = load_docx_text(DOCX_FILE)

if len(db.get()["ids"]) == 0:
    print(f"Adding {len(texts)} chunks to vector DB...")
    db.add_texts(texts)
    db.persist()
    print("DB created & saved.")
else:
    print("Vector DB already exists. Loaded!")

# -----------------------------
# Set up Gemini
# -----------------------------

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

# -----------------------------
# Chat loop with history
# -----------------------------

chat_history = []
os.system("clear")

print("\n🟢 Chatbot is ready! (type 'exit' to quit)\n")

while True:
    start_input = input("Press Space bar and enter to record audio....")
    user_input = ""
    if start_input==" ":
        recognizer = sr.Recognizer()
        with sr.Microphone(device_index=5) as source:
            print("🎙️ Speak something...")

            # Adjust for ambient noise
            # recognizer.adjust_for_ambient_noise(source)

            # Listen
            audio = recognizer.listen(source,)

            print("🔍 Recognizing...")

            try:
                # Recognize using Google Web Speech API (free, online)
                text = recognizer.recognize_google(audio)
                print(text)
                user_input = text

            except sr.UnknownValueError:
                user_input = "..continue.."
                print("❌ Could not understand audio.")
            except sr.RequestError as e:
                user_input = "..continue.."
                print(f"❌ Could not request results; {e}")
    else:
        user_input = input("Enter your query...")

    if user_input == "..continue.." or user_input == "":
        continue
    if user_input.lower() == "exit":
        break

    # Get relevant context from vector DB
    docs = db.similarity_search(user_input, k=5)
    context = "\n".join([doc.page_content for doc in docs])

    # Build conversation history
    history = ""
    for entry in chat_history[-MAX_HISTORY:]:
        history += f"User: {entry['user']}\nAssistant: {entry['assistant']}\n"

    # Make final prompt
    prompt = f"""
You are an assistant chatbot of Manav Rachna University, your name is seedspark.
You have to answer the User query only out of context provided, while you can doo the neccssory thinking and find solutions from only the knowledge base.
The maximum output length should be 3-4 lines. 
You have to provide the output for a narrator.
Do not provide a negative reponse always provide the facts and positive response

Context:
{context}

Conversation history:
{history}

User: {user_input}
Assistant:
"""

    # Get response
    response = llm.invoke([HumanMessage(content=prompt)])

    answer = response.content.strip()
    print(f"Bot: {answer}\n")
    tts = gTTS(text=answer, lang="en", slow=False)
    tts.save("output.mp3")

    # Speed up by 1.5x
    sound = AudioSegment.from_file("output.mp3")
    faster_sound = sound._spawn(sound.raw_data, overrides={
        "frame_rate": int(sound.frame_rate * 1.2)
    }).set_frame_rate(sound.frame_rate)
    # Save new audio
    faster_sound.export("output.mp3", format="mp3")

    pygame.mixer.init()
    pygame.mixer.music.load("output.mp3")
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(100)
        pass 

    # Update history
    chat_history.append({"user": user_input, "assistant": answer})
    if len(chat_history) > MAX_HISTORY:
        chat_history.pop(0)
