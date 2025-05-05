from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from translate import Translator
import nltk
from langdetect import detect

nltk.download('punkt')

# --- 1. Prepare Training Data ---
questions = [
    "what is your name", "who created you", "are you a human", "can you talk",
    "what is ai", "who is elon musk", "what is the capital of japan", 
    "how to learn python", "what is html", "how to lose weight", 
    "tell me a joke", "how to be happy", "tum kon ho", "mujhe madad chahiye",
    "how to cure a headache", "what is python", "what is cryptocurrency",
    "who won the oscars 2024", "23 x 89"
]
answers = [
    "I am SmartBuddy, your AI friend.", "I was created by a developer using Python and AI.",
    "No, I am an AI program.", "Yes, I can talk. Ask me anything!",
    "AI means Artificial Intelligence. It allows machines to think like humans.",
    "Elon Musk is a famous entrepreneur and CEO of Tesla and SpaceX.",
    "The capital of Japan is Tokyo.",
    "Start with basics on w3schools or freeCodeCamp. Practice daily.",
    "HTML is the standard language for creating web pages.",
    "Eat healthy, exercise regularly, and stay hydrated.",
    "Why don’t scientists trust atoms? Because they make up everything!",
    "Be positive, take breaks, and focus on your goals.",
    "Main ek AI hoon, jo aapki madad ke liye yahan hoon.",
    "Zaroor! Batao kis cheez ki madad chahiye?",
    "Drink water, rest in a quiet place, and avoid screen light.",
    "Python is used for web, AI, data science, automation, and more.",
    "Cryptocurrency is digital money like Bitcoin used online.",
    "Everything Everywhere All at Once won Best Picture in 2024.", "2,047"
]

# --- 2. Train Naive Bayes Classifier ---
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(questions)
model_nb = MultinomialNB()
model_nb.fit(X, answers)

# --- 3. Load Transformer Model ---
tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-T5-738M")
model_transformer = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-T5-738M")

# --- 4. Translation Helper with Language Detection ---
translator = Translator(to_lang="en")

def detect_language(text):
    """Detect the language of the input text"""
    return detect(text)

def translate_text(text, src_lang='auto', dest_lang='en'):
    """Translate text to the target language"""
    return translator.translate(text)

# --- 5. Chatbot Response Function ---
def chatbot_response(user_input):
    try:
        # Language detection and translation if needed
        user_lang = detect_language(user_input)
        
        # If the input is not in English, translate it
        if user_lang != 'en':
            print(f"(Detected Language: {user_lang}) Translating to English...")
            translated = translate_text(user_input)
            print(f"(Translated): {translated}")
        else:
            translated = user_input
        
        # Naive Bayes Prediction
        vector = vectorizer.transform([translated])
        prediction = model_nb.predict(vector)[0]
        proba = model_nb.predict_proba(vector).max()

        if proba > 0.6:
            return prediction
        else:
            # Use transformer model if Naive Bayes confidence is low
            inputs = tokenizer(translated, return_tensors="pt", max_length=128, truncation=True)
            outputs = model_transformer.generate(**inputs, max_new_tokens=100)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response
    except Exception as e:
        return f"Error: {str(e)}"

# --- 6. Chat Loop ---
print("SmartBuddy: Hello! Ask me anything (in English or Roman Urdu)...")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("SmartBuddy: Bye! Take care.")
        break
    reply = chatbot_response(user_input)
    print("SmartBuddy:", reply)