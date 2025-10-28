import streamlit as st
import speech_recognition as sr
import pyttsx3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------ FAQ DATABASE ------------------
faqs = {
    "what courses are available": "Our university offers B.Tech, M.Tech, MBA, MCA, BBA, BCA, B.Sc, M.Sc, Law, and PhD programs.",
    "what is the admission process": "You can apply online through our admission portal. Some courses require entrance exams, while others are based on merit.",
    "what is the fee structure": "The fee varies depending on the course. Please check the official website fee section for detailed information.",
    "where is the university located": "We are located on Mussoorie Road, Dehradun, Uttarakhand.",
    "what are the hostel facilities": "Hostels are available for both boys and girls with mess, Wi-Fi, laundry, 24/7 security, and medical facilities.",
    "what is the placement record": "We have a strong placement record with recruiters like Google, Microsoft, Infosys, TCS, Wipro, and Amazon.",
    "what scholarships are available": "We provide scholarships based on merit, financial need, sports achievements, and government schemes.",
    "how to apply for scholarships": "You can apply during admission by submitting required documents. Scholarship forms are available on the official portal.",
    "what are the library facilities": "The library has thousands of books, journals, e-resources, digital labs, and quiet study areas.",
    "what sports facilities are available": "We have cricket, football, basketball, badminton, tennis, athletics, and indoor games facilities.",
    "are there extracurricular activities": "Yes! We have cultural fests, technical clubs, music, dance, theater, debate, photography, and entrepreneurship clubs.",
    "what is the exam pattern": "Most programs follow a semester system with mid-term and end-term exams, along with internal assessments.",
    "is there a transportation facility": "Yes, university buses operate across the city for students and staff.",
    "what is the attendance policy": "Students are required to maintain at least 75% attendance in each subject.",
    "are internships provided": "Yes, internships are arranged through industry tie-ups, career cell, and student-industry connect programs.",
    "what is the ragging policy": "Our university has a strict anti-ragging policy with zero tolerance.",
    "is there an alumni network": "Yes, we have a global alumni network with regular meetups and mentoring opportunities.",
    "what are the canteen facilities": "We have multiple canteens offering vegetarian, non-vegetarian, and healthy food options at affordable prices.",
    "are international students allowed": "Yes, we welcome international students and provide dedicated support for visas, accommodation, and orientation.",
    "how can i contact the university": "You can call +91-844591583 or email dituniversity.edu.in."
}

# ------------------ NLP MODEL ------------------
vectorizer = TfidfVectorizer()
faq_keys = list(faqs.keys())
faq_vectors = vectorizer.fit_transform(faq_keys)

def chatbot_response(user_query):
    query_vector = vectorizer.transform([user_query])
    similarity = cosine_similarity(query_vector, faq_vectors)
    idx = similarity.argmax()
    if similarity[0][idx] > 0.2:
        return faqs[faq_keys[idx]]
    else:
        return "❌ Sorry, I don’t know the answer. Please contact the university office."

# ------------------ VOICE & SPEECH ------------------
def voice_to_text():
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("🎙 Speak now...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        text = recognizer.recognize_google(audio)
        return text
    except Exception:
        st.warning("🎤 Voice input not available in this environment.")
        return ""

def speak_text(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception:
        st.warning("🔊 Speech output not supported in this environment.")

# ------------------ STREAMLIT UI ------------------
st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #e0f7fa, #f1f8e9);
    }
    .chat-bubble-user {
        background-color: #0084ff;
        color: white;
        padding: 10px;
        border-radius: 15px;
        margin: 5px 0;
        width: fit-content;
        max-width: 70%;
    }
    .chat-bubble-bot {
        background-color: #e5e5ea;
        color: black;
        padding: 10px;
        border-radius: 15px;
        margin: 5px 0;
        width: fit-content;
        max-width: 70%;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎓 University Enquiry Chatbot")
st.write("Ask me anything about admissions, fees, courses, hostel, placements, etc.")

# Sidebar
st.sidebar.header("💡 Topics You Can Ask About")
for q in faqs.keys():
    st.sidebar.write("👉 " + q.capitalize())

with st.sidebar.expander("📌 What you can ask me about"):
    st.markdown("""
    - 📚 Courses & Admissions  
    - 💰 Fee Structure & Scholarships  
    - 🏫 Location & Hostel  
    - 💼 Placements & Internships  
    """)

# Quick Buttons
st.markdown("### 💡 Quick Questions")
col1, col2, col3 = st.columns(3)
button_pressed = None
if col1.button("📝 Admissions"):
    button_pressed = "admission process"
elif col2.button("💰 Fees"):
    button_pressed = "fee structure"
elif col3.button("🏫 Hostel"):
    button_pressed = "hostel facilities"

col4, col5, col6 = st.columns(3)
if col4.button("📚 Courses"):
    button_pressed = "courses available"
elif col5.button("💼 Placements"):
    button_pressed = "placement record"
elif col6.button("🎓 Scholarships"):
    button_pressed = "scholarships available"

# Input
user_input = st.text_input("💬 Type your question:")

if st.button("🎤 Ask by Voice"):
    user_input = voice_to_text()
    if user_input:
        st.markdown(f"<div class='chat-bubble-user'>🧑‍🎓 You: {user_input}</div>", unsafe_allow_html=True)

# Final query
final_input = button_pressed if button_pressed else user_input

if final_input:
    st.markdown(f"<div class='chat-bubble-user'>🧑‍🎓 You: {final_input}</div>", unsafe_allow_html=True)
    response = chatbot_response(final_input)
    st.markdown(f"<div class='chat-bubble-bot'>🤖 Bot: {response}</div>", unsafe_allow_html=True)

    # Speech Output
    if st.checkbox("🔊 Read out response"):
        speak_text(response)
