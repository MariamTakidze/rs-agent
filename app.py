import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document

# API Key — Streamlit Secrets-იდან
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

st.set_page_config(
    page_title="RS InfoHub RAG",
    page_icon="🇬🇪",
    layout="centered"
)

# ========================
# 1. დოკუმენტები
# ========================
def load_local_documents():
    docs = [
        Document(
            page_content="დამატებული ღირებულების გადასახადის (დღგ) განაკვეთი საქართველოში შეადგენს 18 პროცენტს. დღგ-ის გადამხდელად რეგისტრაცია სავალდებულოა, თუ ბრუნვა აღემატება 100,000 ლარს. დღგ-ით დაბეგვრის ობიექტია საქართველოს ტერიტორიაზე საქონლის მიწოდება, მომსახურების გაწევა და საქართველოში საქონლის იმპორტი.",
            metadata={"source": "საგადასახადო_კოდექსი_მუხლი_157.txt"}
        ),
        Document(
            page_content="მცირე ბიზნესის სტატუსი შეიძლება მიენიჭოს ფიზიკურ პირს, რომლის კალენდარული წლის განმავლობაში მიღებული ერთობლივი შემოსავალი არ აღემატება 500,000 ლარს. მცირე ბიზნესის სტატუსის მქონე პირი იხდის 1%-იან გადასახადს მიღებულ ბრუნვაზე. სტატუსი გაიცემა საგადასახადო ორგანოს მიერ.",
            metadata={"source": "მცირე_ბიზნესის_რეგულაციები.txt"}
        ),
        Document(
            page_content="საქონლის დეკლარირება და საბაჟო პროცედურები ხორციელდება დეკლარაციის წარდგენით. საბაჟო გამშვები პუნქტები მუშაობენ 24-საათიან რეჟიმში. საქართველოში საქონლის შემოტანისას გადამხდელი ვალდებულია წარადგინოს სასაქონლო დეკლარაცია. საბაჟო გადასახადი განისაზღვრება საქონლის სასაქონლო კოდის მიხედვით.",
            metadata={"source": "საბაჟო_ადმინისტრირება.txt"}
        ),
    ]
    return docs

# ========================
# 2. RAG სისტემა
# ========================
@st.cache_resource
def setup_rag():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    raw_docs = load_local_documents()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    split_docs = text_splitter.split_documents(raw_docs)

    vectorstore = FAISS.from_documents(split_docs, embeddings)

    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama3-70b-8192",
        temperature=0
    )

    template = """შენ ხარ საგადასახადო/საბაჟო ასისტენტი. უპასუხე კითხვას მხოლოდ კონტექსტზე დაყრდნობით ქართულ ენაზე.

კონტექსტი: {context}

კითხვა: {question}

პასუხი ჩამოაყალიბე გარკვევით. ბოლოში აუცილებლად მიუთითე:
1. კონკრეტული ფაილი, საიდანაც არის ინფორმაცია (📄 წყარო: ...)
2. სავალდებულო ტექსტი: "ℹ️ პასუხი მომზადებულია საინფორმაციო და მეთოდოლოგიური ჰაბზე განთავსებული დოკუმენტების მიხედვით — https://infohub.rs.ge/ka"
"""

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

# ========================
# 3. UI
# ========================
st.title("🇬🇪 RS InfoHub — RAG აგენტი")
st.caption("საგადასახადო და საბაჟო კითხვებზე პასუხი 3 დოკუმენტის საფუძველზე")

with st.expander("📂 გამოყენებული დოკუმენტები"):
    st.markdown("""
- 📘 `საგადასახადო_კოდექსი_მუხლი_157.txt` — დღგ-ის განაკვეთი და რეგისტრაცია
- 📗 `მცირე_ბიზნესის_რეგულაციები.txt` — სტატუსი და ბრუნვის ლიმიტი
- 📙 `საბაჟო_ადმინისტრირება.txt` — დეკლარირება და პროცედურები
    """)

st.divider()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
user_query = st.chat_input("დასვი კითხვა ქართულად... (მაგ: რა არის დღგ-ს განაკვეთი?)")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("პასუხი იძებნება..."):
            rag_chain = setup_rag()
            result = rag_chain.invoke(user_query)
            answer = result["result"]
            source_docs = result["source_documents"]

        st.markdown(answer)

        with st.expander("🔍 გამოყენებული Chunk-ები"):
            for doc in source_docs:
                st.markdown(f"**📄 {doc.metadata['source']}**")
                st.caption(doc.page_content)

    st.session_state.messages.append({"role": "assistant", "content": answer})
