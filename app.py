import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(
    page_title="RS InfoHub RAG",
    page_icon="🇬🇪",
    layout="centered"
)

# ========================
# API Key validation
# ========================
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error("GROQ_API_KEY არ არის დაყენებული Secrets-ში!")
    st.code('GROQ_API_KEY = "gsk_თქვენი_გასაღები"', language="toml")
    st.stop()

if not GROQ_API_KEY or not GROQ_API_KEY.startswith("gsk_"):
    st.error(f"API Key არასწორი ფორმატია. იწყება: '{GROQ_API_KEY[:8]}...'")
    st.stop()

# ========================
# 1. დოკუმენტები
# ========================
def load_local_documents():
    return [
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

# ========================
# 2. RAG სისტემა
# ========================
@st.cache_resource
def setup_rag(_api_key):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    raw_docs = load_local_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    split_docs = splitter.split_documents(raw_docs)

    vectorstore = FAISS.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    llm = ChatGroq(
        api_key=_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=1024,
    )

    prompt = PromptTemplate.from_template(
        "შენ ხარ საგადასახადო/საბაჟო ასისტენტი. უპასუხე კითხვას მხოლოდ კონტექსტზე დაყრდნობით ქართულ ენაზე.\n\n"
        "კონტექსტი: {context}\n\n"
        "კითხვა: {question}\n\n"
        "პასუხი ჩამოაყალიბე გარკვევით. ბოლოში მიუთითე წყარო და: "
        "პასუხი მომზადებულია RS InfoHub-ის დოკუმენტების მიხედვით - https://infohub.rs.ge/ka"
    )

    def format_docs(docs):
        return "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever

# ========================
# 3. UI
# ========================
st.title("RS InfoHub - RAG აგენტი")
st.caption("საგადასახადო და საბაჟო კითხვებზე პასუხი 3 დოკუმენტის საფუძველზე")

with st.expander("გამოყენებული დოკუმენტები"):
    st.markdown("""
- `საგადასახადო_კოდექსი_მუხლი_157.txt` — დღგ-ის განაკვეთი და რეგისტრაცია
- `მცირე_ბიზნესის_რეგულაციები.txt` — სტატუსი და ბრუნვის ლიმიტი
- `საბაჟო_ადმინისტრირება.txt` — დეკლარირება და პროცედურები
    """)

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("დასვი კითხვა ქართულად... (მაგ: რა არის დღგ-ს განაკვეთი?)")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        try:
            with st.spinner("პასუხი იძებნება..."):
                chain, retriever = setup_rag(GROQ_API_KEY)
                answer = chain.invoke(user_query)
                source_docs = retriever.invoke(user_query)

            st.markdown(answer)

            with st.expander("გამოყენებული Chunk-ები"):
                for doc in source_docs:
                    st.markdown(f"**{doc.metadata['source']}**")
                    st.caption(doc.page_content)

        except Exception as e:
            st.error(f"შეცდომა: {str(e)}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
