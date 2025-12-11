import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# --- CONFIGURATION ---
st.set_page_config(page_title="Agent Mistral", page_icon="⚡")
st.title("⚡ Mon Agent Expert (Mistral)")

# --- VÉRIFICATION CLÉ API ---
if "GROQ_API_KEY" not in os.environ:
    st.error("⚠️ Clé API manquante ! Configurez-la dans les 'Secrets' de Streamlit.")
    st.stop()

# --- CHARGEMENT ---
@st.cache_resource
def load_db():
    # --- LISTE DES DOCUMENTS ---
    # Mettez EXACTEMENT les noms de vos fichiers ici :
    fichiers = ["Arrêté du 6 février 2025", "[Nomenclature] Arrêté du 31 mars 2025","[Nomenclature] Arrêté modificatif du 10 octobre 2025", "lppr", "Tarifs", "Nomenclature-titre-IV-2025", "Nomenclature-titre-I-2025"] 

    documents = []
    for f in fichiers:
        if os.path.exists(f):
            if f.endswith(".pdf"):
                loader = PyPDFLoader(f)
                documents.extend(loader.load())
            elif f.endswith(".txt"):
                loader = TextLoader(f)
                documents.extend(loader.load())
    
    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    # Embeddings (gratuits)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

if "vectorstore" not in st.session_state:
    with st.spinner("Analyse des documents..."):
        try:
            st.session_state.vectorstore = load_db()
        except Exception as e:
            st.error(f"Erreur de lecture : {e}")

# --- IA ---
if st.session_state.vectorstore:
    llm = ChatGroq(model_name="mixtral-8x7b-32768", temperature=0)
    
    template = """Réponds uniquement avec le contexte fourni. Si tu ne sais pas, dis-le.
    Contexte : {context}
    Question : {question}
    Réponse :"""
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 2}),
        chain_type_kwargs={"prompt": PromptTemplate.from_template(template)}
    )

    # --- CHAT ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Votre question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            response = qa_chain.invoke({"query": prompt})
            st.markdown(response["result"])
            st.session_state.messages.append({"role": "assistant", "content": response["result"]})
