import streamlit as st
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import pint

st.title("⚙️ Engineering Unit Converter Bot")
st.write("Convert engineering units and learn their meanings!")

# Load RAG
embeddings = OpenAIEmbeddings()
db = Chroma(persist_directory="db", embedding_function=embeddings)
retriever = db.as_retriever()
llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = PromptTemplate(template="Explain clearly: {question}", input_variables=["question"])
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs={"prompt": prompt})
ureg = pint.UnitRegistry()

query = st.text_input("Enter your query (e.g. 'Convert 5 bar to psi')")

if st.button("Run"):
    try:
        parts = query.lower().replace("convert", "").split("to")
        value_unit = parts[0].strip()
        target_unit = parts[1].strip()
        value, unit = value_unit.split(" ")
        q = float(value) * ureg(unit)
        result = q.to(target_unit)
        st.success(f"✅ {value_unit} = {result}")
    except:
        st.warning("Format: Convert [value unit] to [target unit]")

    answer = qa.run(query)
    st.info(f"Explanation: {answer}")
