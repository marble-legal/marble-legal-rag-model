from flask import Flask, request, jsonify
import json
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()
VALID_API_KEYS = {os.getenv("FLASK_API_KEY")}
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
app = Flask(__name__)

def chat_scale_ai(query):
    prompt_template = """
        Act as an expert lawyer and use the following pieces of context of cases to formulate an answer based on them. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        {context}
        Question: {question}
        Helpful Answer:"""
    try:
        with open('chat_history.json', 'r') as chat_hist_load:
            chat_hist_dict = json.load(chat_hist_load)
            chat_hist_dict_for_llm = [(x, y) for x, y in chat_hist_dict['Question']]
        if len(chat_hist_dict) > 2:
            chat_hist_dict = {}
            with open('chat_history.json', 'w') as chat_hist_load:
                json.dump({}, chat_hist_load)
    except FileNotFoundError:
        chat_hist_dict = {}
    except json.JSONDecodeError:
        chat_hist_dict = {}

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    loaded_db = Chroma(persist_directory='./chromadb', embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"))
    llm = ChatOpenAI(model_name='gpt-4o', temperature=0.1)
    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)
    test_retriever = loaded_db.as_retriever()
    qa = ConversationalRetrievalChain(
        combine_docs_chain=doc_chain,
        retriever=test_retriever,
        question_generator=question_generator,
    )

    if len(chat_hist_dict) > 0:
        payload = qa({"question": str(query), "chat_history": chat_hist_dict_for_llm})
    else:
        payload = qa({"question": str(query), "chat_history": []})

    if 'Question' not in chat_hist_dict:
        chat_hist_dict["Question"] = []

    chat_hist_dict['Question'].append((query, payload['answer']))

    with open('chat_history.json', 'w') as chat_hist:
        json.dump(chat_hist_dict, chat_hist)

    return payload['answer']

@app.route('/ask', methods=['POST'])
def ask():
    x_api_key = request.headers.get('x-api-key')
    if x_api_key not in VALID_API_KEYS:
        return jsonify({"error": "Invalid API key"}), 401
    data = request.json
    query = data.get('query')
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    answer = chat_scale_ai(query)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
