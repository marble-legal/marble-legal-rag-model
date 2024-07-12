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

def chat_scale_ai(query,history,jurisdiction,follow_up_flag):
    prompt_template = """
        Act as an expert lawyer and use the following pieces of context of cases and chat history to formulate an answer to the question - {question} based on them the jurisdiction is US - {jurisdiction} state. If you don't know the answer, just say that you don't know, don't try to make up an answer. Return only answer and nothing else.
        {context}
        Question: {question}
        Chat History: {chat_history}
        Helpful Answer:"""
   
    try:
        chat_hist_dict_for_llm = [(x,y) for x,y in zip(history['User'],history['Assistant'])] 
    except:
        chat_hist_dict_for_llm = []
    
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question","chat_history"])
    loaded_db = Chroma(persist_directory=f'VDbs/{jurisdiction}', embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"))
    docs = loaded_db.similarity_search_with_relevance_scores(query=query,k=1)
    if docs[0][1] >= 0.1 or follow_up_flag:
        llm = ChatOpenAI(model_name='gpt-4o', temperature=0)
        question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
        doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)
        test_retriever = loaded_db.as_retriever(search_type="similarity_score_threshold",search_kwargs={'score_threshold':0.1})
        qa = ConversationalRetrievalChain(
            combine_docs_chain=doc_chain,
            retriever=test_retriever,
            question_generator=question_generator,
            return_source_documents=True
        )

        if len(chat_hist_dict_for_llm) > 0:
            payload = qa({"question": str(query), "chat_history": chat_hist_dict_for_llm})
        else:
            payload = qa({"question": str(query), "chat_history": []})
        source_urls = []
        for doc in payload['source_documents']:
            source_urls.append(json.loads(doc.metadata['source'].replace('\'','"')))
        # source_urls = list(set(source_urls))
        return payload['answer'],source_urls,False
    else:
        _llm = ChatOpenAI(model_name='gpt-4o', temperature=0)
        SECOND_FOLLOW_UP_PROMPT_TEMPLATE = """ Act as an expert lawyer. Now generate the question based on the {chat_history} and the most recent query {question} the jurisdiction is US - {jurisdiction} state. Identify the main subject of discussion from the chat history and query. Using this generate one question that you ask for additional information after the question to narrow down the choices or to get more clarity (DON'T paraphrase the question or ask similar question and also don't ask anything already present in this conversation, use this for context if related, and don't ask entirely new question only follow up question). Return only the new generated question in a well laid out text format :-  Q. and nothing else"""
        SECOND_FOLLOW_UP_PROMPT = PromptTemplate(template=SECOND_FOLLOW_UP_PROMPT_TEMPLATE,input_variables=["chat_history","question","jurisdiction"])  
        question_generator = LLMChain(llm=_llm, prompt=SECOND_FOLLOW_UP_PROMPT)
        new_question = question_generator({"chat_history": chat_hist_dict_for_llm,"question":query})
        return new_question['text'],[],True
    


@app.route('/ask', methods=['POST'])
def ask():
    x_api_key = request.headers.get('x-api-key')
    if x_api_key not in VALID_API_KEYS:
        return jsonify({"error": "Invalid API key"}), 401
    data = request.json
    query = data.get('query')
    history = data.get('history')
    jurisdiction = data.get('jurisdiction')
    follow_up_flag = data.get('follow_up_flag',False)
    if not query:
        return jsonify({"error": "Query parameter is required"}), 400
    if not history:
        history = []
    if not jurisdiction:
        return jsonify({"error": "Jurisdiction parameter is required"}), 400
    answer,source_documents,follow_up = chat_scale_ai(query,history,jurisdiction,follow_up_flag)
    unique_source_documents = [dict(t) for t in {tuple(d.items()) for d in source_documents}]
    return jsonify({"answer": answer,"source_documents":unique_source_documents,"follow_up":follow_up}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
