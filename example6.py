import os
import uvicorn
import boto3
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import chromadb
from chromadb.utils import embedding_functions
from botocore.config import Config
from boto3 import Session

# Configuración de AWS Bedrock
bedrock_config = Config(
    region_name='us-east-1',
    signature_version='v4',
    retries={'max_attempts': 3}
)
# Cliente AWS Bedrock
sesion = Session(
    aws_access_key_id='',
    aws_secret_access_key='',
)
bedrock_runtime = sesion.client(
    service_name='bedrock-runtime', 
    config=bedrock_config
)

# Configuración de ChromaDB
chroma_client = chromadb.PersistentClient(path="./Storage/chroma_storage")

# Clase para gestionar el contexto de conversación
class ConversationContext:
    def __init__(self, max_history=5):
        """
        Inicializa el gestor de contexto de conversación
        
        :param max_history: Número máximo de mensajes a mantener en el historial
        """
        self.conversations = {}
        self.max_history = max_history
    
    def add_message(self, conversation_id: str, role: str, message: str):
        """
        Añade un mensaje al historial de una conversación
        
        :param conversation_id: ID único de la conversación
        :param role: Rol del mensaje (user/assistant)
        :param message: Contenido del mensaje
        """
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        # Añadir mensaje al historial
        self.conversations[conversation_id].append({
            "role": role,
            "message": message
        })
        
        # Limitar el tamaño del historial
        if len(self.conversations[conversation_id]) > self.max_history * 2:
            self.conversations[conversation_id] = self.conversations[conversation_id][-self.max_history*2:]
    
    def get_conversation_history(self, conversation_id: str) -> str:
        """
        Obtiene el historial de una conversación como texto
        
        :param conversation_id: ID único de la conversación
        :return: Historial de conversación formateado
        """
        if conversation_id not in self.conversations:
            return ""
        
        # Formatear historial como texto
        history = []
        for entry in self.conversations[conversation_id]:
            history.append(f"{entry['role'].upper()}: {entry['message']}")
        
        return "\n".join(history)

# Función para generar embeddings
def get_embedding_function():
    return embedding_functions.DefaultEmbeddingFunction()

# Función para invocar Claude 3 en Bedrock con contexto de conversación
def invoke_claude(prompt: str, context: str = "", conversation_history: str = "") -> str:
    try:
        # Combinar historial, contexto y prompt
        full_prompt = ""
        if conversation_history:
            full_prompt += f"Historial de conversación:\n{conversation_history}\n\n"
        
        if context:
            full_prompt += f"Contexto de documentos relevantes:\n{context}\n\n"
        
        full_prompt += f"Pregunta actual: {prompt}"
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": full_prompt
                }
            ]
        }
        
        response = bedrock_runtime.invoke_model(
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=json.dumps(body)
        )
        
        response_body = json.loads(response["body"].read().decode('utf-8'))
        return response_body['content'][0]['text']
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error invocando Claude: {str(e)}")

# Modelos Pydantic para validación
class DocumentInput(BaseModel):
    id: str
    text: str

class QueryInput(BaseModel):
    conversation_id: str
    query: str
    top_k: int = 3

class RAGChatbotAPI:
    def __init__(self):
        self.collection = chroma_client.get_or_create_collection(
            name="documents", 
            embedding_function=get_embedding_function()
        )
        self.conversation_context = ConversationContext()
    
    def add_documents(self, documents: List[DocumentInput]):
        """
        Añadir documentos a la base de vectores
        """
        try:
            ids = [doc.id for doc in documents]
            texts = [doc.text for doc in documents]
            
            self.collection.add(
                ids=ids,
                documents=texts
            )
            return {"status": "Documentos añadidos exitosamente"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    def query_documents(self, query: QueryInput):
        """
        Realizar búsqueda de documentos relevantes y generar respuesta
        """
        try:
            # Buscar documentos relevantes
            results = self.collection.query(
                query_texts=[query.query],
                n_results=query.top_k
            )
            
            # Obtener los documentos más relevantes como contexto
            context = "\n---\n".join(results['documents'][0])
            
            # Obtener historial de conversación
            conversation_history = self.conversation_context.get_conversation_history(
                query.conversation_id
            )
            
            # Generar respuesta con Claude 3
            response = invoke_claude(
                prompt=query.query, 
                context=context, 
                conversation_history=conversation_history
            )
            
            # Añadir mensajes al historial de conversación
            self.conversation_context.add_message(
                conversation_id=query.conversation_id, 
                role="user", 
                message=query.query
            )
            self.conversation_context.add_message(
                conversation_id=query.conversation_id, 
                role="assistant", 
                message=response
            )
            
            return {
                "conversation_id": query.conversation_id,
                "retrieved_docs": results['documents'][0],
                "response": response
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# Configuración de FastAPI
app = FastAPI(title="RAG Chatbot API con Contexto")
rag_chatbot = RAGChatbotAPI()

@app.post("/add_documents")
def add_documents(documents: List[DocumentInput]):
    return rag_chatbot.add_documents(documents)

@app.post("/query")
def query_chatbot(query: QueryInput):
    return rag_chatbot.query_documents(query)

# Iniciar el servidor
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)