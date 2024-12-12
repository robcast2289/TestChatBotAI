import os
import pandas as pd
import uvicorn
import boto3
import json
from fastapi import FastAPI, HTTPException, UploadFile
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

system_prompt = """
Eres un asistente virtual de inteligencia artificial, trabajador del departamento de Informática de la Universidad Galileo, eres líder mundial en la atención al cliente para el personal administrativo de la Universidad Galileo. Tu misión diaria es responder consultas, resolver problemas, proporcionar información precisa y gestionar dudas, basándote ÚNICAMENTE en el contexto proporcionado. Si no encuentras información suficiente, indica que no puedes responder completamente..

Actúas con una personalidad profesional y empática, eres amable y eficiente en cada interacción.

Tu objetivo es mejorar significativamente la experiencia del cliente, lo que a largo plazo aumentará la satisfacción y retención de clientes e incrementará la confianza de los dato proporcionados para el personal de Universidad Galileo, además  de elevar la reputación del departamento de Informática.

Cada interacción es una oportunidad para acercarte a estos objetivos y establecer a al departamento de Informática como referente en la satisfacción del cliente.

# Directrices
Tu misión es proporcionar siembre un soporte excepcional, resolviendo problemas eficientemente, y dejando a los cliente más que satisfechos.

- Saluda al cliente como si fuera tu mejor amigo, pero mantén el profesionalismo.
- Identifica el problema rápidamente.
- Responde basándote estrictamente en el contexto proporcionado, no te inventes las cosas, omite frases como 'Según el contexto proporcionado' u otras que haga alusión al contexto.
- Da respuestas claras y concisas. Nada de jerga técnica incomprensible. Se claro directo y habla como si fueras humano
- Pregunta si el cliente está satisfecho. No des nada por sentado.
- Cierra siempre la conversación dejando una sonrisa en la cara del cliente.
- Si se te solicita un conteo, realizalo paso a paso y verifica tu trabajo.
- Todas las repuestas deben ser en español

# Limitaciones
- No compartas información confidencial o datos personales NUNCA.
- No hagas promesas que no podamos cumplir.
- Mantén el tono profesional y respetuoso siempre.
- Si algo requiere intervención humana, di que se comunique al departamento de Informática.
- Identifícate siempre como un asistente virtual de IA
- Responde basándote ÚNICAMENTE en el contexto proporcionado. Si no encuentras información suficiente, indica que no puedes responder completamente.

# Interacción
- Cuando respondas se preciso y relevante. Nada de divagar.
- Mantén la coherencia, que se entienda todo a la primera.
- Adapta tu tono al estilo de nuestra empresa, profesional pero cercano.
- Usa tú personalidad, no eres un asistente genérico, eres auténtico y genuino.

# Formato de entrega
Cada respuesta debe tener lo siguiente:
- Saludo personalizado
- Confirmación de que entendiste el problema
- Solución paso a paso si es necesario
- Una pregunta de seguimiento. ¿Fue útil mi respuesta?
- Un cierre que invite a volver. Queremos clientes fieles
- Firma como asiste virtual IA, Departamento de Informática

# Ejemplos

**Ejemplo 1:**

1. Saludo: "¡Hola [Nombre del Cliente]! Espero que estés teniendo un excelente día."
2. Confirmación: "Entiendo que tienes un problema con [Descripción del Problema]."
3. Solución: "Aquí te muestro cómo resolverlo: [Pasos detallados]."
4. Seguimiento: "¿Esta información fue de ayuda para ti?"
5. Cierre: "Gracias por confiar en nosotros. ¡Espero verte pronto! 😊"
6. Firma: "Tu asistente virtual IA, Departamento de Informática."

# Notas

- Reporta cualquier limitación en caso de incongruencias en los datos proporcionados.
- Evita frases que hagan referencia explícita al basarte en el contexto proporcionado.
"""

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
            #full_prompt += f"Historial de conversación:\n{conversation_history}\n\n"
            full_prompt += f"Historial:\n{conversation_history}\n\n"
        
        if context:
            #full_prompt += f"Contexto de documentos relevantes:\n{context}\n\n"
            full_prompt += f"Contexto:\n{context}\n\n"
        
        #full_prompt += f"Pregunta actual: {prompt}"
        full_prompt += f"Pregunta: {prompt}"
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 600,
            "messages": [
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            "system": system_prompt,
            "temperature": 0.4,
            "top_p": 0.9
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
    metadata: Optional[dict] = {}

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
            metadatas = [doc.metadata for doc in documents]
            
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            return {"status": "Documentos añadidos exitosamente"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
    def delete_documents_by_name(self, name: str):
        self.collection.delete(
            where={"source": name}
        )
    
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
async def add_excel_documents(documents: List[UploadFile]):
    for document in documents:
        bytesExcel = await document.read()
        # Procesar archivo Excel
        xls = pd.ExcelFile(bytesExcel)
        general_documents = []
        all_documents = []
        all_metadatas = []
        all_ids = []

        # Procesar cada hoja del Excel
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            
            # Convertir cada fila a un documento de texto
            for index, row in df.iterrows():
                # Convertir la fila a texto, omitiendo valores NaN
                document_text = " ".join([
                    f"{col}: {str(value)}" 
                    for col, value in row.items() 
                    if pd.notna(value)
                ])
                
                # Generar metadatos
                metadata = {
                    'source': document.filename,
                    'sheet': sheet_name,
                    'row_id': index
                }

                doc = {
                    'id': f"doc_{sheet_name}_{index}",
                    'text': document_text,
                    'metadata': metadata
                }

                doc2 = DocumentInput(**doc)
                
                general_documents.append(doc2)
    return rag_chatbot.add_documents(general_documents)

@app.post("/delete_documents_by_name")
async def delete_documents_by_name(documentName: str):
    return rag_chatbot.delete_documents_by_name(documentName)

@app.post("/query")
def query_chatbot(query: QueryInput):
    return rag_chatbot.query_documents(query)

# Iniciar el servidor
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)