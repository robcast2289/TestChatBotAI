import chromadb
import boto3
import pandas as pd
from boto3 import Session
import json
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class BedrockRAGSystem:
    def __init__(
        self, 
        embedding_model: str = 'all-MiniLM-L6-v2',
        region_name: str = 'us-east-1'
    ):
        """
        Inicializar sistema RAG con ChromaDB y AWS Bedrock
        """
        # Configurar ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./Storage/knowledge_base")
        #self.chroma_client.delete_collection(name="documentos_tecnicos")
        self.collection = self.chroma_client.get_or_create_collection(
            name="documentos_tecnicos", 
            metadata={"hnsw:space": "cosine"}
        )
        
        # Modelo de embeddings
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Cliente AWS Bedrock
        self.sesion = Session(
            aws_access_key_id='',
            aws_secret_access_key='',
        )
        self.bedrock_runtime = self.sesion.client(
            service_name='bedrock-runtime', 
            region_name=region_name
        )
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Agregar documentos a la base de conocimientos
        """
        for doc in documents:
            # Generar embedding
            embedding = self.embedding_model.encode(doc['text']).tolist()
            
            # Agregar a ChromaDB
            self.collection.add(
                ids=[doc['id']],
                documents=[doc['text']],
                embeddings=[embedding],
                metadatas=[doc.get('metadata', {})]
            )
    
    def semantic_search(self, query: str, top_k: int = 3) -> List[str]:
        """
        Búsqueda semántica en la base de conocimientos
        """
        # Generar embedding para la consulta
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Realizar búsqueda semántica
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        return results['documents'][0]
    
    def generate_response_bedrock(
        self, 
        query: str, 
        context: List[str]
    ) -> str:
        """
        Generar respuesta usando Bedrock (Llama 3)
        """
        # Formatear contexto y consulta
        context_str = "\n".join(context)
        
        # prompt = f"""
        # Contexto:
        # {context_str}
        # 
        # Pregunta: {query}
        # 
        # Responde basándote en el contexto proporcionado. 
        # Si no encuentras la respuesta en el contexto, 
        # indica que no tienes suficiente información.
        # 
        # Respuesta:
        # """
        # 
        # # Configuración para Llama 3 en Bedrock
        # body = json.dumps({
        #     "prompt": prompt,
        #     "max_tokens_to_sample": 300,
        #     "temperature": 0.7,
        #     "top_p": 0.9
        # })

        # Estructura de prompt para Claude 3
        messages = [
            #{
            #    "role": "system",
            #    "content": "Eres un asistente de IA útil y preciso. Responde basándote estrictamente en el contexto proporcionado."
            #},
            {
                "role": "user",
                "content": f"""
                Contexto:
                {context_str}
                
                Pregunta: {query}
                """
                #Responde basándote ÚNICAMENTE en el contexto proporcionado, sin mencionar que se basa en el contexto. 
                #Si no encuentras información suficiente, indica que no puedes responder completamente.
                #"""
            }
        ]
        
        # Configuración para Claude 3 en Bedrock
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 300,
            "messages": messages,
            #"system": "Eres un asistente de IA útil y preciso. Responde basándote estrictamente en el contexto proporcionado, omite frases como 'Según el contexto proporcionado' u otras que haga alucion al contexto. Todas las repuestas deben ser en español",
            "system": f"""Eres un asistente virtual de inteligencia artificial, trabajador del departamento de Informática de la Universidad Galileo, eres líder mundial en la atención al cliente para el personal administrativo de la Universidad Galileo. Tu misión diaria es responder consultas, resolver problemas, proporcionar información precisa y gestionar dudas.

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
- Usa tú personalidad, no eres una asistente genérico, eres auténtico y genuino.

# Formato de entrega
Cada respuesta debe ser entregado en formato markdown y tener lo siguiente:
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
- Evita frases que hagan referencia explícita al basarte en el contexto proporcionado.""",
            "temperature": 0.4,
            "top_p": 0.9
        })
        
        # Invocar modelo Llama 3 en Bedrock
        response = self.bedrock_runtime.invoke_model(
            #modelId="meta.llama3-8b-instruct-v1:0",
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            body=body,
            contentType="application/json",
            accept="application/json"
        )
        
        # Procesar respuesta
        response_body = json.loads(response['body'].read())
        #print(response_body)
        return response_body['content'][0]['text']
    
    def rag_pipeline(self, query: str) -> str:
        """
        Pipeline completo de Recuperación y Generación Aumentada
        """
        # Recuperar documentos relevantes
        relevant_docs = self.semantic_search(query,1000)
        
        # Generar respuesta usando Bedrock
        response = self.generate_response_bedrock(query, relevant_docs)
        
        return response
    
    def process_excel_documents(self,file_path):
        """
        Procesa documentos de Excel para embeddings en ChromaDB
        """
        # Leer todos los sheets del Excel
        xls = pd.ExcelFile(file_path)
        general_documents = []
        all_documents = []
        all_metadatas = []
        all_ids = []

        # Procesar cada hoja del Excel
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
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
                    'source': file_path,
                    'sheet': sheet_name,
                    'row_id': index
                }

                doc = {
                    'id': f"doc_{sheet_name}_{index}",
                    'text': document_text,
                    'metadata': metadata
                }
                
                general_documents.append(doc)
                # all_documents.append(document_text)
                # all_metadatas.append(metadata)
                # all_ids.append(f"doc_{sheet_name}_{index}")

        #return all_documents, all_metadatas, all_ids
        return general_documents
    

    def procesar_archivo(self,ruta_archivo):
        with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
            texto = archivo.read()
            # Dividir texto en chunks
            chunks = self.dividir_texto_en_chunks(texto)
            return chunks, texto

    def dividir_texto_en_chunks(self, texto, tamano_chunk=500):
        # Dividir texto en chunks de tamaño específico
        chunks = [texto[i:i+tamano_chunk] for i in range(0, len(texto), tamano_chunk)]
        return chunks

def main():
    # Crear instancia del sistema RAG
    rag_system = BedrockRAGSystem()
    
    # Documentos de ejemplo
    # documentos = [
    #     {
    #         'id': 'doc1',
    #         'text': 'Python es un lenguaje de programación de alto nivel, interpretado y de propósito general.',
    #         'metadata': {'categoria': 'programacion'}
    #     },
    #     {
    #         'id': 'doc2', 
    #         'text': 'Machine Learning permite a las computadoras aprender de datos sin ser programadas explícitamente.',
    #         'metadata': {'categoria': 'inteligencia-artificial'}
    #     }
    # ]

     # Ruta al archivo Excel
    # excel_file_path = '/mnt/d/Descargas/datos_inscritos_example.xls'
    # documentos = rag_system.process_excel_documents(excel_file_path)

    file_path = "/mnt/d/Descargas/inscritos_2025_2.txt"
    file_name = "inscritos_2025_2"
    chunks, documentos = rag_system.procesar_archivo(file_path)
    doc = {
             'id': f"doc_{file_name}_{1}",
             'text': documentos,
             'metadata': {
                     'source': file_path,
                     'sheet': file_name,
                     'row_id': 1
                 }
         }
    # for i, chunk in enumerate(chunks):
    #     metadata = {
    #                 'source': file_path,
    #                 'sheet': file_name,
    #                 'row_id': i
    #             }
    #     doc = {
    #         'id': f"doc_{file_name}_{i}",
    #         'text': chunk,
    #         'metadata': metadata
    #     }
    #     rag_system.add_documents(doc)
    
    # Agregar documentos
    rag_system.add_documents([doc])
    
    # Realizar consulta
    #consulta = "¿Qué es Python?"
    #respuesta = rag_system.rag_pipeline(consulta)
    #print(f"Consulta: {consulta}")
    #print(f"Respuesta: {respuesta}")

    # Ejemplos de consultas
    consultas = [
        # "¿Qué es Python?",
        # "Explícame Machine Learning",
        # "¿Cómo funcionan los modelos de lenguaje?"
        "¿Que carreras hay disponibles?",
        "¿Cuantos incritos hay?",
        "¿Cuantos alumnos estan inscritos en el ciclo 3?"
    ]
    
    # Realizar consultas
    for consulta in consultas:
        print(f"\nConsulta: {consulta}")
        respuesta = rag_system.rag_pipeline(consulta)
        print(f"Respuesta: {respuesta}")

if __name__ == "__main__":
    main()