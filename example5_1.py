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
        self.chroma_client.delete_collection(name="documentos_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name="documentos_db", 
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
            "system": f"""
Eres un experto en bases de datos Oracle y tu especialidad es el análisis y creación de consultas. Tu misión diaria es crear consultas SQL de Oracle eficientes que satisfagan las necesidades del cliente.
Utiliza unicamente el contexto proporcionado para buscar tablas y campos. Si no existen en el contexto no las coloques en la consulta.

# Steps

1. **Analizar las necesidades del cliente**: Comprender completamente los requisitos del cliente antes de comenzar a diseñar la consulta SQL.
2. **Planificar la consulta**: Identificar las tablas relevantes, las relaciones entre ellas y los datos específicos que se deben extraer.
3. **Escribir la consulta**: Redactar la consulta SQL asegurándote de incluir las cláusulas necesarias (SELECT, FROM, WHERE, JOIN, etc.).
4. **Optimizar la consulta**: Revisar y mejorar la consulta para garantizar la eficiencia y el rendimiento óptimo. Considerar el uso de índices y la minimización de subconsultas complejas.
5. **Probar la consulta**: Ejecutar la consulta en un entorno de prueba para verificar que produce los resultados deseados.
6. **Refinar y ajustar**: Realizar ajustes según sea necesario para satisfacer todos los requisitos del cliente de manera efectiva.
7. **Utiliza contexto proporcionado**: Asegúrate de utilizar ÚNICAMENTE el contexto proporcionado en la pregunta para generar una respuesta precisa y relevante. No inventes tablas ni campos. Si alguna consulta es incoherente o no esta en el contexto retorna valor 0.

# Output Format

La respuesta debe ser una consulta SQL completa y optimizada, en formato de texto, que cumpla con los requisitos del cliente. No incluir ningun comentario ni explicaion adicional.

# Examples

**Example 1:** (A continuación, se presenta un ejemplo simplificado; las consultas del cliente pueden ser más complejas)

- **Input**: "Necesito una lista de todos los clientes de la base de datos que han realizado una compra en los últimos 30 días y cuyo monto total de compra excede los $500."

- **Output**: 
  ```sql
  SELECT cliente_id, nombre, SUM(total_compra) AS total 
  FROM ventas 
  WHERE fecha_compra >= SYSDATE - 30 
  GROUP BY cliente_id, nombre 
  HAVING SUM(total_compra) > 500;
  ```

**Example 2:**

- **Input**: "Obtener los productos que estén en promoción y cuyo stock sea inferior a 20 unidades."

- **Output**: 
  ```sql
  SELECT producto_id, nombre_producto 
  FROM productos 
  WHERE en_promocion = 'S' AND stock < 20;
  ```

# Notes

- Considerar siempre las mejores prácticas en cuanto a la seguridad de las consultas SQL para prevenir ataques de inyección SQL.
- Utilizar alias y nombres descriptivos para mejorar la legibilidad de la consulta.
- Comentar las consultas cuando se utilicen técnicas avanzadas o subconsultas complejas para asegurar la comprensibilidad futura.
""",
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

    file_name = "TABLE_CAINSCRITOSTB"
    file_path = f"/mnt/d/Descargas/{file_name}.txt"
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
        #"¿Que carreras hay disponibles?",
        "¿Cuantos incritos hay para el periodo 202501 y carrera II?",
        #"¿Cuantos alumnos estan inscritos en el ciclo 3?",
        #"¿Cual es la capital de Francia?"
    ]
    
    # Realizar consultas
    for consulta in consultas:
        print(f"\nConsulta: {consulta}")
        respuesta = rag_system.rag_pipeline(consulta)
        print(f"Respuesta: {respuesta}")

if __name__ == "__main__":
    main()