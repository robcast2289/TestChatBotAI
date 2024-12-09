import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Configurar ChromaDB
client = chromadb.PersistentClient(path="./knowledge_base"  # Directorio para guardar la base de datos
)

# Crear una colección en ChromaDB
collection = client.get_or_create_collection("llama_knowledge_base")

# Modelo de embeddings (SentenceTransformer o similar)
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Puedes usar otros modelos según tu necesidad

# Función para indexar datos en ChromaDB
def index_data(data):
    """
    data: Lista de diccionarios con 'id' y 'content'
    """
    for item in data:
        embedding = embedder.encode(item['content'])
        collection.add(
            embeddings=[embedding],
            documents=[item['content']],
            ids=[item['id']]
        )
    print("Datos indexados correctamente.")

# Datos de ejemplo
data_to_index = [
    {"id": "1", "content": "¿Qué es un modelo de lenguaje grande?"},
    {"id": "2", "content": "Explica el proceso de aprendizaje supervisado."},
    {"id": "3", "content": "¿Qué son las redes neuronales convolucionales?"}
]

# Indexar datos
index_data(data_to_index)

# Consultar datos con un modelo Llama 3
def query_llama(query, top_k=3):
    """
    query: Pregunta en lenguaje natural
    top_k: Número de resultados relevantes a recuperar
    """
    query_embedding = embedder.encode(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    # Inicializar modelo Llama 3
    llama = pipeline("text-generation", model="meta-llama/Llama-3.2-1B")
    
    print("Resultados relevantes encontrados:")
    for doc in results['documents'][0]:
        print(f"- {doc}")
    
    # Generar respuesta usando Llama 3
    prompt = f"Basado en estos documentos: {results['documents'][0]}, responde a la pregunta: {query}"
    response = llama(prompt, max_length=200, num_return_sequences=1)
    return response[0]['generated_text']

# Ejemplo de consulta
question = "¿Qué es un modelo de lenguaje grande?"
response = query_llama(question)
print("\nRespuesta generada por Llama 3:")
print(response)
