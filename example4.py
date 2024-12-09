import chromadb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

class LlamaChromaRAG:
    def __init__(
        self, 
        embedding_model: str = 'all-MiniLM-L6-v2', 
        llama_model: str = 'meta-llama/Llama-3.2-1B'
    ):
        """
        Inicializar RAG con ChromaDB, embeddings y Llama 3
        
        Args:
            embedding_model: Modelo para generar embeddings
            llama_model: Modelo Llama para generación de respuestas
        """
        # Configurar ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./knowledge_base")
        self.collection = self.chroma_client.get_or_create_collection(
            name="documentos_tecnologia", 
            metadata={"hnsw:space": "cosine"}
        )
        
        # Modelo de embeddings
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Modelo Llama 3
        self.tokenizer = AutoTokenizer.from_pretrained(llama_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            llama_model,
            torch_dtype=torch.float16,
            device_map='auto'
        ).to('cuda')
        
        # Pipeline de generación
        self.generator = pipeline(
            'text-generation', 
            model=self.model, 
            tokenizer=self.tokenizer,
            max_new_tokens=200
        )
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Agregar documentos a la base de conocimientos
        
        Args:
            documents: Lista de documentos con texto y metadatos
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
        
        Args:
            query: Consulta de búsqueda
            top_k: Número de documentos a recuperar
        
        Returns:
            Lista de documentos más relevantes
        """
        # Generar embedding para la consulta
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Realizar búsqueda semántica
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        return results['documents'][0]
    
    def generate_response(
        self, 
        query: str, 
        context: List[str]
    ) -> str:
        """
        Generar respuesta usando contexto recuperado
        
        Args:
            query: Pregunta del usuario
            context: Documentos de contexto recuperados
        
        Returns:
            Respuesta generada por Llama 3
        """
        # Formatear contexto y consulta
        context_str = "\n".join(context)
        prompt = f"""
        Contexto:
        {context_str}
        
        Pregunta: {query}
        
        Responde basándote en el contexto proporcionado. 
        Si no encuentras la respuesta en el contexto, 
        indica que no tienes suficiente información.
        
        Respuesta:
        """
        
        # Generar respuesta
        response = self.generator(
            prompt, 
            max_new_tokens=50,
            do_sample=True,
            temperature=0.2,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return response[0]['generated_text'].split('Respuesta:')[-1].strip()
    
    def rag_pipeline(self, query: str) -> str:
        """
        Pipeline completo de Recuperación y Generación Aumentada
        
        Args:
            query: Consulta del usuario
        
        Returns:
            Respuesta generada
        """
        # Recuperar documentos relevantes
        relevant_docs = self.semantic_search(query,1)
        
        # Generar respuesta
        response = self.generate_response(query, relevant_docs)
        
        return response

def main():
    # Ejemplo de uso
    rag_system = LlamaChromaRAG()
    
    # Documentos de ejemplo
    documentos = [
        {
            'id': 'doc1',
            'text': 'Python es un lenguaje de programación de alto nivel, interpretado y de propósito general.',
            'metadata': {'categoria': 'programacion'}
        },
        {
            'id': 'doc2', 
            'text': 'Machine Learning permite a las computadoras aprender de datos sin ser programadas explícitamente.',
            'metadata': {'categoria': 'inteligencia-artificial'}
        },
        {
            'id': 'doc3',
            'text': 'Los modelos de lenguaje como Llama 3 usan transformers y aprendizaje profundo para generar texto.',
            'metadata': {'categoria': 'ia-generativa'}
        }
    ]
    
    # Agregar documentos
    rag_system.add_documents(documentos)
    
    # Ejemplos de consultas
    consultas = [
        "¿Qué es Python?",
        "Explícame Machine Learning",
        "¿Cómo funcionan los modelos de lenguaje?"
    ]
    
    # Realizar consultas
    for consulta in consultas:
        print(f"\nConsulta: {consulta}")
        respuesta = rag_system.rag_pipeline(consulta)
        print(f"Respuesta: {respuesta}")

if __name__ == "__main__":
    main()