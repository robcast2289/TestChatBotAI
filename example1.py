import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import chromadb
from typing import List, Dict

class RestrictedInfoChatbot:
    def __init__(self, 
                 model_name: str = "deepset/roberta-base-squad2",
                 database_path: str = "./knowledge_base"):
        # Cargar modelo de pregunta-respuesta
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        
        # Configurar base de datos vectorial
        self.client = chromadb.PersistentClient(path=database_path)
        self.client.delete_collection(name="knowledge_base")
        self.collection = self.client.create_collection(name="knowledge_base")
    
    def add_document(self, document: str, metadata: Dict[str, str]):
        """Agregar documento a la base de conocimientos"""
        vector = self._generate_embedding(document)
        self.collection.add(
            ids=["id1"],
            embeddings=[vector],
            documents=[document],
            metadatas=[metadata]
        )
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generar embedding para un texto"""
        # Implementación de embedding (puede usar transformers o otros modelos)
        # Este es un placeholder
        return [0.0] * 768
    
    def query_with_restrictions(
        self, 
        query: str, 
        allowed_categories: List[str] = None
    ) -> str:
        """Consultar base de datos con restricciones de categoría"""
        # Buscar documentos relevantes
        query_embedding = self._generate_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        
        # Filtrar por categorías permitidas
        filtered_docs = [
            doc for doc, metadata in zip(results['documents'][0], results['metadatas'][0])
            if not allowed_categories or metadata.get('category') in allowed_categories
        ]
        
        # Procesar pregunta con modelo de QA
        context = " ".join(filtered_docs)
        inputs = self.tokenizer(query, context, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits)
        
        answer = self.tokenizer.decode(
            inputs["input_ids"][0][answer_start:answer_end+1]
        )
        
        return answer
    
    def update_knowledge_base(self, new_documents: List[Dict]):
        """Actualizar base de conocimientos"""
        for doc in new_documents:
            self.add_document(
                document=doc['text'], 
                metadata={'category': doc.get('category', 'general')}
            )

# Ejemplo de uso
chatbot = RestrictedInfoChatbot()

# Agregar documentos
chatbot.add_document(
    "Python es un lenguaje de programación de alto nivel creado por Guido van Rossum en 1991", 
    {"category": "tecnologia"}
)

# Consultar con restricciones
respuesta = chatbot.query_with_restrictions(
    "¿Qué es Python?", 
    allowed_categories=["tecnologia"]
)
print(respuesta)