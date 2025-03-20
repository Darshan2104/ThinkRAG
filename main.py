from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from llm import GroqLLM


class IRCoTSystem:
    def __init__(self, knowledge_base, my_llm, embedding_model="all-MiniLM-L6-v2"):
        self.knowledge_base = knowledge_base
        self.llm = my_llm 
        self.embedding_model = SentenceTransformer(embedding_model)
        
    def retrieve_initial(self, question: str, k: int = 3) -> List[Dict]:
        # Initial retrieval using the question
        question_embedding = self.embedding_model.encode([question])[0]
        doc_embeddings = np.array([doc["embedding"] for doc in self.knowledge_base])
        
        similarities = cosine_similarity([question_embedding], doc_embeddings)[0]
        top_indices = similarities.argsort()[-k:][::-1]
        
        return [self.knowledge_base[i] for i in top_indices]
    
    def retrieve_with_cot(self, cot_sentence: str, k: int = 2) -> List[Dict]:
        # Retrieval using the latest CoT sentence
        cot_embedding = self.embedding_model.encode([cot_sentence])[0]
        doc_embeddings = np.array([doc["embedding"] for doc in self.knowledge_base])
        
        similarities = cosine_similarity([cot_embedding], doc_embeddings)[0]
        top_indices = similarities.argsort()[-k:][::-1]
        
        return [self.knowledge_base[i] for i in top_indices]
    
    def generate_next_cot(self, question: str, paragraphs: List[Dict], cot_so_far: List[str]) -> str:
        # Construct prompt for generating the next CoT sentence
        prompt = self._construct_cot_prompt(question, paragraphs, cot_so_far)
        
        response = self.llm.generate_response(
            prompt=prompt,
            max_tokens=100,
            temperature=0.7
            )

        return response
    
    def _construct_cot_prompt(self, question: str, paragraphs: List[Dict], cot_so_far: List[str]) -> str:
        # Format paragraphs for the prompt
        formatted_paragraphs = ""
        for p in paragraphs:
            formatted_paragraphs += f"Title: {p['title']}\n{p['text']}\n\n"
        
        # Format CoT sentences so far
        cot_text = "\n".join(cot_so_far)
        
        # Construct the full prompt
        prompt = f"{formatted_paragraphs}Q: {question}\nA: {cot_text}\n"
        return prompt
    
    def answer_question(self, question: str, max_iterations: int = 5) -> str:
        # Initial retrieval
        paragraphs = self.retrieve_initial(question)
        cot_sentences = []
        
        # Interleave reasoning and retrieval
        for _ in range(max_iterations):
            # Generate next CoT sentence
            next_sentence = self.generate_next_cot(question, paragraphs, cot_sentences)
            cot_sentences.append(next_sentence)
            
            # Check if answer is found
            if "answer is:" in next_sentence.lower():
                break
                
            # Retrieve more paragraphs based on the latest CoT sentence
            new_paragraphs = self.retrieve_with_cot(next_sentence)
            paragraphs.extend(new_paragraphs)
            
        return "\n".join(cot_sentences)


if __name__ == "__main__":
    # Sample knowledge base (in a real system, this would be much larger)
    model_name="llama-3.3-70b-versatile"
    my_llm = GroqLLM(model_name)
    
    # It can be your pdf, website, wikipedia, or anything else.
    knowledge_base = [
        {
            "title": "IRCoT Framework Overview",
            "text": "The IRCoT framework represents a groundbreaking fusion of interleaved retrieval and chain-of-thought methodologies. This innovative approach creates a dynamic feedback loop between information gathering and reasoning processes. At its core, IRCoT begins with an initial query extraction phase. The system analyzes the user's question to identify key concepts and potential search terms. This preliminary analysis shapes the first retrieval iteration.",
            "embedding": None  # Will be populated by the embedding model
        },
        {
            "title": "Key Components of IRCoT",
            "text": "Key components of the IRCoT process include query understanding and decomposition, initial document retrieval, iterative reasoning steps, dynamic query refinement, evidence collection and synthesis, and continuous evaluation and adjustment. The magic happens in the continuous interplay between retrieval and reasoning. Each retrieved piece of information influences the next thought step, while each reasoning step guides subsequent retrieval actions.",
            "embedding": None
        },
        {
            "title": "IRCoT Implementation Challenges",
            "text": "Common challenges in IRCoT implementation include technical hurdles such as query expansion complexity, resource intensive processing, latency management, result coherence maintenance, and system scalability. Building an effective IRCoT system requires attention to optimization strategies. Performance can be enhanced through caching frequently accessed information, smart query decomposition, and parallel processing capabilities.",
            "embedding": None
        },
        {
            "title": "Prompting Techniques for IRCoT",
            "text": "Effective prompting forms the backbone of successful IRCoT implementation. Chain-of-Thought prompting must be carefully structured to guide both the retrieval and reasoning processes effectively. Core prompting principles include clear step sequencing, logical progression, context maintenance, adaptive refinement, error recovery, and result validation. The 'Let's think step-by-step' approach proves particularly effective when combined with retrieval operations.",
            "embedding": None
        },
        {
            "title": "IRCoT Performance Benefits",
            "text": "Using IRCoT with GPT3 substantially improves retrieval (up to 21 points) as well as downstream QA (up to 15 points) on four datasets: HotpotQA, 2WikiMultihopQA, MuSiQue, and IIRC. Similar substantial gains are observed in out-of-distribution (OOD) settings as well as with much smaller models such as Flan-T5-large without additional training. IRCoT reduces model hallucination, resulting in factually more accurate CoT reasoning.",
            "embedding": None
        },
        {
            "title": "RAG Overview",
            "text": "Retrieval-augmented generation (RAG) is a powerful combination of traditional LLMs with information retrieval systems. By accessing and incorporating relevant information from external sources, RAG models can produce more accurate and contextually relevant responses. RAG architecture means that you can constrain generative AI to your enterprise content sourced from vectorized documents and images.",
            "embedding": None
        },
        {
            "title": "Building AI Knowledge Bases",
            "text": "To build an AI knowledge base, start by defining your goals and scope. Then gather and preprocess data from relevant sources, including existing documents, FAQs, customer interactions, and other information. Preprocess the data by cleaning, organizing, and structuring it for AI analysis. Ensure data quality and accuracy to improve the effectiveness of your AI model. Select the right structure for organizing your content and implement appropriate AI models.",
            "embedding": None
        },
        {
            "title": "AI Knowledge Base Components",
            "text": "Key components of an AI knowledge base include machine learning models, natural language processing, and a data repository. Machine learning models enable the AI to learn from data, identify patterns, and make predictions with minimal human intervention. NLP allows AI to understand and interpret human language, which is essential for analyzing and responding to user queries. A centralized storage system keeps all the relevant data.",
            "embedding": None
        },
        {
            "title": "Amazon Bedrock Knowledge Bases",
            "text": "Amazon Bedrock Knowledge Bases is a fully managed capability with in-built session context management and source attribution that helps implement the entire RAG workflow from ingestion to retrieval and prompt augmentation. It automatically fetches data from sources such as Amazon S3, Confluence, Salesforce, SharePoint, or Web Crawler. Once the content is ingested, it converts it into blocks of text, the text into embeddings, and stores the embeddings in your vector database.",
            "embedding": None
        },
        {
            "title": "IRCoT vs Traditional RAG",
            "text": "Unlike traditional one-step retrieve-and-read approaches, IRCoT operates through an iterative process that alternates between extending CoT (reasoning) and expanding retrieved information. The system uses the question, previously collected paragraphs, and previously generated CoT sentences to generate the next reasoning step, then uses the last CoT sentence as a query to retrieve additional relevant paragraphs.",
            "embedding": None
        }
    ]    # Pre-compute embeddings for the knowledge base
    
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    for doc in knowledge_base:
        doc["embedding"] = embedding_model.encode(doc["text"])

    # Initialize the IRCoT system
    ircot = IRCoTSystem(knowledge_base, my_llm=my_llm)

    # Answer a multi-step question
    question = "How does IRCoT improve over traditional RAG approaches and what are its key components?"
    # question = "What is IRCoT in layman's term? Give me answer in simple language and in fewer words."
    answer = ircot.answer_question(question)
    
    print(f"Question: {question}\n{answer}")
