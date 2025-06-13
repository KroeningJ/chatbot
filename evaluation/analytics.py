import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_similarity,
    answer_correctness
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import Dataset
import pandas as pd
from datetime import datetime
from config.settings import OPENAI_API_KEY
from database.datastore import ChatDataStore
from database.vectorstore import VectorStore

class RAGASEvaluator:
    def __init__(self):
        """Initialisiert den RAGAS Evaluator"""
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            openai_api_key=OPENAI_API_KEY
        )
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        self.datastore = ChatDataStore()
        self.vectorstore = VectorStore()
        
        # Standard-Metriken
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            answer_similarity,
            answer_correctness
        ]
    
    def prepare_evaluation_data(self, qa_pairs=None, include_ground_truth=False):
        """
        Bereitet Daten für die Evaluation vor.
        
        Args:
            qa_pairs (list): Optionale Liste von Q&A-Paaren
            include_ground_truth (bool): Ob Ground Truth inkludiert werden soll
            
        Returns:
            Dataset: Vorbereiteter Dataset für RAGAS
        """
        # Hole Q&A-Paare aus der Datenbank, falls keine übergeben wurden
        if qa_pairs is None:
            qa_pairs = self.datastore.get_messages_for_evaluation()
        
        if not qa_pairs:
            raise ValueError("Keine Q&A-Paare für die Evaluation vorhanden")
        
        # Daten für RAGAS vorbereiten
        evaluation_data = {
            'question': [],
            'answer': [],
            'contexts': [],
            'ground_truth': []
        }
        
        retriever = self.vectorstore.get_retriever()
        
        for qa in qa_pairs:
            evaluation_data['question'].append(qa['question'])
            evaluation_data['answer'].append(qa['answer'])
            
            # Hole relevante Kontexte für die Frage
            if retriever:
                docs = retriever.get_relevant_documents(qa['question'])
                contexts = [doc.page_content for doc in docs]
                evaluation_data['contexts'].append(contexts)
            else:
                evaluation_data['contexts'].append([])
            
            # Ground Truth (falls verfügbar)
            if include_ground_truth and 'ground_truth' in qa:
                evaluation_data['ground_truth'].append(qa['ground_truth'])
            else:
                evaluation_data['ground_truth'].append(qa['answer'])  # Verwende Antwort als Fallback
        
        return Dataset.from_dict(evaluation_data)
    
    def evaluate_rag_system(self, dataset=None, custom_metrics=None):
        """
        Führt die RAGAS-Evaluation durch.
        
        Args:
            dataset (Dataset): Optionaler vorbereiteter Dataset
            custom_metrics (list): Optionale benutzerdefinierte Metriken
            
        Returns:
            dict: Evaluationsergebnisse
        """
        try:
            # Dataset vorbereiten, falls keiner übergeben wurde
            if dataset is None:
                dataset = self.prepare_evaluation_data()
            
            # Metriken auswählen
            metrics_to_use = custom_metrics if custom_metrics else self.metrics
            
            # Evaluation durchführen
            results = evaluate(
                dataset=dataset,
                metrics=metrics_to_use,
                llm=self.llm,
                embeddings=self.embeddings
            )
            
            # Ergebnisse formatieren
            formatted_results = {
                'timestamp': datetime.now().isoformat(),
                'num_samples': len(dataset),
                'scores': results,
                'metrics': {
                    metric.name: {
                        'score': float(results[metric.name]),
                        'description': getattr(metric, 'description', '')
                    }
                    for metric in metrics_to_use
                    if metric.name in results
                }
            }
            
            return formatted_results
            
        except Exception as e:
            raise Exception(f"Fehler bei der RAGAS-Evaluation: {e}")
    
    def evaluate_single_qa(self, question, answer, context=None):
        """
        Evaluiert ein einzelnes Q&A-Paar.
        
        Args:
            question (str): Die Frage
            answer (str): Die generierte Antwort
            context (list): Optionaler Kontext
            
        Returns:
            dict: Evaluationsergebnisse
        """
        # Kontext abrufen, falls nicht übergeben
        if context is None and self.vectorstore.get_retriever():
            retriever = self.vectorstore.get_retriever()
            docs = retriever.get_relevant_documents(question)
            context = [doc.page_content for doc in docs]
        
        # Dataset für einzelnes Q&A erstellen
        single_data = {
            'question': [question],
            'answer': [answer],
            'contexts': [context or []],
            'ground_truth': [answer]  # Verwende Antwort als Ground Truth
        }
        
        dataset = Dataset.from_dict(single_data)
        
        # Relevante Metriken für Einzelevaluation
        single_metrics = [
            answer_relevancy,
            faithfulness
        ]
        
        return self.evaluate_rag_system(dataset, single_metrics)
    
    def generate_evaluation_report(self, results):
        """
        Generiert einen Evaluationsbericht.
        
        Args:
            results (dict): Evaluationsergebnisse
            
        Returns:
            str: Formatierter Bericht
        """
        report = f"""
# RAG System Evaluation Report
Generated at: {results['timestamp']}
Number of samples evaluated: {results['num_samples']}

## Overall Scores
"""
        
        for metric_name, metric_info in results['metrics'].items():
            score = metric_info['score']
            description = metric_info.get('description', '')
            
            # Bewertung des Scores
            if score >= 0.8:
                rating = "Excellent"
            elif score >= 0.6:
                rating = "Good"
            elif score >= 0.4:
                rating = "Fair"
            else:
                rating = "Needs Improvement"
            
            report += f"\n### {metric_name.replace('_', ' ').title()}\n"
            report += f"- **Score**: {score:.3f} ({rating})\n"
            if description:
                report += f"- **Description**: {description}\n"
        
        # Empfehlungen basierend auf den Scores
        report += "\n## Recommendations\n"
        
        if 'faithfulness' in results['metrics'] and results['metrics']['faithfulness']['score'] < 0.7:
            report += "- **Improve Faithfulness**: The system is generating content not well-supported by the retrieved context. Consider improving retrieval quality or adjusting the generation prompt.\n"
        
        if 'answer_relevancy' in results['metrics'] and results['metrics']['answer_relevancy']['score'] < 0.7:
            report += "- **Improve Answer Relevancy**: Answers are not fully addressing the questions. Review and improve the prompt template.\n"
        
        if 'context_precision' in results['metrics'] and results['metrics']['context_precision']['score'] < 0.7:
            report += "- **Improve Context Precision**: Retrieved contexts contain irrelevant information. Consider adjusting retrieval parameters or chunk sizes.\n"
        
        return report
    
    def save_evaluation_results(self, results, filepath=None):
        """
        Speichert Evaluationsergebnisse.
        
        Args:
            results (dict): Evaluationsergebnisse
            filepath (str): Optionaler Dateipfad
            
        Returns:
            str: Pfad zur gespeicherten Datei
        """
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"evaluation_results_{timestamp}.json"
        
        # Erstelle Evaluationsordner, falls nicht vorhanden
        eval_dir = "evaluation_results"
        os.makedirs(eval_dir, exist_ok=True)
        filepath = os.path.join(eval_dir, filepath)
        
        import json
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generiere auch einen Textbericht
        report = self.generate_evaluation_report(results)
        report_path = filepath.replace('.json', '.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        
        return filepath