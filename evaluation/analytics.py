import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
import csv

# Robuste Imports mit Fehlerbehandlung
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    RAGAS_AVAILABLE = True
except ImportError as e:
    print(f"RAGAS Import Fehler: {e}")
    RAGAS_AVAILABLE = False

try:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from datasets import Dataset
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"LangChain Import Fehler: {e}")
    LANGCHAIN_AVAILABLE = False

try:
    from config.settings import OPENAI_API_KEY
    from database.datastore import ChatDataStore
    from database.vectorstore import VectorStore
    from core.retrieval import create_qa_chain, query_knowledge_base
    LOCAL_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Lokale Import Fehler: {e}")
    LOCAL_IMPORTS_AVAILABLE = False

class RAGASEvaluator:
    def __init__(self):
        """Initialisiert den RAGAS Evaluator mit Fehlerbehandlung"""
        
        # Prüfe ob alle Imports verfügbar sind
        if not RAGAS_AVAILABLE:
            raise Exception("RAGAS nicht verfügbar. Installieren Sie: pip install ragas")
        
        if not LANGCHAIN_AVAILABLE:
            raise Exception("LangChain nicht verfügbar. Installieren Sie: pip install langchain-openai datasets")
            
        if not LOCAL_IMPORTS_AVAILABLE:
            raise Exception("Lokale Module nicht verfügbar. Prüfen Sie Pfade und Konfiguration.")
        
        # Prüfe API Key
        if not OPENAI_API_KEY or OPENAI_API_KEY == "":
            raise Exception("OPENAI_API_KEY nicht gesetzt")
        
        try:
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo", 
                openai_api_key=OPENAI_API_KEY,
                temperature=0
            )
            self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        except Exception as e:
            raise Exception(f"OpenAI Initialisierung fehlgeschlagen: {e}")
        
        try:
            self.datastore = ChatDataStore()
            self.vectorstore = VectorStore()
        except Exception as e:
            raise Exception(f"Datenbank Initialisierung fehlgeschlagen: {e}")
        
        # Alle Metriken - jetzt mit reference und ground_truth verfügbar
        self.all_metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
        
       # Test Cases mit question, expected_answer UND reference für TechFlow Solutions
        self.test_cases = [
    {
        "question": "Was ist die Vision von TechFlow Solutions?",
        "expected_answer": "Die Vision von TechFlow Solutions ist es, Europas führender Anbieter für intelligente Workflow-Automatisierung zu werden. Die Mission ist es, komplexe Geschäftsprozesse durch intuitive Software-Lösungen zu vereinfachen.",
        "reference": ["confluence"]
    },
    {
        "question": "Wie läuft der Bewerbungsprozess bei TechFlow ab?",
        "expected_answer": "Der Bewerbungsprozess umfasst 5 Schritte: 1) Online-Bewerbung über das Portal, 2) HR-Screening (30 Min.), 3) Technical Interview mit Team Lead (60 Min.), 4) Coding Challenge (Take-Home, 2-3 Stunden), 5) Final Interview mit CTO und Team (90 Min.).",
        "reference": ["pdf"]
    },
    {
        "question": "Welche IT-Sicherheitsrichtlinien gelten bei TechFlow?",
        "expected_answer": "IT-Sicherheitsrichtlinien umfassen: Passwörter mindestens 12 Zeichen mit Zwei-Faktor-Authentifizierung, regelmäßige Sicherheitsupdates und Patches, keine Installation privater Software auf Firmengeräten, USB-Sticks nur mit Genehmigung.",
        "reference": ["pdf"]
    },
    {
        "question": "Wer ist der CTO von TechFlow Solutions und wie kann ich ihn kontaktieren?",
        "expected_answer": "Der CTO ist Dr. Michael Schmidt. Er ist erreichbar unter michael.schmidt@techflow-solutions.de, Telefon +49 30 555-0111, Mobil +49 175 555-2011, Standort Berlin Büro 3.02, Slack @michael.schmidt.",
        "reference": ["confluence"]
    },
    {
        "question": "Was sind die Hauptziele des CloudSync Projekts für Q3-Q4 2025?",
        "expected_answer": "Die CloudSync-Ziele umfassen: 10.000 aktive Nutzer bis Ende Q4, API Response Time unter 500ms, 50+ Enterprise-Kunden gewinnen, $2M ARR Zielwert erreichen. Das Projekt entwickelt eine SaaS-Plattform für automatisierte Dokumentensynchronisation zwischen Cloud-Diensten.",
        "reference": ["miro"]
    },
    {
        "question": "Welche Benefits bietet TechFlow seinen Mitarbeitern?",
        "expected_answer": "TechFlow bietet: Betriebliche Krankenversicherung, Jobticket für öffentliche Verkehrsmittel, kostenlose Getränke und wöchentlicher Obstkorb, Firmenevents und Team-Building, Sabbatical-Möglichkeiten nach 3 Jahren, jährliches Weiterbildungsbudget von 2.000€, flexible Arbeitszeiten und Remote-Work bis 3 Tage/Woche.",
        "reference": ["confluence", "pdf"]
    },
    {
        "question": "Wie ist das Onboarding in den ersten 30 Tagen strukturiert?",
        "expected_answer": "Die ersten 30 Tage umfassen: Woche 1 - Administrative Einrichtung, IT-Equipment, Buddy-Meeting, CEO-Welcome-Session. Woche 2 - Entwicklungsumgebung einrichten, Technologie-Deep-Dive, GitHub-Zugang. Woche 3-4 - Erste Bug-Fixes, Code-Reading, Pair-Programming, erste User Story. Abschluss mit 30-Tage-Feedback-Gespräch.",
        "reference": ["pdf"]
    },
    {
        "question": "Welche Compliance-Richtlinien gelten für Geschenke und Korruption?",
        "expected_answer": "Anti-Korruptions-Richtlinien: Jede Form von Korruption und Bestechung ist untersagt. Geschenke bis 50€ sind erlaubt, Geschäftsessen bis 100€ pro Person zulässig, alle Geschenke über 25€ müssen dokumentiert werden, keine Geschenke in Vergabeverfahren.",
        "reference": ["pdf"]
    },
    {
        "question": "Wer ist für das CloudSync Projekt verantwortlich und wie kann ich die Teammitglieder erreichen?",
        "expected_answer": "Projektleiter ist Elena Rodriguez (VP Product Management), Tech Lead David Kim (Backend), Frontend Lead Anna Schneider. Elena ist erreichbar unter elena.rodriguez@techflow-solutions.de, David unter david.kim@techflow-solutions.de, Anna unter anna.schneider@techflow-solutions.de.",
        "reference": ["miro", "confluence"]
    },
    {
        "question": "Wie funktioniert das API Monitoring System und was passiert bei kritischen Alerts?",
        "expected_answer": "Das Monitoring-System nutzt AWS CloudWatch, New Relic APM, Grafana und PagerDuty. Bei kritischen Alerts (API Response Time >5000ms, Fehlerrate >15%, kompletter Ausfall) erfolgt sofortige Benachrichtigung des On-call Engineers, DevOps Lead und CTO. Erste Maßnahmen: PagerDuty-Alert bestätigen, AWS Status prüfen, Incident Bridge Call initiieren.",
        "reference": ["pdf"]
    }
]
    
    def evaluate_rag_system(self):
        """
        Führt die vollständige RAGAS-Evaluation mit allen 6 Metriken durch
        
        Returns:
            dict: Evaluationsergebnisse
        """
        try:
            print("🔍 Starte vollständige RAGAS Evaluation...")
            
            # Prüfe Vectorstore
            retriever = self.vectorstore.get_retriever()
            if not retriever:
                raise Exception("Kein Retriever verfügbar. Bitte laden Sie zuerst Dokumente.")
            
            print(f"✅ Retriever verfügbar")
            print(f"🎯 Evaluiere mit {len(self.test_cases)} Test Cases...")
            
            # Generiere Antworten für alle Test Cases
            print("🔄 Generiere Antworten...")
            qa_chain = create_qa_chain(retriever)
            
            generated_answers = []
            retrieved_contexts = []
            
            for i, case in enumerate(self.test_cases):
                print(f"  📝 Verarbeite Case {i+1}/{len(self.test_cases)}: {case['question'][:50]}...")
                try:
                    result = query_knowledge_base(qa_chain, case['question'])
                    generated_answers.append(result['answer'])
                    
                    # Kontext für diese Frage abrufen
                    docs = retriever.get_relevant_documents(case['question'])
                    contexts = [doc.page_content for doc in docs]
                    retrieved_contexts.append(contexts)
                    
                except Exception as e:
                    print(f"    ⚠️ Fehler bei Case {i+1}: {str(e)}")
                    generated_answers.append(f"Fehler: {str(e)}")
                    retrieved_contexts.append([])
            
            print("🔄 Erstelle Dataset...")
            dataset = Dataset.from_dict({
                'question': [case['question'] for case in self.test_cases],
                'answer': generated_answers,
                'contexts': retrieved_contexts,
                'ground_truth': [case['expected_answer'] for case in self.test_cases],
                'reference': ['; '.join(case['reference']) for case in self.test_cases]
            })
            
            print("🧠 Starte RAGAS Evaluation mit allen 6 Metriken...")
            results = evaluate(
                dataset=dataset,
                metrics=self.all_metrics,
                llm=self.llm,
                embeddings=self.embeddings
            )
            
            print("✅ Vollständige Evaluation abgeschlossen")
            
            return {
                'timestamp': datetime.now().isoformat(),
                'evaluation_type': 'complete_evaluation',
                'num_samples': len(self.test_cases),
                'metrics': {
                    metric.name: {
                        'score': float(results[metric.name]),
                        'description': self._get_metric_description(metric.name)
                    }
                    for metric in self.all_metrics
                    if metric.name in results
                },
                'test_cases': self.test_cases,
                'generated_answers': generated_answers
            }
            
        except Exception as e:
            print(f"❌ Detaillierter Fehler: {str(e)}")
            raise Exception(f"Fehler bei der RAGAS-Evaluation: {str(e)}")
    
    def _get_metric_description(self, metric_name):
        """Gibt Beschreibungen für Metriken zurück"""
        descriptions = {
            'faithfulness': 'Misst ob die Antwort durch den Kontext gestützt wird',
            'answer_relevancy': 'Misst wie relevant die Antwort zur Frage ist',
            'context_precision': 'Misst die Präzision des abgerufenen Kontexts',
            'context_recall': 'Misst ob der relevante Kontext abgerufen wurde',
            'answer_similarity': 'Semantische Ähnlichkeit zur erwarteten Antwort',
            'answer_correctness': 'Korrektheit verglichen mit erwarteter Antwort'
        }
        return descriptions.get(metric_name, 'Keine Beschreibung verfügbar')
    
    def export_results_to_csv(self, results):
        """
        Exportiert Evaluationsergebnisse als CSV ohne Rating.
        
        Args:
            results (dict): Evaluationsergebnisse
            
        Returns:
            str: Pfad zur Haupt-CSV-Datei
        """
        try:
            # Erstelle evaluation_exports Verzeichnis
            export_dir = "evaluation_exports"
            os.makedirs(export_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            eval_type = results['evaluation_type']
            
            # 1. Metriken-Übersicht (ohne Rating)
            metrics_file = os.path.join(export_dir, f"ragas_metrics_{eval_type}_{timestamp}.csv")
            
            with open(metrics_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                writer.writerow(['Timestamp', 'Evaluation_Type', 'Samples', 'Metric', 'Score'])
                
                for metric_name, metric_info in results['metrics'].items():
                    score = metric_info['score']
                    
                    writer.writerow([
                        results['timestamp'],
                        eval_type,
                        results['num_samples'],
                        metric_name,
                        f"{score:.4f}"
                    ])
            
            # 2. Detailvergleich
            detail_file = os.path.join(export_dir, f"test_case_comparison_{timestamp}.csv")
            
            with open(detail_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Question', 'Expected_Answer', 'Generated_Answer', 'Reference_Docs', 'Question_Length', 'Expected_Length', 'Generated_Length'])
                
                for i, case in enumerate(results['test_cases']):
                    generated = results['generated_answers'][i] if i < len(results['generated_answers']) else ""
                    reference_str = '; '.join(case['reference'])
                    
                    writer.writerow([
                        case['question'],
                        case['expected_answer'],
                        generated,
                        reference_str,
                        len(case['question']),
                        len(case['expected_answer']),
                        len(generated)
                    ])
            
            # 3. Summary Report
            summary_file = os.path.join(export_dir, f"evaluation_summary_{eval_type}_{timestamp}.txt")
            self._create_summary_report(results, summary_file)
            
            print(f"✅ CSV Export erfolgreich: {metrics_file}")
            return metrics_file
            
        except Exception as e:
            raise Exception(f"CSV Export Fehler: {str(e)}")
    
    def _create_summary_report(self, results, filepath):
        """Erstellt einen zusammenfassenden Textbericht"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("# RAGAS Complete Evaluation Summary Report\n")
            f.write(f"Generated: {results['timestamp']}\n")
            f.write(f"Evaluation Type: {results['evaluation_type']}\n")
            f.write(f"Test Cases evaluated: {results['num_samples']}\n\n")
            
            f.write("## All 6 RAGAS Metric Scores\n")
            for metric_name, metric_info in results['metrics'].items():
                score = metric_info['score']
                f.write(f"- {metric_name}: {score:.4f}\n")
            
            avg_score = sum(info['score'] for info in results['metrics'].values()) / len(results['metrics'])
            f.write(f"\n## Overall Performance\n")
            f.write(f"Average Score: {avg_score:.4f}\n")
            
            f.write(f"\n## Evaluation Details\n")
            f.write("- Complete evaluation with ground_truth and reference data\n")
            f.write("- All 6 RAGAS metrics included\n")
            f.write("- Test cases cover company-specific scenarios\n")
            f.write("- Reference documents specified for context evaluation\n")