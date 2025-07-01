#!/usr/bin/env python3
"""
RAG Chatbot Evaluation Script - Testversion

Evaluiert den RAG-Chatbot mit einem Testfall und RAGAS-Metriken.

Verwendung:
    python analytics.py
"""

import os
import csv
import sys
from datetime import datetime

# Sicherstellen, dass das Hauptverzeichnis im Python-Pfad ist
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

print(f"🔍 Script-Verzeichnis: {current_dir}")
print(f"🔍 Projekt-Verzeichnis: {parent_dir}")
print(f"🔍 Python-Pfad erweitert um: {parent_dir}")

# Imports
try:
    print("📦 Lade externe Abhängigkeiten...")
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from datasets import Dataset
    print("✅ Externe Abhängigkeiten geladen")
    
    print("📦 Lade lokale Module...")
    from database.vectorstore import VectorStore
    from core.retrieval import create_qa_chain, query_knowledge_base
    from config.settings import OPENAI_API_KEY
    print("✅ Lokale Module geladen")
    
except ImportError as e:
    print(f"❌ Import-Fehler: {e}")
    print(f"🔍 Aktueller Python-Pfad:")
    for path in sys.path:
        print(f"   {path}")
    print("\nPrüfen Sie:")
    print("1. Sind alle externen Abhängigkeiten installiert?")
    print("   pip install ragas langchain langchain-openai langchain-community chromadb datasets")
    print("2. Existieren alle lokalen Module?")
    print(f"   - {parent_dir}/database/vectorstore.py")
    print(f"   - {parent_dir}/core/retrieval.py") 
    print(f"   - {parent_dir}/config/settings.py")
    sys.exit(1)

class RAGEvaluator:
    def __init__(self):
        """Initialisiert den RAG Evaluator"""
        print("🚀 Initialisiere RAG Evaluator...")
        
        # LLM und Embeddings für RAGAS
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            openai_api_key=OPENAI_API_KEY,
            temperature=0
        )
        self.embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        
        # Vectorstore laden (gleiche Logik wie im Chatbot)
        self.vectorstore = VectorStore()
        
        # RAGAS Metriken
        self.metrics = [
            faithfulness,
            answer_relevancy, 
            context_precision,
            context_recall
        ]
        
        # Testfall
        self.test_cases = [
              {
                "question": "Wie hoch ist das vertraglich geregelte Bruttomonatsgehalt von Jens Kröning und gibt es weitere Regelungen zu Boni?",
                "expected_answer": "Das Bruttomonatsgehalt von Jens Kröning beträgt € 5.800. Zusätzlich gewährt der Arbeitgeber einen jährlichen Bonus in Höhe von bis zu 10% des Jahresgehalts, abhängig von Unternehmens- und individueller Leistung.",
                "reference": ["pdf"]
            },
            {
                "question": "Bis zu welchem Wert sind Geschenke bei TechFlow Solutions erlaubt und wen kann ich bei Compliance-Verstößen kontaktieren?",
                "expected_answer": "Geschenke bis €50 sind erlaubt, alle Geschenke über €25 müssen dokumentiert werden. Geschäftsessen bis €100 pro Person sind zulässig. Bei Compliance-Verstößen können Sie sich wenden an: Compliance-Hotline 030-555-COMP (2667), E-Mail compliance@techflow-solutions.de oder persönlich an Compliance Officer Lisa Schmidt.",
                "reference": ["pdf"]
            },
            {
                "question": "Wer ist mein Buddy/Mentor während des Onboardings und welche wichtigen Ziele muss ich in den ersten 60 Tagen erreichen?",
                "expected_answer": "Ihr Buddy/Mentor ist David Kim (Senior Tech Lead), erreichbar unter d.kim@techflow-solutions.de und Slack @david.kim. Die 60-Tage-Ziele umfassen: eigenständige Feature-Entwicklung, Durchführung von Code-Reviews, vertiefte Produktkenntnisse und das Verstehen von Architektur-Entscheidungen.",
                "reference": ["pdf"]
            },
            {
                "question": "Welche kritischen Alert-Schwellenwerte gelten für die CloudSync API und wer ist für den Wissenstransfer verantwortlich?",
                "expected_answer": "Kritische Alert-Schwellenwerte für die CloudSync API: API Response Time >5000ms für >2 Minuten, Fehlerrate >15% für >1 Minute, kompletter Service-Ausfall (0% Verfügbarkeit). Der Wissenstransfer wird von Kevin Schulz (DevOps Engineer) an Martin Fischer (IT Lead) übertragen. Das System ist als KRITISCH eingestuft und erfordert sofortige Übergabe.",
                "reference": ["pdf"]
            },
            {
                "question": "Wann ist der bestätigte Beta-Launch-Termin für CloudSync und welche Pricing-Strategie wurde für das Freemium-Modell festgelegt?",
                "expected_answer": "Der Beta-Launch-Termin wurde für den 15. Juni 2025 mit 50 ausgewählten Kunden bestätigt. Die Pricing-Strategie sieht ein Freemium-Modell mit 5GB Speicher-Limit vor, und $15/Monat für unbegrenzten Speicher. Der Feature Freeze ist für den 1. Mai 2025 geplant.",
                "reference": ["confluence"]
            },
            {
                "question": "Wer ist der CTO von TechFlow Solutions und wie kann ich ihn kontaktieren? Außerdem benötige ich die Kontaktdaten des IT-Support für einen Notfall.",
                "expected_answer": "Der CTO ist Dr. Michael Schmidt. Er ist erreichbar unter: E-Mail michael.schmidt@techflow-solutions.de, Telefon +49 30 555-0111, Mobil +49 175 555-2011, Standort Berlin Büro 3.02, Slack @michael.schmidt, GitHub @mschmidt-techflow. Für IT-Notfälle: 24/7 IT-Support unter +49 30 555-0199, oder Martin Fischer (IT Lead) unter martin.fischer@techflow-solutions.de, Telefon +49 30 555-0511, Mobil +49 175 555-2511.",
                "reference": ["confluence"]
            },
            {
                "question": "Wann wurde TechFlow Solutions gegründet, wie viele Mitarbeiter arbeiten dort und welcher Technologie-Stack wird verwendet?",
                "expected_answer": "TechFlow Solutions wurde 2018 in Berlin gegründet und beschäftigt 150 Experten. Der Technologie-Stack umfasst: Frontend (React, TypeScript), Backend (Java Spring Boot, Python), Cloud (AWS, Docker, Kubernetes) und Datenbanken (PostgreSQL, MongoDB). Das Unternehmen entwickelt innovative Workflow-Management-Software für mittelständische Unternehmen.",
                "reference": ["confluence"]
            },
            {
                "question": "Wie viele Mitarbeiter arbeiten an den verschiedenen TechFlow Solutions Standorten und welche Remote-Work-Regelungen gelten?",
                "expected_answer": "TechFlow Solutions hat zwei Standorte: Berlin (Hauptsitz) mit 110 Mitarbeitern und München (Development Hub) mit 40 Mitarbeitern. Die Remote Work Policy ermöglicht ein Hybrid-Arbeitsmodell mit bis zu 3 Tagen pro Woche remote. Für Senior-Rollen sind Vollremote-Optionen nach Vereinbarung möglich. Quarterly Team-Events sorgen für Remote-Integration.",
                "reference": ["miro"]
            },
            {
                "question": "Welche konkreten Expansionspläne hat TechFlow Solutions für Q2 und Q4 2025 und wie funktionieren die Eskalationswege bei technischen Problemen?",
                "expected_answer": "Für Q2 2025 plant TechFlow +15 Backend-Entwickler für CloudSync, +5 Sales Representatives (International) und +3 Customer Success Manager. Q4 2025 Ziel: 180 Mitarbeitende, dritter Standort (Frankfurt oder Hamburg) und ein Data Science Team (8 Personen). Bei technischen Issues gilt der Eskalationsweg: Team Lead → CTO. Management-Meetings finden statt: Daily Standups (9:30 Uhr), Weekly Leadership (Dienstag 14:00 Uhr), Monthly All-Hands (letzter Freitag 16:00 Uhr).",
                "reference": ["miro"]
            },
            {
                "question": "Nenn mir die direkte Berichtslinie von Sandra Weber und Ihren Fokus.",
                "expected_answer": "Sandra Weber gibt einen direct report an Lisa Müller. Thomas Wagner (CFO) managed die Abteilungen Sales & Marketing und Customer Support.",
                "reference": ["miro"]
            }
        ]
        
        print("✅ Evaluator erfolgreich initialisiert")
    
    def check_vectorstore(self):
        """Überprüft, ob der Vektorspeicher verfügbar ist"""
        retriever = self.vectorstore.get_retriever()
        if not retriever:
            print("❌ Fehler: Vektorspeicher nicht verfügbar!")
            print("   Stellen Sie sicher, dass:")
            print("   1. Der Chatbot bereits ausgeführt wurde")
            print("   2. Dokumente geladen und verarbeitet wurden")
            print("   3. Der ChromaDB-Ordner existiert")
            return False
        
        print("✅ Vektorspeicher verfügbar")
        return True
    
    def generate_answer(self):
        """Generiert Antwort für den Testfall mit der Chatbot-Logik"""
        print("\n📝 Generiere Antwort mit Chatbot-Logik...")
        
        retriever = self.vectorstore.get_retriever()
        qa_chain = create_qa_chain(retriever)
        
        try:
            # Verwende die gleiche Logik wie der Chatbot
            result = query_knowledge_base(qa_chain, self.test_case['question'])
            
            # Hole relevante Kontexte
            docs = retriever.get_relevant_documents(self.test_case['question'])
            contexts = [doc.page_content for doc in docs]
            
            qa_pair = {
                'question': self.test_case['question'],
                'answer': result['answer'],
                'contexts': contexts,
                'ground_truth': self.test_case['expected_answer'],
                'sources': result.get('sources', [])
            }
            
            print("✅ Antwort erfolgreich generiert")
            return qa_pair
            
        except Exception as e:
            print(f"❌ Fehler bei der Antwortgenerierung: {e}")
            return None
    
    def run_ragas_evaluation(self, qa_pair):
        """Führt die RAGAS-Evaluation durch"""
        print("\n🔍 Führe RAGAS-Evaluation durch...")
        
        # Dataset für RAGAS vorbereiten
        evaluation_data = {
            'question': [qa_pair['question']],
            'answer': [qa_pair['answer']],
            'contexts': [qa_pair['contexts']],
            'ground_truth': [qa_pair['ground_truth']]
        }
        
        dataset = Dataset.from_dict(evaluation_data)
        
        try:
            # RAGAS Evaluation
            results = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=self.llm,
                embeddings=self.embeddings
            )
            
            print("✅ RAGAS-Evaluation abgeschlossen")
            return results
            
        except Exception as e:
            print(f"❌ Fehler bei RAGAS-Evaluation: {e}")
            return None
    
    def print_results(self, results, qa_pair):
        """Zeigt die Ergebnisse im Terminal an"""
        print("\n" + "="*80)
        print("📊 RAG CHATBOT EVALUATION ERGEBNISSE")
        print("="*80)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Evaluiert am: {timestamp}")
        
        # Testfall anzeigen
        print(f"\n❓ TESTFRAGE:")
        print("-" * 50)
        print(f"'{self.test_case['question']}'")
        
        print(f"\n🤖 GENERIERTE ANTWORT:")
        print("-" * 50)
        print(f"'{qa_pair['answer']}'")
        
        print(f"\n🎯 ERWARTETE ANTWORT:")
        print("-" * 50)
        print(f"'{self.test_case['expected_answer']}'")
        
        print(f"\n📚 VERWENDETE KONTEXTE:")
        print("-" * 50)
        for i, context in enumerate(qa_pair['contexts'][:3], 1):  # Zeige nur erste 3
            print(f"Kontext {i}:")
            print(f"'{context[:200]}...'")
            print()
        
        # Quellen
        if qa_pair['sources']:
            print(f"📄 QUELLEN:")
            print("-" * 50)
            for source in qa_pair['sources']:
                print(f"- {source}")
        
        # Metriken anzeigen - korrigierte Version für EvaluationResult
        print(f"\n🎯 RAGAS METRIKEN:")
        print("-" * 50)
        
        # RAGAS gibt ein EvaluationResult-Objekt zurück, das wie ein DataFrame funktioniert
        metric_scores = {}
        
        # Konvertiere EvaluationResult zu Dictionary
        if hasattr(results, 'to_pandas'):
            # RAGAS v0.1.x
            df = results.to_pandas()
            for column in df.columns:
                if column in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
                    metric_scores[column] = float(df[column].iloc[0])
        else:
            # Fallback: Versuche direkt auf Attribute zuzugreifen
            for metric_name in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
                if hasattr(results, metric_name):
                    score = getattr(results, metric_name)
                    if isinstance(score, (list, tuple)) and len(score) > 0:
                        metric_scores[metric_name] = float(score[0])
                    else:
                        metric_scores[metric_name] = float(score)
        
        # Falls immer noch leer, versuche dict() Konvertierung
        if not metric_scores:
            try:
                results_dict = dict(results)
                for metric_name, score in results_dict.items():
                    if metric_name in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
                        metric_scores[metric_name] = float(score)
            except:
                print("❌ Konnte Metriken nicht extrahieren")
                print(f"Results type: {type(results)}")
                print(f"Results: {results}")
                return
        
        # Zeige Metriken an
        for metric_name, score in metric_scores.items():
            rating = self._get_rating(score)
            print(f"{metric_name.replace('_', ' ').title():20} | {score:.3f} | {rating}")
        
        # Durchschnittsscore
        if metric_scores:
            avg_score = sum(metric_scores.values()) / len(metric_scores)
            avg_rating = self._get_rating(avg_score)
            print("-" * 50)
            print(f"{'DURCHSCHNITT':20} | {avg_score:.3f} | {avg_rating}")
        
        # Detaillierte Metrik-Erklärungen
        print(f"\n📖 METRIK-ERKLÄRUNGEN:")
        print("-" * 50)
        explanations = {
            'faithfulness': 'Treue zu Quellen - Ist die Antwort durch den Kontext gestützt?',
            'answer_relevancy': 'Antwortrelevanz - Ist die Antwort relevant zur Frage?',
            'context_precision': 'Kontextpräzision - Ist der abgerufene Kontext relevant?',
            'context_recall': 'Kontext-Vollständigkeit - Wurden alle relevanten Infos gefunden?'
        }
        
        for metric, explanation in explanations.items():
            if metric in metric_scores:
                score = metric_scores[metric]
                rating = self._get_rating(score)
                print(f"{metric.replace('_', ' ').title()}: {explanation}")
                print(f"  Score: {score:.3f} ({rating})")
                print()
        
        # Empfehlungen
        self._print_recommendations(metric_scores)
        self.save_to_csv(getattr(self, '_current_testfall_nr', 1), qa_pair, metric_scores)
        print("="*80)
    
    def _get_rating(self, score):
        """Konvertiert numerischen Score in Bewertung"""
        if score >= 0.85:
            return "🟢 Exzellent"
        elif score >= 0.70:
            return "🟡 Gut"
        elif score >= 0.55:
            return "🟠 Befriedigend"
        else:
            return "🔴 Verbesserungsbedürftig"
    
    def _print_recommendations(self, scores):
        """Zeigt Empfehlungen basierend auf den Scores"""
        print("💡 EMPFEHLUNGEN:")
        print("-" * 50)
        
        recommendations = []
        
        if 'faithfulness' in scores and scores['faithfulness'] < 0.7:
            recommendations.append(
                "• Faithfulness verbessern: Die Antwort enthält Informationen, die nicht\n"
                "  ausreichend durch den Kontext gestützt werden. Überprüfen Sie die\n"
                "  Prompt-Templates und Quelldokumente."
            )
        
        if 'answer_relevancy' in scores and scores['answer_relevancy'] < 0.7:
            recommendations.append(
                "• Answer Relevancy erhöhen: Die Antwort geht nicht vollständig auf\n"
                "  die Frage ein. Überarbeiten Sie die Prompts für direktere Antworten."
            )
        
        if 'context_precision' in scores and scores['context_precision'] < 0.7:
            recommendations.append(
                "• Context Precision optimieren: Zu viele irrelevante Informationen\n"
                "  werden abgerufen. Justieren Sie die Retrieval-Parameter (k-Wert,\n"
                "  Chunk-Größe, Similarity-Threshold)."
            )
        
        if 'context_recall' in scores and scores['context_recall'] < 0.7:
            recommendations.append(
                "• Context Recall verbessern: Nicht alle relevanten Informationen\n"
                "  werden gefunden. Erhöhen Sie möglicherweise die Anzahl der\n"
                "  abgerufenen Dokumente oder optimieren Sie das Chunking."
            )
        
        if not recommendations:
            print("🎉 Alle Metriken zeigen gute Performance!")
            print("   Das RAG-System funktioniert gut für diesen Testfall.")
        else:
            for rec in recommendations:
                print(rec)
                print()

     
    def save_to_csv(self, testfall_nr, qa_pair, metric_scores):
        """Speichert Ergebnisse in CSV-Datei (append-Modus)"""
        
        # CSV-Datei im evaluation-Ordner
        csv_file = os.path.join(current_dir, "evaluation_results.csv")
        
        # Prüfen ob Datei existiert (für Header)
        file_exists = os.path.exists(csv_file)
        
        try:
            with open(csv_file, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, delimiter=';')  # Semikolon für deutsche Excel-Kompatibilität
                
                # Header nur beim ersten Mal schreiben
                if not file_exists:
                    header = [
                        'Timestamp',
                        'Testfall_Nr',
                        'Frage',
                        'Generierte_Antwort',
                        'Erwartete_Antwort',
                        'Faithfulness',
                        'Answer_Relevancy',
                        'Context_Precision',
                        'Context_Recall',
                        'Durchschnitt',
                        'Quellen'
                    ]
                    writer.writerow(header)
                
                # Timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Durchschnitt berechnen
                avg_score = sum(metric_scores.values()) / len(metric_scores) if metric_scores else 0
                
                # Quellen zusammenfassen
                sources = ', '.join(qa_pair.get('sources', [])) if qa_pair.get('sources') else 'Keine'
                
                # Datenzeile
                row = [
                    timestamp,
                    testfall_nr,
                    qa_pair['question'],
                    qa_pair['answer'],
                    qa_pair['ground_truth'],
                    metric_scores.get('faithfulness', 0),
                    metric_scores.get('answer_relevancy', 0),
                    metric_scores.get('context_precision', 0),
                    metric_scores.get('context_recall', 0),
                    avg_score,
                    sources
                ]
                
                writer.writerow(row)
                
            print(f"💾 Ergebnisse in CSV gespeichert: {csv_file}")
            
        except Exception as e:
            print(f"❌ Fehler beim Speichern in CSV: {e}")           
    
    def run_evaluation(self):
        """Führt die komplette Evaluation durch"""
        print("🎯 Starte RAG Chatbot Evaluation (10 Testfälle)")
        print("="*50)
        
        # 1. Vektorspeicher prüfen
        if not self.check_vectorstore():
            return False
        
        # 2. Alle Testfälle durchlaufen
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n📝 TESTFALL {i}/10")
            print("-" * 40)
            
            # Aktuellen Testfall setzen
            self.test_case = test_case
            self._current_testfall_nr = i  # Für CSV Export
            
            # Antwort generieren
            qa_pair = self.generate_answer()
            if not qa_pair:
                print(f"❌ Testfall {i} fehlgeschlagen: Keine Antwort konnte generiert werden.")
                continue
            
            # RAGAS Evaluation
            results = self.run_ragas_evaluation(qa_pair)
            if not results:
                print(f"❌ Testfall {i} fehlgeschlagen: RAGAS-Evaluation nicht möglich.")
                continue
            
            # Ergebnisse anzeigen (inklusive CSV Export)
            self.print_results(results, qa_pair)
            
            print(f"✅ Testfall {i} abgeschlossen")
        
        # Abschlussmeldung
        csv_file = os.path.join(current_dir, "evaluation_results.csv")
        print(f"\n📊 Alle Ergebnisse wurden gespeichert in: {csv_file}")
        
        return True

def main():
    """Hauptfunktion"""
    print("🤖 RAG Chatbot Evaluation Script (Testversion)")
    print("=" * 50)
    
    try:
        evaluator = RAGEvaluator()
        success = evaluator.run_evaluation()
        
        if success:
            print("\n✅ Evaluation erfolgreich abgeschlossen!")
        else:
            print("\n❌ Evaluation fehlgeschlagen!")
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Evaluation abgebrochen.")
    except Exception as e:
        print(f"\n❌ Unerwarteter Fehler: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()