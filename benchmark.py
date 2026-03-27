import os
import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from service import process_cv_pipeline
from outils.llm_client import create_client, get_model

# Configuration du logging pour le benchmark
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark")

# Désactiver les liens symboliques HuggingFace pour éviter les erreurs de privilèges Windows
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

def run_benchmark(file_path: Path, modes: List[str]):
    """
    Exécute le pipeline sur un fichier avec différents modes de parsing.
    """
    results = {}
    file_path = Path(file_path)
    
    if not file_path.exists():
        logger.error(f"Fichier introuvable : {file_path}")
        return

    logger.info(f"Démarrage du benchmark pour : {file_path.name}")
    
    for mode in modes:
        logger.info(f"--- Test du mode : {mode} ---")
        
        # On force le mode de parsing via variable d'environnement
        os.environ["PDF_PARSE_MODE"] = mode
        
        start_time = time.time()
        try:
            # On crée un dossier de sortie spécifique pour ce mode
            mode_output_dir = Path("output") / mode
            mode_output_dir.mkdir(parents=True, exist_ok=True)
            
            # On lance le pipeline AVEC génération Word
            data = process_cv_pipeline(
                file_path, 
                output_dir=mode_output_dir, 
                generate_word_doc=True
            )
            duration = time.time() - start_time
            
            # Renommer le fichier Word pour inclure le mode
            original_word = mode_output_dir / f"{file_path.stem}_finaxys.docx"
            final_word = mode_output_dir / f"{file_path.stem}_{mode}_finaxys.docx"
            if original_word.exists():
                if final_word.exists(): os.remove(final_word)
                original_word.rename(final_word)
            
            # SAUVEGARDER LE JSON POUR DEBUG
            final_json = mode_output_dir / f"{file_path.stem}_{mode}.json"
            with open(final_json, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            results[mode] = {
                "success": True,
                "duration_sec": round(duration, 2),
                "word_file": str(final_word),
                "char_count": data["metadata"].get("char_count", 0),
                "num_competences": len(data.get("competences", [])),
                "num_experiences": len(data.get("experiences", [])),
                "score_completude": data["metadata"].get("score_completude", 0),
                "validation": data["metadata"].get("validation_report", {}).get("structural_valid", False)
            }
            logger.info(f"[OK] Mode {mode} terminé en {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"[ERR] Mode {mode} a échoué : {e}")
            results[mode] = {"success": False, "error": str(e)}

    return results

def compare_results(all_benchmarks: Dict[str, Any]):
    """Affiche un résumé comparatif des résultats."""
    print("\n" + "="*80)
    print(f"{'MODE':<20} | {'TEMPS':<8} | {'CHARS':<8} | {'SKILLS':<8} | {'EXP':<5} | {'VALID'}")
    print("-" * 80)
    
    for mode, metrics in all_benchmarks.items():
        if metrics["success"]:
            # Valeurs par défaut sécurisées
            dur = metrics.get('duration_sec')
            chars = metrics.get('char_count')
            skills = metrics.get('num_competences')
            exps = metrics.get('num_experiences')
            valid = metrics.get('validation')
            
            # Formattage manuel pour éviter TypeError sur None
            dur_str = f"{dur:.2f}" if dur is not None else "0.00"
            chars_str = str(chars) if chars is not None else "0"
            skills_str = str(skills) if skills is not None else "0"
            exps_str = str(exps) if exps is not None else "0"
            
            print(f"{mode:<20} | {dur_str:<8} | {chars_str:<8} | "
                  f"{skills_str:<8} | {exps_str:<5} | {valid}")
        else:
            print(f"{mode:<20} | ERREUR : {metrics.get('error')[:50]}...")
    print("="*80 + "\n")

if __name__ == "__main__":
    # Liste des CVs à tester (tu peux en ajouter d'autres)
    test_files = [
        #Path("input/cv-developpeur-web.pdf"),
        #Path("input/cvsofianedevlast.pdf"),
        Path("input/radiologue.pdf")

    ]
    
    # Modes de parsing à comparer
    modes_to_test = ["docling", "vision", "pypdf"]
    
    for cv in test_files:
        if cv.exists():
            report = run_benchmark(cv, modes_to_test)
            print(f"Rapport pour {cv.name}:")
            compare_results(report)
