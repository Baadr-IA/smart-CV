---
name: cv-extractor-expert
description: "Expert pour le débogage et l'optimisation de l'extraction de données depuis des CV PDF/Word complexes. À utiliser quand le pipeline automatique échoue, quand le texte est mal segmenté ou quand le format (colonnes, tableaux) pose problème."
---

# CV Extractor Expert

Ce skill intervient en support du pipeline `outils/parser.py` et `outils/extractor.py`.

## Diagnostic d'extraction
Si l'extraction automatique est de mauvaise qualité, vérifie les points suivants :

### 1. Problème de mise en page (Layout)
Les CV modernes utilisent souvent deux colonnes. Une extraction brute peut mélanger le texte de la colonne de gauche avec celle de droite.
- **Solution recommandée** : Utilise `pdfplumber` avec `layout=True` pour conserver la structure spatiale.
- **Vérification** : Demande à l'agent de lire les 500 premiers caractères. Si les mots semblent mélangés (ex: "Compétences Expérience Java 2022"), c'est un problème de layout.

### 2. CV "Image" (OCR)
Si le texte extrait est vide ou contient des caractères incohérents (ex: ""), le CV est probablement une image.
- **Action** : Utilise l'outil `pytesseract` ou `pdf2image` pour effectuer une reconnaissance optique de caractères.

### 3. Segmentation des sections
Si l'extracteur mélange les expériences et les formations :
- Vérifie si les mots-clés de rupture (Expériences, Formation, Projets) sont bien détectés.
- Assure-toi que l'extracteur ne s'arrête pas prématurément (limite de tokens).

## Instructions pour l'Assistant
1. **Analyse visuelle** : Si possible, demande à voir une capture du CV pour comprendre sa structure.
2. **Extraction ciblée** : Si une section manque, essaie d'extraire uniquement la zone géographique correspondante dans le PDF.
3. **Nettoyage** : Supprime les icônes (téléphone, enveloppe) qui peuvent polluer l'interprétation du LLM.
