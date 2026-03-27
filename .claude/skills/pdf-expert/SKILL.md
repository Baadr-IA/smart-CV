---
name: pdf-expert
description: "Expert technique pour la manipulation, l'extraction et l'analyse de fichiers PDF. À utiliser pour extraire du texte complexe (colonnes, tableaux), effectuer de l'OCR sur des documents scannés ou analyser les métadonnées techniques d'un PDF."
---

# PDF Technical Expert

Ce skill fournit l'expertise technique de bas niveau pour la manipulation des fichiers PDF.

## Outils recommandés
- **pdfplumber** : À privilégier pour l'extraction de texte "layout-aware" (respectant la mise en page).
- **pytesseract + pdf2image** : Pour les documents scannés nécessitant une reconnaissance optique (OCR).
- **pypdf** : Pour manipuler les métadonnées et fusionner/diviser des fichiers.

## Méthodes d'extraction
1. **Texte avec Layout** (pdfplumber) : `page.extract_text(layout=True)`. Indispensable pour les documents multi-colonnes.
2. **Tableaux** : `page.extract_table()` ou `page.find_tables()`.
3. **OCR (Images)** : Si le PDF ne contient pas de flux de texte, convertir en image via `pdf2image` puis passer dans `tesseract`.

## Instructions pour l'Assistant
1. **Validation** : Toujours vérifier si le texte extrait est cohérent (pas de caractères spéciaux aléatoires).
2. **Métadonnées** : Consulter l'auteur, le logiciel de création et la date de modification pour comprendre l'origine du fichier.
3. **Optimisation** : Si l'extraction est lente sur un gros PDF, suggérer d'extraire uniquement les pages pertinentes.
