---
name: skills-classifier
description: Use this skill to normalize, deduplicate, and categorize CV skills into Finaxys categories with calibrated proficiency levels and optional years of experience. Apply it when consolidating skills from both explicit skill sections and technologies found in professional experiences.
---

# Skills Classifier

## When to use
- A CV JSON contains raw or inconsistent competences.
- Technologies in experiences must be merged into competences.

## Instructions
1. **LANGUE OBLIGATOIRE : FRANÇAIS.** Tous les noms de catégories, niveaux et noms de compétences (si traduisibles comme 'Soft skills') doivent être en français.
2. **NETTOYAGE CRITIQUE** : Supprime TOUS les artefacts de barres de progression (ex: "Python [#######---]", "Java 80%", "React (Expert)"). Ne garde QUE le nom propre de la compétence (ex: "Python", "Java", "React").
3. Normalize abbreviations and aliases consistently.
4. Deduplicate equivalent technologies.
5. Include technologies from experiences even if absent in competences.
6. Assign one category only from this list:
   - Langages de programmation
   - Frameworks & Librairies
   - Bases de données
   - Cloud & DevOps
   - Outils & Méthodologies
   - Compétences fonctionnelles
   - Soft skills
7. Assign one level only from: Débutant, Intermédiaire, Confirmé, Expert.
8. Keep annees_experience when explicit, otherwise null.

## Output contract
- Return a JSON array only.
- Item format:
  { "nom": string, "categorie": string, "niveau": string, "annees_experience": number|null }
- No markdown fences or extra text.
