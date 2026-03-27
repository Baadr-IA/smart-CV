---
name: cv-classifier-expert
description: "Expert en classification de compétences selon le référentiel Finaxys. À utiliser pour valider ou corriger la catégorisation des technologies extraites d'un CV."
---

# CV Classifier Expert

Ce skill garantit que les compétences sont rangées dans les bonnes cases pour le template Word final.

## Référentiel Finaxys
Chaque technologie doit être assignée à UNE SEULE catégorie parmi celles-ci :

1. **Langages de programmation** : Java, Python, C++, TypeScript, SQL, etc.
2. **Frameworks & Librairies** : Spring Boot, React, Angular, Django, Pandas, etc.
3. **Bases de données** : PostgreSQL, MongoDB, Oracle, Redis, etc.
4. **Cloud & DevOps** : AWS, Azure, Docker, Kubernetes, Jenkins, Terraform, etc.
5. **Outils & Méthodologies** : Jira, Git, Agile, Scrum, Kanban, VS Code, etc.
6. **Compétences fonctionnelles** : Analyse métier, Gestion de projet, Finance de marché, etc.
7. **Soft skills** : Leadership, Communication, Esprit d'équipe, Adaptabilité.

## Règles de Normalisation
- **JavaScript** : Pas "JS" ou "Javascript".
- **React** : Pas "React.js" ou "ReactJS".
- **AWS** : Pas "Amazon Web Services".
- **Niveaux** : Utilise strictement [Débutant, Intermédiaire, Confirmé, Expert].

## Logique de Classification complexe
- Si une techno est à la fois un langage et un outil (ex: SQL), privilégie la catégorie la plus technique (**Langages**).
- Les technos transverses (ex: Kafka) vont dans **Cloud & DevOps** ou **Frameworks** selon le contexte de l'expérience.

## Instructions pour l'Assistant
1. **Dépassement du CV** : Si une techno est citée dans les expériences mais pas dans la liste des compétences, AJOUTE-LA automatiquement à la classification.
2. **Dédoublonnage** : Fusionne "Java 8" et "Java 11" en "Java" si les années d'expérience sont cumulables.
