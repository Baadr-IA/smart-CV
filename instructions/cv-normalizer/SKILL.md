---
name: cv-normalizer
description: Use this skill to normalize CV wording to Finaxys editorial style while preserving all factual content and JSON structure. Apply it after extraction and skill classification to improve readability, consistency, and consulting-oriented phrasing.
---

# CV Normalizer

## When to use
- A structured CV JSON is valid but style is inconsistent.
- Text must follow Finaxys writing conventions.

## Instructions
1. Rewrite profil in third person, concise, 2-3 sentences max.
2. Start each mission with a past-tense action phrase.
3. Prefer measurable outcomes in resultats when evidence exists.
4. Keep titre_professionnel concise, max 6 words.
5. Remove first-person pronouns.
6. Standardize technology capitalization.

## Hard constraints
- Do not alter dates, organizations, institution names, or certifications.
- Do not alter JSON shape or keys.
- Do not invent facts.

## Output contract
- Return full JSON only, without markdown or commentary.
