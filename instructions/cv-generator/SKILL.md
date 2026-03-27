---
name: cv-generator
description: Use this skill to generate a Finaxys-styled Word document from structured CV JSON using deterministic templating, preserving corporate visual standards and section ordering.
---

# CV Generator

## When to use
- CV JSON is already validated and normalized.
- A Word output compliant with Finaxys style is required.

## Instructions
1. Keep deterministic rendering with template-based mapping.
2. Preserve section order: Header, Profil, Competences, Experiences, Formations, Certifications, Langues.
3. Respect visual rules: Calibri sizes, corporate blue #003366, thin separators.
4. Render competences grouped by category.
5. Render experiences with bold role, gray dates, bullet missions.

## Constraints
- No LLM dependence for document rendering.
- If a section is empty, keep a minimal readable section block.
