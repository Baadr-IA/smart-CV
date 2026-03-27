---
name: cv-validator
description: Use this skill to perform semantic validation of a Finaxys CV JSON after schema validation, checking chronology, skill-experience consistency, education-career plausibility, and suspicious data patterns with a machine-readable report.
---

# CV Validator

## When to use
- JSON schema validation is complete.
- A semantic consistency pass is required.

## Instructions
1. Validate chronology and detect implausible date overlaps.
2. Validate coherence between competences and technologies used in experiences.
3. Validate education-career plausibility.
4. Flag suspicious or likely fabricated values.

## Output contract
- Return exactly one JSON object with fields:
  - is_valid: boolean
  - score_coherence: number between 0.0 and 1.0
  - warnings: array of strings
  - errors: array of strings
- No markdown or explanatory prose.
