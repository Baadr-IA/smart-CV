from outils.job_referentiel import build_prefilter_terms, resolve_poste


def test_build_prefilter_terms_includes_job_title_aliases_and_skills():
    poste = resolve_poste("DevOps")

    terms = build_prefilter_terms("DevOps", poste, max_aliases=3, max_skills=4)

    assert "DevOps" in terms
    assert "DevOps Engineer" in terms
    assert "Ingénieur DevOps" in terms
    assert "Docker" in terms
    assert "Kubernetes" in terms


def test_build_prefilter_terms_dedupes_case_insensitively():
    poste = {
        "label": "Backend Java",
        "aliases": ["backend java", "Java Developer"],
        "required_skills": ["Java", "java", "Spring Boot"],
    }

    terms = build_prefilter_terms("Backend Java", poste)

    assert terms.count("Backend Java") == 1
    assert "Java" in terms
    assert "Spring Boot" in terms
