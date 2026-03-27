"""
Génération du dossier de compétences Finaxys (3 pages).
Alignement strict sur les marqueurs Jinja2 du template.
"""

import logging
from pathlib import Path
from docxtpl import DocxTemplate

logger = logging.getLogger(__name__)

TEMPLATE_PATH = Path(__file__).parent.parent / "templates" / "finaxys_template.docx"


def build_context(cv_data: dict) -> dict:
    """
    Transforme le JSON en contexte Jinja2 parfaitement aligné sur vos marqueurs.
    """
    # 1. Identité
    id_data = cv_data.get("identite") or cv_data.get("informations_personnelles") or {}
    prenom = id_data.get("prenom") or str(id_data.get("nom", "")).split(" ")[0]
    nom = id_data.get("nom") or ""
    if " " in str(nom) and len(str(nom).split(" ")) > 1:
        parts = str(nom).split(" ", 1)
        prenom, nom = parts[0], parts[1]

    # 2. Formations
    form_raw = cv_data.get("formations") or cv_data.get("formation") or []
    formations_fmt = []
    for f in (form_raw if isinstance(form_raw, list) else [form_raw]):
        if not isinstance(f, dict): continue
        
        # Récupération des dates (format YYYY)
        d_debut = str(f.get("date_debut") or "").split("-")[0]
        d_fin = str(f.get("date_fin") or f.get("annee") or f.get("date_obtention") or "").split("-")[0]
        
        formations_fmt.append({
            "date_debut": d_debut,
            "date_fin": d_fin,
            "annee": d_fin, # On garde annee pour la compatibilité avec d'autres templates
            "diplome": f.get("diplome") or f.get("titre") or "",
            "etablissement": f.get("etablissement") or f.get("ecole") or ""
        })

    # 3. Compétences
    comp_raw = cv_data.get("competences") or cv_data.get("competences_principales") or []
    comp_map = {}
    for c in (comp_raw if isinstance(comp_raw, list) else [comp_raw]):
        if isinstance(c, str):
            comp_map.setdefault("COMPÉTENCES", []).append(c)
        elif isinstance(c, dict):
            cat = c.get("categorie") or c.get("category") or "COMPÉTENCES"
            n = c.get("nom") or c.get("name") or ""
            if n: comp_map.setdefault(cat, []).append(n)
    competences_list = [{"categorie": k, "noms": ", ".join(v)} for k, v in comp_map.items()]

    # 4. Langues (Vérification stricte pour la boucle {% for l in langues %}{{ l.langue }} )
    lang_raw = cv_data.get("langues") or cv_data.get("langue") or []
    lang_fmt = []
    for l in (lang_raw if isinstance(lang_raw, list) else [lang_raw]):
        if isinstance(l, str): 
            lang_fmt.append({"langue": l, "niveau": "", "certification": ""})
        elif isinstance(l, dict):
            lang_fmt.append({
                "langue": l.get("langue") or l.get("nom") or l.get("name") or "",
                "niveau": l.get("niveau") or l.get("level") or "",
                "certification": l.get("certification") or ""
            })

    # 5. Expériences
    exp_raw = cv_data.get("experiences") or cv_data.get("experience_professionnelle") or []
    exp_fmt = []
    for e in (exp_raw if isinstance(exp_raw, list) else [exp_raw]):
        if not isinstance(e, dict): continue
        m_list = e.get("missions") or e.get("responsabilites") or e.get("description") or []
        if isinstance(m_list, str): m_list = [m_list]
        exp_fmt.append({
            "titre": e.get("titre") or e.get("poste") or "",
            "entreprise": e.get("entreprise") or "",
            "date_debut": _format_date(e.get("date_debut", "")),
            "date_fin": "En cours" if e.get("en_cours") else _format_date(e.get("date_fin", "")),
            "projet": e.get("projet") or "",
            "equipe": e.get("equipe") or "",
            "methodologie": e.get("methodologie") or "",
            "missions": m_list,
            "env_technique": ", ".join(e.get("technologies") or e.get("env_technique") or [])
        })

    # 6. Projets (Personnels / Académiques)
    proj_raw = cv_data.get("projets") or cv_data.get("projets_personnels") or []
    proj_fmt = []
    for p in (proj_raw if isinstance(proj_raw, list) else [proj_raw]):
        if not isinstance(p, dict): continue
        m_proj = p.get("missions") or p.get("description") or []
        if isinstance(m_proj, str): m_proj = [m_proj]

        # Enrichir le nom avec la description si nécessaire
        nom_complet = p.get("nom") or p.get("titre") or "Projet sans nom"

        proj_fmt.append({
            "nom": nom_complet,
            "description": p.get("description") or "",
            "missions": m_proj,
            "env_technique": ", ".join(p.get("technologies") or [])
        })


    return {
        "initiales": (prenom[:1] + nom[:1]).upper() if prenom and nom else "",
        "prenom": prenom,
        "nom": str(nom).upper(),
        "titre_professionnel": cv_data.get("titre_professionnel") or cv_data.get("titre") or "",
        "type_poste": cv_data.get("type_poste") or "",
        "formations": formations_fmt,
        "competences": competences_list,
        "langues": lang_fmt,
        "experiences": exp_fmt,
        "projets": proj_fmt,
        "exp_projets": ""
    }


def generate_word(cv_data: dict, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not TEMPLATE_PATH.exists(): raise FileNotFoundError(f"Template introuvable : {TEMPLATE_PATH}")
    tpl = DocxTemplate(str(TEMPLATE_PATH))
    context = build_context(cv_data)
    tpl.render(context)
    tpl.save(str(output_path))
    return output_path


def _format_date(date_str: str) -> str:
    MOIS = {"01": "Janvier", "02": "Février", "03": "Mars", "04": "Avril", "05": "Mai", "06": "Juin",
            "07": "Juillet", "08": "Août", "09": "Septembre", "10": "Octobre", "11": "Novembre", "12": "Décembre"}
    date_str = str(date_str)
    if len(date_str) < 7: return date_str
    year, month = date_str[:4], date_str[5:7]
    return f"{MOIS.get(month, month)} {year}"
