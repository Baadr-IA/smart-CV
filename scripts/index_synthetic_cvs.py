"""
Script d'indexation de CVs synthétiques dans PostgreSQL/pgvector.
Génère 30 profils IT diversifiés directement en CVData (sans LLM).
Usage: python scripts/index_synthetic_cvs.py
"""

import sys
import os
import random
import json
from datetime import datetime, date
from pathlib import Path
from dateutil.relativedelta import relativedelta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from schemas.models import CVData, Identite, Competence, Experience, Formation, Certification, Langue, Metadata
from outils.rag_utils import VectorStoreManager

# ─── Données de base ────────────────────────────────────────────────────────

PRENOMS = ["Alice", "Baptiste", "Camille", "David", "Emma", "Francois", "Gabriel", "Helene",
           "Ismail", "Julie", "Kevin", "Laura", "Mehdi", "Nathalie", "Omar", "Pierre",
           "Quentin", "Rachel", "Sebastien", "Theo", "Ugo", "Valeria", "Wassim", "Xavier",
           "Yasmine", "Zineb", "Arnaud", "Beatrice", "Cedric", "Diane"]

NOMS = ["Martin", "Dupont", "Leroy", "Moreau", "Simon", "Laurent", "Lefebvre", "Michel",
        "Garcia", "David", "Bernard", "Thomas", "Robert", "Petit", "Dubois", "Fontaine",
        "Girard", "Bonnet", "Lambert", "Chevalier", "El Mansouri", "Benali", "Traore",
        "Nguyen", "Pham", "Rousseau", "Vincent", "Faure", "Roux", "Blanc"]

ENTREPRISES = ["Finaxys", "BNP Paribas", "Societe Generale", "Capgemini", "Sopra Steria",
               "Thales", "Orange", "Atos", "CGI", "Accenture", "IBM France",
               "Credit Agricole", "AXA", "SNCF", "Air France", "Natixis",
               "Amundi", "La Poste", "Michelin", "TotalEnergies", "EDF", "Engie"]

ECOLES = [
    ("Ingenieur Informatique", "Centrale Paris"),
    ("Master Informatique", "Universite Paris-Saclay"),
    ("Licence Informatique", "Universite Pierre et Marie Curie"),
    ("Ingenieur Data Science", "ENSAI"),
    ("Master Machine Learning", "ENS Lyon"),
    ("BTS Informatique", "Lycee Louis le Grand"),
    ("Master DevOps", "Universite de Bordeaux"),
    ("Ingenieur Reseaux", "Telecom SudParis"),
    ("Master Cybersecurite", "EPITA"),
    ("MBA Management SI", "HEC Paris"),
]

# ─── Profils métier ──────────────────────────────────────────────────────────

PROFILES = [
    {
        "titre": "Developpeur Java/Spring Boot Senior",
        "type_poste": "Developpeur Backend",
        "profil": "Developpeur backend avec 6 ans d experience sur Java et Spring Boot, specialise dans les architectures microservices et les APIs REST.",
        "skills": [
            ("Java", "Backend", "Expert", 6.0),
            ("Spring Boot", "Backend", "Expert", 5.0),
            ("Spring Cloud", "Backend", "Confirme", 3.0),
            ("REST API", "Backend", "Expert", 6.0),
            ("Maven", "DevOps", "Confirme", 5.0),
            ("JUnit", "Tests", "Confirme", 4.0),
            ("Mockito", "Tests", "Confirme", 3.0),
            ("PostgreSQL", "Base de donnees", "Confirme", 4.0),
            ("Docker", "DevOps", "Intermediaire", 2.0),
            ("Kubernetes", "DevOps", "Intermediaire", 1.5),
            ("Git", "DevOps", "Expert", 6.0),
            ("Jenkins", "CI/CD", "Intermediaire", 2.0),
        ],
        "certifications": [("Oracle Certified Professional Java SE 11", "Oracle", 2022)],
        "missions": [
            "Developpement de microservices Spring Boot pour une plateforme bancaire",
            "Mise en place de pipelines CI/CD avec Jenkins et Maven",
            "Optimisation des performances SQL et tuning des requetes PostgreSQL",
        ],
        "base_years": 6
    },
    {
        "titre": "Data Engineer MLOps",
        "type_poste": "Data Engineer",
        "profil": "Data Engineer passionne par le MLOps, specialise dans la construction de pipelines de donnees scalables et le deploiement de modeles ML en production.",
        "skills": [
            ("Python", "Data", "Expert", 5.0),
            ("Apache Spark", "Data", "Confirme", 3.0),
            ("Apache Kafka", "Data", "Confirme", 2.5),
            ("Airflow", "Data", "Confirme", 3.0),
            ("dbt", "Data", "Intermediaire", 1.5),
            ("MLflow", "MLOps", "Confirme", 2.0),
            ("Docker", "DevOps", "Expert", 4.0),
            ("Kubernetes", "DevOps", "Confirme", 2.0),
            ("AWS", "Cloud", "Confirme", 3.0),
            ("S3", "Cloud", "Confirme", 3.0),
            ("PostgreSQL", "Base de donnees", "Confirme", 4.0),
            ("Pandas", "Data", "Expert", 5.0),
            ("scikit-learn", "Machine Learning", "Confirme", 3.0),
        ],
        "certifications": [("AWS Certified Data Analytics", "Amazon", 2023)],
        "missions": [
            "Construction de pipelines Spark pour traitement de donnees financieres a grande echelle",
            "Deploiement de modeles ML avec MLflow et Kubernetes",
            "Orchestration de workflows data avec Apache Airflow",
        ],
        "base_years": 5
    },
    {
        "titre": "Architecte Solutions Cloud AWS Azure",
        "type_poste": "Architecte Cloud",
        "profil": "Architecte cloud certifie AWS et Azure avec 8 ans d experience, specialise dans la conception d architectures serverless et la migration cloud.",
        "skills": [
            ("AWS", "Cloud", "Expert", 7.0),
            ("Azure", "Cloud", "Expert", 5.0),
            ("Terraform", "DevOps", "Expert", 4.0),
            ("Kubernetes", "DevOps", "Expert", 4.0),
            ("Docker", "DevOps", "Expert", 6.0),
            ("Ansible", "DevOps", "Confirme", 3.0),
            ("Python", "Backend", "Confirme", 4.0),
            ("Serverless", "Cloud", "Expert", 3.0),
            ("Lambda", "Cloud", "Expert", 3.0),
            ("RDS", "Base de donnees", "Confirme", 4.0),
            ("CI/CD", "DevOps", "Expert", 6.0),
            ("GitLab CI", "CI/CD", "Expert", 4.0),
        ],
        "certifications": [
            ("AWS Solutions Architect Professional", "Amazon", 2022),
            ("Microsoft Azure Administrator", "Microsoft", 2023),
        ],
        "missions": [
            "Architecture et migration d applications monolithiques vers microservices AWS",
            "Implementation d infrastructure as code avec Terraform",
            "Conception d architectures event-driven sur Azure Service Bus",
        ],
        "base_years": 8
    },
    {
        "titre": "Developpeur Full Stack Angular Node.js",
        "type_poste": "Developpeur Full Stack",
        "profil": "Developpeur full stack avec expertise Angular et Node.js, passionne par l UX et les applications web modernes.",
        "skills": [
            ("Angular", "Frontend", "Expert", 5.0),
            ("TypeScript", "Frontend", "Expert", 5.0),
            ("Node.js", "Backend", "Expert", 4.0),
            ("Express.js", "Backend", "Confirme", 4.0),
            ("React", "Frontend", "Confirme", 2.0),
            ("HTML5/CSS3", "Frontend", "Expert", 6.0),
            ("MongoDB", "Base de donnees", "Confirme", 3.0),
            ("PostgreSQL", "Base de donnees", "Intermediaire", 2.0),
            ("GraphQL", "Backend", "Confirme", 2.0),
            ("Docker", "DevOps", "Intermediaire", 2.0),
            ("Jest", "Tests", "Confirme", 3.0),
            ("Cypress", "Tests", "Intermediaire", 1.5),
        ],
        "certifications": [],
        "missions": [
            "Developpement de SPA Angular pour plateforme de gestion RH",
            "Creation d APIs REST Node.js/Express avec authentification JWT",
            "Mise en place de tests E2E avec Cypress",
        ],
        "base_years": 5
    },
    {
        "titre": "Ingenieur DevOps SRE",
        "type_poste": "DevOps SRE",
        "profil": "Ingenieur DevOps SRE avec 5 ans d experience, expert en automatisation monitoring et fiabilite des systemes en production.",
        "skills": [
            ("Docker", "DevOps", "Expert", 5.0),
            ("Kubernetes", "DevOps", "Expert", 4.0),
            ("Terraform", "DevOps", "Expert", 3.0),
            ("Ansible", "DevOps", "Expert", 4.0),
            ("CI/CD", "DevOps", "Expert", 5.0),
            ("GitHub Actions", "CI/CD", "Expert", 3.0),
            ("Prometheus", "Monitoring", "Expert", 3.0),
            ("Grafana", "Monitoring", "Expert", 3.0),
            ("ELK Stack", "Monitoring", "Confirme", 2.5),
            ("Python", "Scripting", "Confirme", 4.0),
            ("Bash", "Scripting", "Expert", 5.0),
            ("Linux", "Systeme", "Expert", 6.0),
            ("AWS", "Cloud", "Confirme", 3.0),
        ],
        "certifications": [("Certified Kubernetes Administrator CKA", "CNCF", 2023)],
        "missions": [
            "Mise en place de monitoring Prometheus/Grafana pour 50 microservices",
            "Automatisation des deploiements avec GitOps ArgoCD",
            "Reduction du MTTR de 40 pct grace a l amelioration des alertes et runbooks",
        ],
        "base_years": 5
    },
    {
        "titre": "Data Scientist ML Engineer",
        "type_poste": "Data Scientist",
        "profil": "Data Scientist avec expertise en NLP et Computer Vision, experience en deploiement de modeles en production.",
        "skills": [
            ("Python", "Data Science", "Expert", 6.0),
            ("scikit-learn", "Machine Learning", "Expert", 5.0),
            ("TensorFlow", "Deep Learning", "Confirme", 3.0),
            ("PyTorch", "Deep Learning", "Confirme", 3.0),
            ("NLP", "Machine Learning", "Expert", 4.0),
            ("Transformers", "Deep Learning", "Confirme", 2.0),
            ("Pandas", "Data", "Expert", 6.0),
            ("NumPy", "Data", "Expert", 6.0),
            ("SQL", "Base de donnees", "Confirme", 4.0),
            ("Spark", "Big Data", "Intermediaire", 2.0),
            ("Docker", "DevOps", "Intermediaire", 2.0),
            ("MLflow", "MLOps", "Intermediaire", 1.5),
        ],
        "certifications": [("Google Professional Data Engineer", "Google", 2022)],
        "missions": [
            "Developpement de modeles NLP de classification de documents (F1 > 92 pct)",
            "Construction d un systeme de recommandation pour 500K utilisateurs",
            "Deploiement de modeles ML avec FastAPI et Docker",
        ],
        "base_years": 6
    },
    {
        "titre": "Expert Cybersecurite Pentester",
        "type_poste": "Cybersecurite",
        "profil": "Expert en cybersecurite offensif et defensif, specialise dans les tests d intrusion et la gestion des vulnerabilites.",
        "skills": [
            ("Pentest", "Cybersecurite", "Expert", 5.0),
            ("OWASP", "Cybersecurite", "Expert", 5.0),
            ("Metasploit", "Cybersecurite", "Confirme", 4.0),
            ("Burp Suite", "Cybersecurite", "Expert", 5.0),
            ("Wireshark", "Cybersecurite", "Confirme", 4.0),
            ("Nmap", "Cybersecurite", "Expert", 5.0),
            ("Linux", "Systeme", "Expert", 6.0),
            ("Python", "Scripting", "Confirme", 4.0),
            ("SIEM", "Cybersecurite", "Confirme", 3.0),
            ("IAM", "Securite Cloud", "Confirme", 2.0),
            ("Kali Linux", "Cybersecurite", "Expert", 5.0),
        ],
        "certifications": [
            ("OSCP Offensive Security Certified Professional", "Offensive Security", 2022),
            ("CEH", "EC-Council", 2021),
        ],
        "missions": [
            "Tests d intrusion sur applications web bancaires (OWASP Top 10)",
            "Red team exercise sur infrastructure critique nationale",
            "Formation des equipes DevOps aux bonnes pratiques de securite",
        ],
        "base_years": 5
    },
    {
        "titre": "Developpeur Python FastAPI Backend",
        "type_poste": "Developpeur Backend Python",
        "profil": "Developpeur backend Python specialise dans FastAPI et les APIs REST haute performance, avec experience en traitement NLP.",
        "skills": [
            ("Python", "Backend", "Expert", 5.0),
            ("FastAPI", "Backend", "Expert", 3.0),
            ("SQLAlchemy", "Backend", "Confirme", 3.0),
            ("PostgreSQL", "Base de donnees", "Confirme", 4.0),
            ("Redis", "Base de donnees", "Confirme", 2.0),
            ("Celery", "Backend", "Confirme", 2.0),
            ("Docker", "DevOps", "Expert", 3.0),
            ("pytest", "Tests", "Expert", 4.0),
            ("pydantic", "Backend", "Expert", 3.0),
            ("OpenAI API", "IA", "Confirme", 1.5),
            ("LangChain", "IA", "Intermediaire", 1.0),
            ("pgvector", "IA", "Intermediaire", 1.0),
        ],
        "certifications": [],
        "missions": [
            "Developpement d une API FastAPI de traitement NLP avec OpenAI",
            "Mise en place d un systeme RAG avec PostgreSQL et pgvector",
            "Architecture event-driven avec Redis Pub/Sub et Celery",
        ],
        "base_years": 5
    },
    {
        "titre": "Consultant SAP FICO",
        "type_poste": "Consultant SAP",
        "profil": "Consultant SAP FICO avec 7 ans d experience dans les secteurs banque et industrie, expert en configurations et integrations financieres.",
        "skills": [
            ("SAP FICO", "ERP", "Expert", 7.0),
            ("SAP S/4HANA", "ERP", "Confirme", 3.0),
            ("ABAP", "Developpement", "Intermediaire", 3.0),
            ("SAP BW", "Business Intelligence", "Confirme", 4.0),
            ("Comptabilite", "Finance", "Expert", 7.0),
            ("Controlling", "Finance", "Expert", 6.0),
            ("SQL", "Base de donnees", "Intermediaire", 2.0),
            ("Excel VBA", "Bureautique", "Confirme", 5.0),
            ("Power BI", "Business Intelligence", "Intermediaire", 2.0),
            ("Gestion de projet", "Management", "Confirme", 5.0),
        ],
        "certifications": [("SAP Certified Application Associate S/4HANA Finance", "SAP", 2022)],
        "missions": [
            "Implementation SAP S/4HANA pour migration d un systeme legacy bancaire",
            "Configuration des modules FI/CO pour le groupe BNP Paribas",
            "Formation des key users et redaction des specifications fonctionnelles",
        ],
        "base_years": 7
    },
    {
        "titre": "Product Manager Owner Agile",
        "type_poste": "Product Manager",
        "profil": "Product Manager certifie Agile avec 6 ans d experience dans les Fintech, passionne par l innovation produit et la satisfaction utilisateur.",
        "skills": [
            ("Agile/Scrum", "Management", "Expert", 6.0),
            ("Product Roadmap", "Management", "Expert", 5.0),
            ("Jira", "Outils", "Expert", 6.0),
            ("Confluence", "Outils", "Expert", 6.0),
            ("UX/UI Design", "Design", "Confirme", 3.0),
            ("Data Analytics", "Data", "Confirme", 3.0),
            ("SQL", "Data", "Intermediaire", 2.0),
            ("OKR", "Management", "Expert", 4.0),
            ("A/B Testing", "Product", "Confirme", 3.0),
            ("Figma", "Design", "Intermediaire", 2.0),
        ],
        "certifications": [
            ("CSPO Certified Scrum Product Owner", "Scrum Alliance", 2021),
            ("PMP", "PMI", 2020),
        ],
        "missions": [
            "Product ownership d une plateforme de paiement mobile 500K utilisateurs",
            "Priorisation roadmap et coordination de 3 equipes de developpement",
            "Lancement de 5 nouvelles features avec +20 pct d engagement utilisateur",
        ],
        "base_years": 6
    },
    {
        "titre": "Ingenieur Reseaux Telecommunications",
        "type_poste": "Ingenieur Reseaux",
        "profil": "Ingenieur reseaux avec expertise en infrastructure Cisco, specialise dans la virtualisation reseau et la securite.",
        "skills": [
            ("Cisco", "Reseaux", "Expert", 7.0),
            ("VLAN", "Reseaux", "Expert", 6.0),
            ("BGP/OSPF", "Reseaux", "Expert", 5.0),
            ("Firewall", "Securite", "Expert", 6.0),
            ("VPN", "Securite", "Expert", 5.0),
            ("VMware NSX", "Virtualisation", "Confirme", 3.0),
            ("Python", "Scripting", "Confirme", 3.0),
            ("Ansible", "Automatisation", "Confirme", 2.5),
            ("Wireshark", "Monitoring", "Expert", 6.0),
            ("Linux", "Systeme", "Confirme", 4.0),
        ],
        "certifications": [
            ("CCNP Enterprise", "Cisco", 2022),
            ("CCNA Security", "Cisco", 2020),
        ],
        "missions": [
            "Conception et deploiement d infrastructure reseau pour datacenter 500 serveurs",
            "Migration WAN vers SD-WAN avec reduction des couts de 30 pct",
            "Automatisation des configurations reseau avec Ansible",
        ],
        "base_years": 7
    },
    {
        "titre": "Developpeur React Frontend Senior",
        "type_poste": "Developpeur Frontend",
        "profil": "Developpeur frontend senior specialise React/Redux avec 5 ans d experience, expert en performance web et accessibilite.",
        "skills": [
            ("React", "Frontend", "Expert", 5.0),
            ("Redux", "Frontend", "Expert", 4.0),
            ("TypeScript", "Frontend", "Expert", 4.0),
            ("Next.js", "Frontend", "Confirme", 2.5),
            ("GraphQL", "Backend", "Confirme", 3.0),
            ("HTML5/CSS3", "Frontend", "Expert", 6.0),
            ("Jest", "Tests", "Expert", 4.0),
            ("Webpack", "Build", "Confirme", 3.0),
            ("Storybook", "Frontend", "Confirme", 2.0),
            ("Figma", "Design", "Intermediaire", 2.0),
        ],
        "certifications": [],
        "missions": [
            "Refactorisation d une SPA legacy Angular vers React avec TypeScript",
            "Amelioration des performances Core Web Vitals de 60 pct",
            "Mise en place d un design system React avec Storybook",
        ],
        "base_years": 5
    },
    {
        "titre": "Ingenieur Intelligence Artificielle LLM",
        "type_poste": "Ingenieur IA",
        "profil": "Ingenieur IA specialise en LLMs et RAG, avec experience en fine-tuning et deploiement de modeles generatifs en production.",
        "skills": [
            ("Python", "IA", "Expert", 4.0),
            ("LangChain", "IA", "Expert", 2.5),
            ("OpenAI API", "IA", "Expert", 3.0),
            ("RAG", "IA", "Expert", 2.0),
            ("Fine-tuning LLM", "IA", "Confirme", 2.0),
            ("PyTorch", "Deep Learning", "Confirme", 3.0),
            ("Transformers", "Deep Learning", "Expert", 3.0),
            ("pgvector", "IA", "Confirme", 1.5),
            ("FastAPI", "Backend", "Confirme", 2.0),
            ("Docker", "DevOps", "Confirme", 2.5),
            ("Hugging Face", "IA", "Expert", 3.0),
            ("MLflow", "MLOps", "Intermediaire", 1.5),
        ],
        "certifications": [("DeepLearning.AI LangChain for LLM Application Development", "DeepLearning.AI", 2023)],
        "missions": [
            "Developpement d un systeme RAG d analyse de CVs avec LangChain et pgvector",
            "Fine-tuning de Mistral 7B sur donnees metier avec QLoRA",
            "Deploiement d agents IA multi-outils sur FastAPI",
        ],
        "base_years": 4
    },
    {
        "titre": "Tech Lead Java Microservices",
        "type_poste": "Tech Lead",
        "profil": "Tech Lead avec 9 ans d experience, expert en architecture microservices Java et en management d equipe de developpement.",
        "skills": [
            ("Java", "Backend", "Expert", 9.0),
            ("Spring Boot", "Backend", "Expert", 7.0),
            ("Spring Cloud", "Backend", "Expert", 5.0),
            ("Architecture Microservices", "Architecture", "Expert", 5.0),
            ("Kafka", "Messaging", "Expert", 4.0),
            ("Redis", "Base de donnees", "Confirme", 3.0),
            ("PostgreSQL", "Base de donnees", "Expert", 7.0),
            ("Docker", "DevOps", "Expert", 5.0),
            ("Kubernetes", "DevOps", "Confirme", 3.0),
            ("Design Patterns", "Architecture", "Expert", 8.0),
            ("Agile/Scrum", "Management", "Expert", 8.0),
        ],
        "certifications": [
            ("Oracle Certified Master Java EE Enterprise Architect", "Oracle", 2021),
        ],
        "missions": [
            "Lead technique d une equipe de 8 developpeurs sur une plateforme fintech",
            "Migration d un monolithe Java EE vers 15 microservices Spring Boot/Kafka",
            "Mise en place de pratiques TDD et pair programming (couverture tests > 85 pct)",
        ],
        "base_years": 9
    },
    {
        "titre": "Business Analyst Analyste Fonctionnel",
        "type_poste": "Business Analyst",
        "profil": "Business Analyst avec 5 ans d experience dans le secteur bancaire, specialise dans la redaction de specifications fonctionnelles et la gestion de projet.",
        "skills": [
            ("Analyse fonctionnelle", "Analyse", "Expert", 5.0),
            ("UML", "Modelisation", "Expert", 5.0),
            ("BPMN", "Modelisation", "Confirme", 4.0),
            ("SQL", "Base de donnees", "Confirme", 4.0),
            ("Jira", "Outils", "Expert", 5.0),
            ("Confluence", "Outils", "Expert", 5.0),
            ("Agile/Scrum", "Methode", "Expert", 5.0),
            ("Gestion de projet", "Management", "Confirme", 4.0),
            ("Excel", "Bureautique", "Expert", 6.0),
            ("Power BI", "Business Intelligence", "Confirme", 3.0),
        ],
        "certifications": [("CBAP Certified Business Analysis Professional", "IIBA", 2022)],
        "missions": [
            "Redaction des specifications fonctionnelles pour refonte du systeme de trading",
            "Animation des ateliers metier avec les equipes Front Office et IT",
            "Coordination entre equipes developpement et metier methode Agile",
        ],
        "base_years": 5
    },
]


def make_date_range(base_date, years_ago_start, duration_years):
    start = base_date - relativedelta(years=years_ago_start)
    end = start + relativedelta(months=int(duration_years * 12))
    if end >= base_date:
        return start.strftime("%m/%Y"), None
    return start.strftime("%m/%Y"), end.strftime("%m/%Y")


def generate_cv(profile, prenom, nom, seed):
    rng = random.Random(seed)
    today = date.today()

    entreprises = rng.sample(ENTREPRISES, min(3, len(ENTREPRISES)))
    ecole = rng.choice(ECOLES)

    competences = []
    for (nom_skill, cat, niveau, annees) in profile["skills"]:
        annees_var = round(annees + rng.uniform(-0.5, 0.5), 1)
        competences.append(Competence(
            nom=nom_skill,
            categorie=cat,
            niveau=niveau,
            annees_experience=max(0.5, annees_var),
            confidence=round(rng.uniform(0.85, 1.0), 2),
        ))

    base_years = profile["base_years"]
    exp1_start, exp1_end = make_date_range(today, base_years, base_years / 3)
    exp2_start, exp2_end = make_date_range(today, int(base_years * 2 / 3), base_years / 3)
    exp3_start, exp3_end = make_date_range(today, int(base_years / 3), base_years / 3)

    experiences = [
        Experience(
            titre=profile["titre"],
            entreprise=entreprises[0],
            date_debut=exp1_start,
            date_fin=exp1_end,
            en_cours=exp1_end is None,
            missions=profile["missions"],
            technologies=[s[0] for s in profile["skills"][:5]],
        ),
        Experience(
            titre=profile["titre"].replace("Senior", "").replace("Expert", "Confirme").strip(),
            entreprise=entreprises[1] if len(entreprises) > 1 else entreprises[0],
            date_debut=exp2_start,
            date_fin=exp2_end,
            en_cours=False,
            missions=profile["missions"][:2],
            technologies=[s[0] for s in profile["skills"][2:7]],
        ),
        Experience(
            titre="Developpeur Junior Stagiaire",
            entreprise=entreprises[2] if len(entreprises) > 2 else entreprises[0],
            date_debut=exp3_start,
            date_fin=exp3_end,
            en_cours=False,
            missions=["Developpement de fonctionnalites", "Correction de bugs"],
            technologies=[s[0] for s in profile["skills"][:3]],
        ),
    ]

    formations = [Formation(diplome=ecole[0], etablissement=ecole[1], annee=today.year - base_years)]

    certifications = [
        Certification(nom=c[0], organisme=c[1], annee=c[2])
        for c in profile.get("certifications", [])
    ]

    langues = [
        Langue(langue="Francais", niveau="Natif"),
        Langue(langue="Anglais", niveau="Courant" if rng.random() > 0.3 else "Professionnel"),
    ]

    return CVData(
        identite=Identite(
            nom=nom,
            prenom=prenom,
            email=f"{prenom.lower()}.{nom.lower().replace(' ', '')}@gmail.com",
            localisation=rng.choice(["Paris", "Lyon", "Bordeaux", "Nantes", "Lille", "Toulouse"]),
        ),
        titre_professionnel=profile["titre"],
        type_poste=profile["type_poste"],
        profil=profile["profil"],
        competences=competences,
        experiences=experiences,
        formations=formations,
        certifications=certifications,
        langues=langues,
        metadata=Metadata(
            date_extraction=datetime.now().isoformat(),
            source_fichier=f"synthetic_{prenom.lower()}_{nom.lower().replace(' ', '')}.pdf",
            score_completude=round(rng.uniform(0.85, 0.99), 2),
            version_pipeline="1.1.0-synthetic",
        ),
    )


def main():
    vdb = VectorStoreManager()
    print(f"Vector store actuel : {vdb.collection.count()} CVs indexes")

    existing = vdb.collection.get(include=["metadatas"])
    already_indexed = {m.get("source", "") for m in existing["metadatas"]}
    print(f"Deja indexes : {already_indexed}")

    # 1. Index existing JSON CVs not yet indexed
    import glob as glob_mod
    json_files = glob_mod.glob(str(PROJECT_ROOT / "output" / "*.json"))
    indexed_json = 0
    for jf in json_files:
        fname = os.path.basename(jf)
        if fname in already_indexed or fname.endswith("_result.json"):
            print(f"  [SKIP] {fname}")
            continue
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
            cv = CVData(**data)
            vdb.add_cv(cv, fname)
            print(f"  [OK] JSON: {cv.identite.prenom} {cv.identite.nom} ({fname})")
            indexed_json += 1
        except Exception as e:
            print(f"  [ERR] {fname}: {e}")

    # 2. Generate and index synthetic CVs (2 per profile = 30 total)
    print(f"\nGeneration de {len(PROFILES) * 2} CVs synthetiques...")
    indexed_synth = 0
    for i, profile in enumerate(PROFILES):
        for j in range(2):
            seed = i * 100 + j
            prenom = PRENOMS[(i * 2 + j) % len(PRENOMS)]
            nom = NOMS[(i * 3 + j + 7) % len(NOMS)]
            safe_type = profile["type_poste"].lower().replace(" ", "_").replace("/", "_")
            doc_id = f"synthetic_{safe_type}_{j+1}_{prenom.lower()}.pdf"

            if doc_id in already_indexed:
                print(f"  [SKIP] {doc_id}")
                continue
            try:
                cv = generate_cv(profile, prenom, nom, seed)
                vdb.add_cv(cv, doc_id)
                print(f"  [OK] {cv.identite.prenom} {cv.identite.nom} - {profile['titre']}")
                indexed_synth += 1
            except Exception as e:
                print(f"  [ERR] {doc_id}: {e}")

    total = vdb.collection.count()
    print(f"\n=== Termine ===")
    print(f"JSON reels indexes  : {indexed_json}")
    print(f"Synthetiques indexes: {indexed_synth}")
    print(f"Total vector store  : {total} CVs")


if __name__ == "__main__":
    main()
