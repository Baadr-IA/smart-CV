from typing import List, Optional, Union
from pydantic import BaseModel, Field, AliasChoices, field_validator, model_validator

class Identite(BaseModel):
    nom: str
    prenom: Optional[str] = "Candidat" # Par défaut si absent
    email: Optional[str] = None
    localisation: Optional[str] = None
    telephone: Optional[str] = None
    linkedin: Optional[str] = None

    @model_validator(mode='before')
    @classmethod
    def split_name_if_needed(cls, data):
        """Si le nom contient un espace et que le prénom est absent, on split."""
        if isinstance(data, dict):
            nom = data.get("nom", "")
            prenom = data.get("prenom")
            
            if nom and (not prenom or prenom == "Candidat") and " " in nom:
                parts = nom.split(" ", 1)
                data["prenom"] = parts[0]
                data["nom"] = parts[1]
        return data

class Competence(BaseModel):
    nom: str
    categorie: Optional[str] = "Général"
    niveau: Optional[str] = "Intermédiaire"
    annees_experience: Optional[int] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    justification: Optional[str] = None

class Experience(BaseModel):
    titre: str
    entreprise: str
    date_debut: str
    date_fin: Optional[str] = None
    en_cours: bool = False
    projet: Optional[str] = None
    equipe: Optional[Union[str, int]] = None
    methodologie: Optional[str] = None
    missions: List[str] = Field(default_factory=list)
    technologies: List[str] = Field(default_factory=list)
    resultats: List[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    justification: Optional[str] = None

class Formation(BaseModel):
    diplome: str = Field(..., validation_alias=AliasChoices('diplome', 'titre', 'nom', 'formation', 'etude'))
    etablissement: str = Field(..., validation_alias=AliasChoices('etablissement', 'ecole', 'universite', 'lieu'))
    annee: Optional[Union[int, str]] = None

class Certification(BaseModel):
    nom: str
    organisme: Optional[str] = None
    annee: Optional[Union[int, str]] = None
    score: Optional[str] = None

class Langue(BaseModel):
    langue: str
    niveau: str
    certification: Optional[str] = None

class Metadata(BaseModel):
    date_extraction: Optional[str] = None
    source_fichier: Optional[str] = None
    fichier_word: Optional[str] = None # Nouveau champ
    score_completude: float = 0.0
    champs_incertains: List[str] = Field(default_factory=list)
    version_pipeline: str = "1.1.0"
    char_count: Optional[int] = None
    method_parsing: Optional[str] = None
    validation_report: Optional[dict] = None

class CVData(BaseModel):
    identite: Identite = Field(validation_alias=AliasChoices(
        'identite', 'informations_personnelles', 'personal_info', 'personnal_info', 
        'personnal_informations', 'informations_du_candidat'
    ))
    titre_professionnel: Optional[str] = None
    type_poste: Optional[str] = None
    profil: Optional[str] = None
    
    competences: List[Union[Competence, str]] = Field(default_factory=list)
    experiences: List[Experience] = Field(default_factory=list)
    formations: List[Formation] = Field(default_factory=list)
    certifications: List[Certification] = Field(default_factory=list)
    langues: List[Langue] = Field(default_factory=list)
    centres_interet: Optional[str] = None
    metadata: Metadata

    @field_validator('competences', mode='before')
    @classmethod
    def transform_skills(cls, v):
        """Déplie les structures complexes (ex: Vision) en une liste plate de compétences."""
        if not isinstance(v, list):
            return []
        
        flat_list = []
        for item in v:
            if isinstance(item, str): 
                flat_list.append({"nom": item})
            elif isinstance(item, dict):
                if "nom" in item:
                    flat_list.append(item)
                elif "details" in item and isinstance(item["details"], list):
                    cat = item.get("categorie", "Général")
                    for detail in item["details"]:
                        flat_list.append({"nom": str(detail), "categorie": cat})
                elif "competences" in item and isinstance(item["competences"], list):
                    cat = item.get("categorie", "Général")
                    for skill in item["competences"]:
                        flat_list.append({"nom": str(skill), "categorie": cat})
                else:
                    flat_list.append({"nom": str(list(item.values())[0])})
            else:
                flat_list.append({"nom": str(item)})
                
        return flat_list


# Ensure all annotations are resolved in Pydantic v2.
CVData.model_rebuild()
