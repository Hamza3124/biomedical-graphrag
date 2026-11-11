// ============================================================================
// CYPHIR VALIDATION QUERIES FOR BIOMEDICAL CAUSAL ANALYSIS
// Project: Diseases Cause Oriented Analysis using RAG and Knowledge Graphs
// Pathologies ciblées: Alzheimer's Disease (DOID:10652) & Type 2 Diabetes (DOID:9352)
// ============================================================================

// ----------------------------------------------------------------------------
// REQUÊTE 1 : Profilage multi-relationnel direct (Voisinage 1-hop)
// Objectif : Dénombrer et caractériser tous les voisins directs d'Alzheimer
// (Gènes associés/modulés, symptômes cliniques, tissus anatomiques, traitements).
// ----------------------------------------------------------------------------
MATCH (d:Disease {name: "Alzheimer's disease"})-[r]-(neighbor)
RETURN type(r) AS RelationType,
       labels(neighbor)[0] AS TargetLabel,
       count(neighbor) AS NeighborCount,
       collect(neighbor.name)[0..5] AS SampleEntities
ORDER BY NeighborCount DESC;


// ----------------------------------------------------------------------------
// REQUÊTE 2 : Extraction de la chaîne de causalité complète
// Schéma : (Maladie) -> (Gène muté/associé) -> (Voie métabolique altérée)
//          et (Maladie) -> (Symptôme/Phénotype clinique)
// Objectif : Reconstruire le mécanisme physiopathologique complet du niveau
// moléculaire au niveau clinique.
// ----------------------------------------------------------------------------
MATCH (d:Disease {name: "Alzheimer's disease"})-[:ASSOCIATES_WITH]->(g:Gene)
MATCH (g)-[:PARTICIPATES_IN]->(p:Pathway)
MATCH (d)-[:MANIFESTS]->(s:Symptom)
WHERE p.name CONTAINS "Alzheimer" OR p.name CONTAINS "Apoptosis" OR p.name CONTAINS "Signaling"
RETURN d.name AS Pathologie,
       g.name AS GeneAssocie,
       g.identifier_type + ":" + g.external_id AS GeneEntrezID,
       p.name AS VoieMetabolique,
       collect(DISTINCT s.name)[0..3] AS SymptomesAssocies
ORDER BY g.name ASC, p.name ASC
LIMIT 25;


// ----------------------------------------------------------------------------
// REQUÊTE 3 : Recherche de voisinage k-hop (profondeur 2 à 3)
// Objectif : Détecter les chaînes causales régulatrices :
// (Maladie) -[ASSOCIATES_WITH]-> (Gène A) -[REGULATES|INTERACTS_WITH]-> (Gène B) -[PARTICIPATES_IN]-> (Voie)
// ----------------------------------------------------------------------------
MATCH path = (d:Disease {name: "Alzheimer's disease"})-[:ASSOCIATES_WITH]->(g1:Gene)-[:REGULATES]->(g2:Gene)-[:PARTICIPATES_IN]->(p:Pathway)
RETURN d.name AS Disease,
       g1.name AS RegulatingGene,
       g2.name AS RegulatedGene,
       p.name AS DownstreamPathway,
       length(path) AS HopLength
ORDER BY g1.name, g2.name
LIMIT 20;


// ----------------------------------------------------------------------------
// REQUÊTE 4 : Analyse thérapeutique et cibles moléculaires (Drug Repurposing)
// Objectif : Relier les molécules thérapeutiques (Donepezil, Memantine, Galantamine, Rivastigmine)
// à leurs cibles géniques et voies métaboliques cibles.
// ----------------------------------------------------------------------------
MATCH (c:Compound)-[:TREATS]->(d:Disease {name: "Alzheimer's disease"})
OPTIONAL MATCH (c)-[rb:BINDS|DOWNREGULATES|UPREGULATES]->(g:Gene)
OPTIONAL MATCH (g)-[:PARTICIPATES_IN]->(p:Pathway)
RETURN c.name AS MoleculeTherapeutique,
       c.external_id AS DrugBankID,
       type(rb) AS ActionSurGene,
       g.name AS GeneCible,
       collect(DISTINCT p.name)[0..3] AS VoiesImpactees
ORDER BY c.name, g.name;


// ----------------------------------------------------------------------------
// REQUÊTE 5 : Analyse causale comparative cross-pathologies (Axe Métabolique / Neurodégénératif)
// Contexte : Hypothèse du "Diabète de Type 3" - Résistance à l'insuline cérébrale
// Objectif : Identifier les gènes et voies métaboliques partagés entre
// Alzheimer's Disease et Type 2 Diabetes Mellitus.
// ----------------------------------------------------------------------------
MATCH (ad:Disease {name: "Alzheimer's disease"})-[:ASSOCIATES_WITH]->(shared_gene:Gene)<-[:ASSOCIATES_WITH]-(t2d:Disease {name: "Type 2 diabetes mellitus"})
MATCH (shared_gene)-[:PARTICIPATES_IN]->(p:Pathway)
RETURN shared_gene.name AS GenePartage,
       collect(DISTINCT p.name)[0..4] AS VoiesMetaboliquesCommunes,
       count(DISTINCT p) AS NombreVoiesTotal
ORDER BY NombreVoiesTotal DESC
LIMIT 15;


// ----------------------------------------------------------------------------
// REQUÊTE 6 : Arborescence "Human Lifestyle" et Mécanismes Causaux Comportementaux
// Contexte : Facteurs modifiables (Nutrition, Exercice, Sommeil, Stress, Habitudes)
// Objectif : Traverser l'arborescence du mode de vie et identifier les mécanismes
// moléculaires protecteurs ou aggravants liant le comportement aux pathologies.
// ----------------------------------------------------------------------------
MATCH (cat:LifestyleCategory)<-[:SUBCLASS_OF]-(lf:LifestyleFactor)
OPTIONAL MATCH (lf)-[r:PROTECTS|AGGRAVATES]->(d:Disease)
OPTIONAL MATCH (lf)-[rg:UPREGULATES|DOWNREGULATES]->(g:Gene)
RETURN cat.name AS CategorieLifestyle,
       lf.name AS FacteurDeModeDeVie,
       type(r) AS ImpactCliniqueDirect,
       d.name AS PathologieCible,
       type(rg) AS RegulationMoleculaire,
       g.name AS GeneCible
ORDER BY CategorieLifestyle, FacteurDeModeDeVie;


