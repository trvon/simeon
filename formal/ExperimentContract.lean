import Std

namespace Simeon.Experiment

inductive TrainingRegime where
  | artifactFree
  | corpusAdaptive
  | evaluationOnly
  deriving BEq, Repr

inductive EvidenceSource where
  | fixedAlgorithm
  | corpusStatistics
  | relevanceLabels
  | pretrainedModel
  deriving BEq, Repr

def sourceAllowed : TrainingRegime → EvidenceSource → Bool
  | .artifactFree, .fixedAlgorithm => true
  | .corpusAdaptive, .fixedAlgorithm => true
  | .corpusAdaptive, .corpusStatistics => true
  | .evaluationOnly, _ => true
  | _, _ => false

theorem artifactFreeRejectsCorpusStatistics :
    sourceAllowed .artifactFree .corpusStatistics = false := rfl

theorem artifactFreeRejectsLabels :
    sourceAllowed .artifactFree .relevanceLabels = false := rfl

theorem artifactFreeRejectsPretrainedModels :
    sourceAllowed .artifactFree .pretrainedModel = false := rfl

inductive EncoderMechanism where
  | asciiCaseFold
  | wordBoundaryNGrams
  | fixedFeatureHash
  | fixedProjection
  | fixedScoreFusion
  | fixedCoordinateGate
  | fixedFeaturePartition
  | hashedDocumentFrequency
  | corpusTermStatistics
  | corpusCoordinateMoments
  | corpusFamilyRms
  | queryCorpusScoreMoments
  | relevanceTunedWeights
  | pretrainedEncoder
  deriving BEq, Repr

def mechanismSource : EncoderMechanism → EvidenceSource
  | .asciiCaseFold => .fixedAlgorithm
  | .wordBoundaryNGrams => .fixedAlgorithm
  | .fixedFeatureHash => .fixedAlgorithm
  | .fixedProjection => .fixedAlgorithm
  | .fixedScoreFusion => .fixedAlgorithm
  | .fixedCoordinateGate => .fixedAlgorithm
  | .fixedFeaturePartition => .fixedAlgorithm
  | .hashedDocumentFrequency => .corpusStatistics
  | .corpusTermStatistics => .corpusStatistics
  | .corpusCoordinateMoments => .corpusStatistics
  | .corpusFamilyRms => .corpusStatistics
  | .queryCorpusScoreMoments => .corpusStatistics
  | .relevanceTunedWeights => .relevanceLabels
  | .pretrainedEncoder => .pretrainedModel

def recipeAllowed (regime : TrainingRegime) (recipe : List EncoderMechanism) : Bool :=
  recipe.all (fun mechanism => sourceAllowed regime (mechanismSource mechanism))

def compactWordBoundaryRecipe : List EncoderMechanism :=
  [.asciiCaseFold, .wordBoundaryNGrams, .fixedFeatureHash, .fixedProjection]

theorem compactWordBoundaryRecipeIsArtifactFree :
    recipeAllowed .artifactFree compactWordBoundaryRecipe = true := rfl

theorem artifactFreeRecipeRejectsCorpusStatistics :
    recipeAllowed .artifactFree [.wordBoundaryNGrams, .corpusTermStatistics] = false := rfl

def compactHashedIdfRecipe : List EncoderMechanism :=
  compactWordBoundaryRecipe ++ [.hashedDocumentFrequency]

theorem compactHashedIdfRecipeIsCorpusAdaptive :
    recipeAllowed .corpusAdaptive compactHashedIdfRecipe = true := rfl

theorem compactHashedIdfRecipeIsNotArtifactFree :
    recipeAllowed .artifactFree compactHashedIdfRecipe = false := rfl

-- Increasing the fixed projection width changes representation geometry, not
-- the evidence consumed by the recipe. The 768-dimensional quality point
-- therefore remains corpus-adaptive and does not become relevance-trained.
def qualityHashedIdfRecipe : List EncoderMechanism :=
  compactHashedIdfRecipe

theorem qualityHashedIdfRecipeIsCorpusAdaptive :
    recipeAllowed .corpusAdaptive qualityHashedIdfRecipe = true := rfl

def fixedProjectionGeometryValid (sketchDim outputDim : Nat) : Bool :=
  outputDim > 0 && outputDim ≤ sketchDim

theorem qualityProjectionGeometryIsValid :
    fixedProjectionGeometryValid 8192 768 = true := by decide

def qualityHashedIdfFusionRecipe : List EncoderMechanism :=
  qualityHashedIdfRecipe ++ [.corpusTermStatistics, .fixedScoreFusion]

theorem qualityHashedIdfFusionRecipeIsCorpusAdaptive :
    recipeAllowed .corpusAdaptive qualityHashedIdfFusionRecipe = true := rfl

theorem qualityHashedIdfFusionRecipeIsNotArtifactFree :
    recipeAllowed .artifactFree qualityHashedIdfFusionRecipe = false := rfl

-- Centering/diagonal calibration consumes document-only coordinate moments.
-- The runtime gate and its frozen thresholds are fixed algorithms, but the
-- complete recipe remains corpus-adaptive because IDF and moments are fitted
-- from the deployment corpus.
def coordinateCalibratedHashedIdfRecipe : List EncoderMechanism :=
  qualityHashedIdfRecipe ++ [.corpusCoordinateMoments, .fixedCoordinateGate]

theorem coordinateCalibratedHashedIdfRecipeIsCorpusAdaptive :
    recipeAllowed .corpusAdaptive coordinateCalibratedHashedIdfRecipe = true := rfl

theorem coordinateCalibratedHashedIdfRecipeIsNotArtifactFree :
    recipeAllowed .artifactFree coordinateCalibratedHashedIdfRecipe = false := rfl

-- Feature-family charts use fixed partitioning and score composition. Separate
-- document-frequency tables and optional RMS scales are corpus statistics;
-- relevance-selected runtime weights would cross the training-free boundary.
def featureFamilyAtlasRecipe : List EncoderMechanism :=
  qualityHashedIdfRecipe ++ [.fixedFeaturePartition, .fixedScoreFusion]

def rmsFeatureFamilyAtlasRecipe : List EncoderMechanism :=
  featureFamilyAtlasRecipe ++ [.corpusFamilyRms]

theorem featureFamilyAtlasRecipeIsCorpusAdaptive :
    recipeAllowed .corpusAdaptive featureFamilyAtlasRecipe = true := rfl

theorem rmsFeatureFamilyAtlasRecipeIsCorpusAdaptive :
    recipeAllowed .corpusAdaptive rmsFeatureFamilyAtlasRecipe = true := rfl

theorem rmsFeatureFamilyAtlasRecipeIsNotArtifactFree :
    recipeAllowed .artifactFree rmsFeatureFamilyAtlasRecipe = false := rfl

theorem corpusAdaptiveFamilyAtlasRejectsLabelTunedWeights :
    recipeAllowed .corpusAdaptive
      (featureFamilyAtlasRecipe ++ [.relevanceTunedWeights]) = false := rfl

-- A residual keeps the combined chart as its base and appends a fixed family
-- partition. Raw-cosine and rank fusion are fixed algorithms. Per-query
-- z-normalization additionally observes the two score fields over the
-- deployment corpus, but consumes neither qrels nor pretrained evidence.
def featureResidualRecipe : List EncoderMechanism :=
  featureFamilyAtlasRecipe

def zScoreFeatureResidualRecipe : List EncoderMechanism :=
  featureResidualRecipe ++ [.queryCorpusScoreMoments]

theorem featureResidualRecipeIsCorpusAdaptive :
    recipeAllowed .corpusAdaptive featureResidualRecipe = true := rfl

theorem zScoreFeatureResidualRecipeIsCorpusAdaptive :
    recipeAllowed .corpusAdaptive zScoreFeatureResidualRecipe = true := rfl

theorem zScoreFeatureResidualRecipeIsNotArtifactFree :
    recipeAllowed .artifactFree zScoreFeatureResidualRecipe = false := rfl

theorem corpusAdaptiveResidualRejectsLabelTunedWeights :
    recipeAllowed .corpusAdaptive
      (featureResidualRecipe ++ [.relevanceTunedWeights]) = false := rfl

structure FamilyChartAllocation where
  characterDim : Nat
  wordDim : Nat
  storageBudgetDim : Nat
  deriving Repr, BEq

def FamilyChartAllocation.usesDeclaredBudget (allocation : FamilyChartAllocation) : Bool :=
  allocation.characterDim > 0 &&
    allocation.wordDim > 0 &&
    allocation.characterDim + allocation.wordDim == allocation.storageBudgetDim

theorem familyAllocation512x256Uses768Floats :
    FamilyChartAllocation.usesDeclaredBudget {
      characterDim := 512
      wordDim := 256
      storageBudgetDim := 768
    } = true := by decide

theorem familyAllocationCannotHideEnsembleStorage
    (allocation : FamilyChartAllocation)
    (valid : allocation.usesDeclaredBudget = true) :
    allocation.characterDim + allocation.wordDim = allocation.storageBudgetDim := by
  simp [FamilyChartAllocation.usesDeclaredBudget] at valid
  exact valid.2

structure FeatureResidualAllocation where
  baseDim : Nat
  characterResidualDim : Nat
  wordResidualDim : Nat
  storageBudgetDim : Nat
  deriving Repr, BEq

def FeatureResidualAllocation.usesDeclaredBudget
    (allocation : FeatureResidualAllocation) : Bool :=
  allocation.baseDim > 0 &&
    allocation.characterResidualDim > 0 &&
    allocation.wordResidualDim > 0 &&
    allocation.baseDim + allocation.characterResidualDim + allocation.wordResidualDim ==
      allocation.storageBudgetDim

theorem featureResidual768x192x64Uses1024Floats :
    FeatureResidualAllocation.usesDeclaredBudget {
      baseDim := 768
      characterResidualDim := 192
      wordResidualDim := 64
      storageBudgetDim := 1024
    } = true := by decide

theorem featureResidualAllocationCannotHideEnsembleStorage
    (allocation : FeatureResidualAllocation)
    (valid : allocation.usesDeclaredBudget = true) :
    allocation.baseDim + allocation.characterResidualDim + allocation.wordResidualDim =
      allocation.storageBudgetDim := by
  simp [FeatureResidualAllocation.usesDeclaredBudget] at valid
  exact valid.2

/-- Label-free observations used to decide whether a family residual may
augment the safe combined chart. Explicit observation bits prevent missing
counters from being interpreted as measured zeroes. -/
structure FeatureResidualObservation where
  wordEnergyPerMillion : Nat := 0
  familyOverlapPerThousand : Nat := 0
  baseFamilyOverlapPerThousand : Nat := 0
  wordEnergyObserved : Bool := false
  familyOverlapObserved : Bool := false
  baseFamilyOverlapObserved : Bool := false
  deriving Repr, BEq

structure FeatureResidualConfig where
  minWordEnergyPerMillion : Nat := 0
  maxWordEnergyPerMillion : Nat := 1000000
  minFamilyOverlapPerThousand : Nat := 0
  maxFamilyOverlapPerThousand : Nat := 1000
  minBaseFamilyOverlapPerThousand : Nat := 0
  maxBaseFamilyOverlapPerThousand : Nat := 1000
  deriving Repr, BEq

def FeatureResidualAdmissible
    (cfg : FeatureResidualConfig) (observation : FeatureResidualObservation) : Prop :=
  observation.wordEnergyObserved = true ∧
    observation.familyOverlapObserved = true ∧
    observation.baseFamilyOverlapObserved = true ∧
    cfg.minWordEnergyPerMillion ≤ observation.wordEnergyPerMillion ∧
    observation.wordEnergyPerMillion ≤ cfg.maxWordEnergyPerMillion ∧
    cfg.minFamilyOverlapPerThousand ≤ observation.familyOverlapPerThousand ∧
    observation.familyOverlapPerThousand ≤ cfg.maxFamilyOverlapPerThousand ∧
    cfg.minBaseFamilyOverlapPerThousand ≤ observation.baseFamilyOverlapPerThousand ∧
    observation.baseFamilyOverlapPerThousand ≤ cfg.maxBaseFamilyOverlapPerThousand

instance featureResidualAdmissibleDecidable
    (cfg : FeatureResidualConfig) (observation : FeatureResidualObservation) :
    Decidable (FeatureResidualAdmissible cfg observation) := by
  unfold FeatureResidualAdmissible
  infer_instance

inductive FeatureResidualAction where
  | base
  | augment
  deriving BEq, Repr, DecidableEq

/-- The combined chart is the total fallback: residual evidence may augment it
only after every declared observation has been measured and admitted. -/
def selectFeatureResidualAction
    (cfg : FeatureResidualConfig) (observation : FeatureResidualObservation) :
    FeatureResidualAction :=
  if FeatureResidualAdmissible cfg observation then .augment else .base

theorem featureResidualRoute_augment_iff_admissible
    (cfg : FeatureResidualConfig) (observation : FeatureResidualObservation) :
    selectFeatureResidualAction cfg observation = .augment ↔
      FeatureResidualAdmissible cfg observation := by
  by_cases h : FeatureResidualAdmissible cfg observation <;>
    simp [selectFeatureResidualAction, h]

theorem featureResidualRoute_rejected_isBase
    {cfg : FeatureResidualConfig} {observation : FeatureResidualObservation}
    (hRejected : ¬ FeatureResidualAdmissible cfg observation) :
    selectFeatureResidualAction cfg observation = .base := by
  simp [selectFeatureResidualAction, hRejected]

theorem featureResidualRoute_unobserved_isBase
    {cfg : FeatureResidualConfig} {observation : FeatureResidualObservation}
    (hMissing : observation.wordEnergyObserved ≠ true ∨
      observation.familyOverlapObserved ≠ true ∨
      observation.baseFamilyOverlapObserved ≠ true) :
    selectFeatureResidualAction cfg observation = .base := by
  apply featureResidualRoute_rejected_isBase
  intro hAdmitted
  rcases hMissing with hWord | hFamily | hBaseFamily
  · exact hWord hAdmitted.1
  · exact hFamily hAdmitted.2.1
  · exact hBaseFamily hAdmitted.2.2.1

/-- Integer observations mirror the YAMS retrieval-coordinate boundary without
pulling floating-point semantics into the experiment contract. A zero counter
is evidence only when its producer also marks that class as observed. -/
structure DimensionChartObservation where
  overlapPerThousand : Nat := 0
  distortionPerMillion : Nat := 0
  energyDeviationPerMillion : Nat := 0
  distortionObserved : Bool := false
  uncertaintyObserved : Bool := false
  deriving Repr, BEq

structure DimensionChartConfig where
  minOverlapPerThousand : Nat := 0
  maxDistortionPerMillion : Nat := 0
  maxEnergyDeviationPerMillion : Nat := 0
  augmentRejected : Bool := true
  deriving Repr, BEq

def DimensionChartAdmissible
    (cfg : DimensionChartConfig) (observation : DimensionChartObservation) : Prop :=
  observation.distortionObserved = true ∧
    observation.uncertaintyObserved = true ∧
    cfg.minOverlapPerThousand ≤ observation.overlapPerThousand ∧
    observation.distortionPerMillion ≤ cfg.maxDistortionPerMillion ∧
    observation.energyDeviationPerMillion ≤ cfg.maxEnergyDeviationPerMillion

instance dimensionChartAdmissibleDecidable
    (cfg : DimensionChartConfig) (observation : DimensionChartObservation) :
    Decidable (DimensionChartAdmissible cfg observation) := by
  unfold DimensionChartAdmissible
  infer_instance

inductive DimensionRoutingAction where
  | full
  | augment
  | narrow
  deriving BEq, Repr, DecidableEq

/-- A local prefix replaces the full chart only after every declared evidence
class is observed and admitted. Rejected charts augment or fall back. -/
def selectDimensionRoutingAction
    (cfg : DimensionChartConfig) (observation : DimensionChartObservation) :
    DimensionRoutingAction :=
  if DimensionChartAdmissible cfg observation then
    .narrow
  else if cfg.augmentRejected then
    .augment
  else
    .full

theorem dimensionRoute_narrow_iff_admissible
    (cfg : DimensionChartConfig) (observation : DimensionChartObservation) :
    selectDimensionRoutingAction cfg observation = .narrow ↔
      DimensionChartAdmissible cfg observation := by
  by_cases h : DimensionChartAdmissible cfg observation
  · simp [selectDimensionRoutingAction, h]
  · cases hAugment : cfg.augmentRejected <;>
      simp [selectDimensionRoutingAction, h, hAugment]

theorem dimensionRoute_unobserved_neverNarrows
    {cfg : DimensionChartConfig} {observation : DimensionChartObservation}
    (hMissing : observation.distortionObserved ≠ true ∨
      observation.uncertaintyObserved ≠ true) :
    selectDimensionRoutingAction cfg observation ≠ .narrow := by
  intro hNarrow
  have hAdmitted :=
    (dimensionRoute_narrow_iff_admissible cfg observation).mp hNarrow
  rcases hMissing with hDistortion | hUncertainty
  · exact hDistortion hAdmitted.1
  · exact hUncertainty hAdmitted.2.1

theorem dimensionRoute_rejected_isNotNarrow
    {cfg : DimensionChartConfig} {observation : DimensionChartObservation}
    (hRejected : ¬ DimensionChartAdmissible cfg observation) :
    selectDimensionRoutingAction cfg observation ≠ .narrow := by
  intro hNarrow
  exact hRejected ((dimensionRoute_narrow_iff_admissible cfg observation).mp hNarrow)

inductive Phase where
  | exploration
  | frozen
  deriving BEq, Repr

structure Manifest where
  phase : Phase
  selectionManifest : Option String
  variantCount : Nat
  deriving Repr

def isFrozen : Phase → Bool
  | .exploration => false
  | .frozen => true

def mayRunHoldout (manifest : Manifest) : Bool :=
  isFrozen manifest.phase &&
    manifest.selectionManifest.isSome &&
    manifest.variantCount == 1

theorem explorationBlocksHoldout (lineage : Option String) (variants : Nat) :
    mayRunHoldout {
      phase := .exploration
      selectionManifest := lineage
      variantCount := variants
    } = false := by
  rfl

theorem frozenRequiresLineage (variants : Nat) :
    mayRunHoldout {
      phase := .frozen
      selectionManifest := none
      variantCount := variants
    } = false := by
  simp [mayRunHoldout]

theorem frozenRequiresOneVariant (lineage : String) (variants : Nat)
    (notOne : variants ≠ 1) :
    mayRunHoldout {
      phase := .frozen
      selectionManifest := some lineage
      variantCount := variants
    } = false := by
  simp [mayRunHoldout, notOne]

theorem permittedHoldoutIsFrozen (manifest : Manifest)
    (allowed : mayRunHoldout manifest = true) :
    manifest.phase = .frozen := by
  cases phase : manifest.phase with
  | exploration =>
      simp [mayRunHoldout, isFrozen, phase] at allowed
  | frozen => rfl

end Simeon.Experiment
