"""Prompt text and response parsing from Supplementary Methods S2."""
import re

SYSTEM_PROMPT = (
    "Output MUST be exactly one word: normal or disease.\n"
    "Do not output punctuation, quotes, JSON, or any extra words."
)
MODALITY_TEXT = {"mri": "brain MRI scan", "oct": "retina OCT scan",
                 "xray": "chest X-ray image", "fundus": "retinal fundus photograph"}
DISEASE_CUES = {
    "mri": "Focus on brain tissue structure, symmetry, and signal characteristics. Determine whether there are true abnormalities such as tumors, edema, mass effect, or focal signal changes.",
    "oct": "Focus on retina layer integrity and the foveal contour. Determine whether there are pathological features such as drusen-like elevations, intraretina or subretina fluid, or abnormal layer thickening or disruption.",
    "xray": "Focus on the lung fields, heart borders, and costophrenic angles. Determine whether there are pneumonia-related abnormalities such as focal opacity, consolidation, or interstitial infiltrates.",
}
ARTEFACT_CUE = "The image may contain imaging artefacts or noise. Artefacts are not disease, so ignore them and focus only on true anatomical structures."
PROMPT_CONFIGS = ("base", "disease", "artefact", "combined")


def prompts(configuration="base"):
    if configuration not in PROMPT_CONFIGS:
        raise ValueError(f"Unknown prompt configuration: {configuration}")
    result = {}
    for modality, description in MODALITY_TEXT.items():
        parts = [f"This is a {description}."]
        if configuration in {"disease", "combined"}:
            if modality not in DISEASE_CUES:
                continue
            parts.append(DISEASE_CUES[modality])
        if configuration in {"artefact", "combined"}:
            parts.append(ARTEFACT_CUE)
        parts += ["Question: Does this image show normal anatomy or signs of disease?", "Answer:"]
        result[modality] = "\n".join(parts)
    return result


def parse_label(text):
    """Case-insensitive label matching; both labels or neither is unparseable."""
    labels = set(re.findall(r"\b(normal|disease)\b", str(text or "").lower()))
    return {frozenset({"normal"}): 0, frozenset({"disease"}): 1}.get(frozenset(labels))


def validate_demonstrations(frame):
    """Require the training-only six-view cases in Supplementary Method S3.5."""
    required = {"dataset", "fileindex", "binarylabel", "severity", "corruption", "use_type"}
    if required - set(frame.columns):
        raise ValueError(f"Few-shot CSV missing columns: {sorted(required - set(frame.columns))}")
    if frame.empty or not frame.use_type.astype(str).str.strip().str.lower().eq("train").all():
        raise ValueError("Few-shot demonstrations must all belong to the training split")
    for modality, group in frame.groupby("dataset", sort=False):
        labels = []
        for source, case in group.groupby("fileindex", sort=False):
            artefacts = case.corruption.astype(str).str.lower().replace(
                {"gaussian_noise": "noise", "bias_field": "biasfield", "cropping": "crop", "rotation": "rotate"})
            if len(case) != 6 or set(artefacts) != {"clean", "noise", "biasfield", "crop", "rotate", "motion"}:
                raise ValueError(f"{modality}/{source}: expected one clean and five distinct seen artefacts")
            if case.binarylabel.nunique() != 1 or str(case.binarylabel.iloc[0]) not in {"0", "1"}:
                raise ValueError(f"{modality}/{source}: inconsistent binary labels")
            labels.append(int(case.binarylabel.iloc[0]))
        if sorted(labels) not in ([0], [1], [0, 1], [0, 0, 1, 1]):
            raise ValueError(f"{modality}: expected 1N, 1D, balanced two-case, or balanced four-case condition")
