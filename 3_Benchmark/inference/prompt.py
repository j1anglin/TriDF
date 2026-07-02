from __future__ import annotations

"""
Centralized system hints and prompt snippets for the main benchmark task families.
"""

from textwrap import dedent
from typing import Dict, Tuple


def _tagged_prompt(modality: str, body: str) -> str:
    return f"<{modality}>\n{body.strip()}"


DETECTION_SYSTEM_HINT = dedent(
    """
You are a forensic media authenticity inspector.
Role:
- Decide whether each provided sample is authentic or manipulated.
- Explain concrete, observable artifacts found in the sample.

Hard Constraints:
- Follow the required output format exactly.
- The first line of your response must be a single line: either "Likely Authentic." or "Likely Manipulated."
- Use precise, neutral, technical language.
"""
).strip()

_DETECTION_PROMPT_TEMPLATE = dedent(
    """\
Scope & Tailoring:
- The data is {modality}. Tailor methods, vocabulary, and artifacts to this modality.

Your Task:
- Decide whether the provided sample is authentic or manipulated.
- Perform a detailed analysis of artifacts that appear inauthentic or indicative of synthesis/manipulation.
- Focus on concrete, observable evidence. Avoid speculation.

Guidelines:
- Be Thorough. Cover all noticeable artifacts and inconsistencies relevant to this modality.
- Be Accurate. Base claims only on what is present in the {modality}. Explain why each artifact is suspicious in technical terms.
- Avoid False Positives. Do not label authentic features as inauthentic. If uncertain, state the uncertainty and what additional evidence would be needed.
- Organize Your Response. Use clear headings for each artifact and include short evidence quotes.

Output Format:
1) First line (choose one, exactly):
   - Likely Authentic.
   - Likely Manipulated.
2) Artifact Findings
   For each finding, provide:
   - Title of artifact
   - Reason: brief technical rationale
"""
).strip()

_MODALITIES: Tuple[str, ...] = ("image", "video", "audio")
DETECTION_PROMPTS: Dict[str, str] = {
    modality: _tagged_prompt(modality, _DETECTION_PROMPT_TEMPLATE.format(modality=modality))
    for modality in _MODALITIES
}


PERCEPTION_OE_SYSTEM_HINT: str | None = None
_PERCEPTION_OE_BODY = dedent(
    """\
You are a forensic vision assistant working with synthetic, non-sensitive benchmark data.
This sample is part of a academic research benchmark on manipulated media.
Your Task:
Please perform a detailed analysis of artifacts in this sample that seem inauthentic.
Guidelines:
- Be Thorough: Offer a comprehensive analysis covering all noticeable artifacts.
- Be Accurate: Ensure your explanations correctly reflect the information present in the sample.
- Avoid False Positives: Do not incorrectly identify authentic features as inauthentic. Focus only on actual artifacts.
- Organize Your Response: Use clear headings and structure to separate your analysis for each artifact.
"""
).strip()
PERCEPTION_OE_PROMPTS: Dict[str, str] = {
    modality: _tagged_prompt(modality, _PERCEPTION_OE_BODY) for modality in _MODALITIES
}


PERCEPTION_MC_SYSTEM_HINT = dedent(
    """
You are a forensic media authenticity inspector.

Task:
- Given a sample (e.g., image, audio, video) and a list of artifact options labeled A–E, select all options that are clearly present.

Output Constraints:
1) The FIRST LINE must be exactly one line with no spaces or trailing characters.
   - A comma-separated subset of uppercase letters, e.g., `A,C,E`
   - Do NOT output `None` under any circumstances.
2) Select only options directly supported by clear, observable evidence. If uncertain, exclude them.
3) Consider ONLY the provided option set; ignore anything outside it.

Validation Rules:
- Allowed option set: {A, B, C, D, E}
- Allowed outputs (regex): ^(?:[A-E](?:,[A-E])*)$
- Use uppercase letters only; commas as separators; no spaces.
"""
).strip()

PERCEPTION_MC_PROMPT_HEADER = dedent(
    """\
Which of the following artifacts is present in the sample?
Select all that apply. If you select 'None of the options are correct,' do not select any other options.
"""
).strip()


PERCEPTION_TF_SYSTEM_HINT = dedent(
    """
You are a forensic media authenticity inspector.

Task:
- Given a single yes/no question asking whether a specific artifact appears in a specified region of a sample (image/audio/video), answer strictly "yes" or "no".

Output Constraints:
1) The FIRST LINE must be exactly one of: yes or no (lowercase).
   - No punctuation, spaces, or trailing characters.
   - Do NOT output `None` under any circumstances.

Decision Rules:
- Treat yes/no as equally likely.
- If the ROI is fully out-of-frame, fully occluded, or too low-res to perceive basic shape → answer "no".
- If any meaningful part of the ROI is visible, judge based on the visible portion; do not auto-"no" solely due to partial occlusion.
- Answer "yes" if either (a) a clear, distinctive cue of the named artifact is present in the ROI, or (b) two or more consistent subtle cues are present; otherwise answer "no".

Validation Rules:
- Allowed outputs (regex): ^(?:yes|no)$

Examples:
- "Does banding appear on the shoulder in the image?" → yes/no
- "Is there motion blur in the subject's face?" → yes/no
"""
).strip()

PERCEPTION_TF_PROMPT_GUIDE = dedent(
    """\
Each TF question is delivered as:
<{modality}>
[Yes/No question targeting a specific artifact and optional region.] (Answer yes if you think it is true, and answer no otherwise.)
"""
).strip()


__all__ = [
    "DETECTION_PROMPTS",
    "DETECTION_SYSTEM_HINT",
    "PERCEPTION_MC_PROMPT_HEADER",
    "PERCEPTION_MC_SYSTEM_HINT",
    "PERCEPTION_OE_PROMPTS",
    "PERCEPTION_OE_SYSTEM_HINT",
    "PERCEPTION_TF_PROMPT_GUIDE",
    "PERCEPTION_TF_SYSTEM_HINT",
]
