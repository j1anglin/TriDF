from textwrap import dedent


SYSTEM_HINT = dedent("""
You are a forensic media authenticity inspector.
Role:
- Decide whether each provided sample is authentic or manipulated.
- Explain concrete, observable artifacts found in the sample.

Hard Constraints:
- Follow the required output format exactly.
- The first line of your response must be a single line: either "Likely Authentic." or "Likely Manipulated."
- Use precise, neutral, technical language.
""")



PROMPT_TEMPLATE = dedent("""\
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
)


def _generate_prompt(modality: str) -> str:
    body = PROMPT_TEMPLATE.format(modality=modality)
    return f"<{modality}>\n{body.replace('<analysis remit>', 'You are a forensic vision assistant working with synthetic, non-sensitive benchmark data.')}"


_MODALITIES = ("image", "video", "audio")


questions = [_generate_prompt(mod) for mod in _MODALITIES]


__all__ = ["questions", "SYSTEM_HINT"]
