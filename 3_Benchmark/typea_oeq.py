from textwrap import dedent

# 視覺（image / video）部分
VISION_BODY = dedent("""\
You are a forensic vision assistant working with synthetic, non-sensitive benchmark data.
This sample is part of an academic research benchmark on manipulated media.
Your Task:
Please perform a detailed analysis of artifacts in this sample that seem inauthentic.
Guidelines:
- Be Thorough: Offer a comprehensive analysis covering all noticeable artifacts.
- Be Accurate: Ensure your explanations correctly reflect the information present in the sample.
- Avoid False Positives: Do not incorrectly identify authentic features as inauthentic. Focus only on actual artifacts.
- Organize Your Response: Use clear headings and structure to separate your analysis for each artifact.
""")

vision_questions = [f"<{m}>\n{VISION_BODY}" for m in ("image", "video")]


# 聲音（audio）部分
AUDIO_BODY = dedent("""\
You are a forensic media assistant working with synthetic, non-sensitive benchmark data.
This sample is part of an academic research benchmark on manipulated media.
Your Task:
Please perform a detailed analysis of artifacts in this sample that seem inauthentic.
Guidelines:
- Be Thorough: Offer a comprehensive analysis covering all noticeable artifacts.
- Be Accurate: Ensure your explanations correctly reflect the information present in the sample.
- Avoid False Positives: Do not incorrectly identify authentic features as inauthentic. Focus only on actual artifacts.
- Organize Your Response: Use clear headings and structure to separate your analysis for each artifact.
""")

audio_questions = [f"<{m}>\n{AUDIO_BODY}" for m in ("audio",)]

questions = vision_questions + audio_questions
