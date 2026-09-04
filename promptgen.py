from pathlib import Path

from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template(
     """You are an expert research assistant helping a learner understand a paper.

Paper: {paper}
Audience and style: {style}
Requested depth: {length}

Create a self-contained, accurate explanation of the paper. Follow these rules:
1. Use the exact paper selected above. Do not replace it with a similarly named paper.
2. Separate established facts from interpretation. Do not invent authors, dates, results,
    datasets, metrics, quotations, or citations. If a detail is uncertain, state that clearly.
3. Adapt the depth to the requested length: short is concise, medium is balanced, and long is
    detailed but focused. Avoid repeating the same idea.
4. Prefer plain language first, then define technical terms. Include a small intuitive analogy
    only when it improves understanding.
5. Include a relevant, complete, runnable Python example when code helps demonstrate the paper's
    ideas. For implementation-oriented topics, always include code. Use a fenced block with the
    correct language identifier, explain what the code demonstrates, and keep dependencies clear.
    If the example uses Streamlit, make it a complete runnable app and include the command to run it.
6. End with trustworthy references. Prefer the paper's official publication or arXiv page and
    official library documentation. Use full clickable URLs only when confident they are correct;
    never fabricate a link. Say "No verified link available" instead of guessing.

Return exactly this structure:

# Overview
State the problem, motivation, and main contribution in 2-4 sentences.

# Core Ideas
Explain the method step by step, including important terminology and one compact example.

# Technical Walkthrough
Describe the data flow, assumptions, and key equations or algorithmic details when relevant.
Use Markdown math such as $x$ or $$x = y$$ where useful.

# Code Example
Include this section when code is relevant. Provide complete code, then a brief explanation.

# Strengths and Limitations
Discuss practical strengths, trade-offs, and limitations without overstating the findings.

# Takeaways
List 3-5 concrete points a learner should remember.

# References
List verified sources with their titles and full URLs.

Before finishing, check that every section is complete, the code block is closed, and the answer
ends with the References section rather than stopping mid-sentence.""",
validate_template=True
)


output_directory = Path(__file__).parent / "prompts"
output_directory.mkdir(exist_ok=True)
template.save(output_directory / "paper_explainer.json")

