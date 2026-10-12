import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel, Field

load_dotenv()

token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
if not token:
    raise RuntimeError("Set HUGGINGFACEHUB_API_TOKEN or HF_TOKEN in .env")

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    provider="featherless-ai",
    huggingfacehub_api_token=token,
    max_new_tokens=256,
    temperature=0.2,
    repetition_penalty=1.05,
)

model = ChatHuggingFace(llm=llm)

class Review(BaseModel):
    key_themes: Optional[list[str]] = Field(default=None, description="The main topics or themes discussed in the review")
    summary: Optional[str] = Field(default=None, description="A concise summary of the main issues")
    sentiment: Optional[str] = Field(default=None, description="The overall sentiment of the review")
    pros: Optional[list[str]] = Field(default = None, description="A list of positive aspects mentioned in the review")
    cons: Optional[list[str]] = Field(default = None, description="A list of negative aspects mentioned in the review")
    name: Optional[str] = Field(default = None, description="The name of the reviewer")

structured_model = model.with_structured_output(Review, method="json_mode")

review_text = """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don’t use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:

Insanely powerful processor (great for gaming and productivity)

Stunning 200MP camera with incredible zoom capabilities

Long battery life with fast charging

S-Pen support is unique and useful

Cons:

Bulky and heavy—not great for one-handed use

Bloatware still exists in One UI

Expensive compared to competitors"""

prompt = f"""Extract the review details as a valid JSON object with these fields:
key_themes, summary, sentiment, pros, cons, and name.

Review:
{review_text}"""

result = structured_model.invoke(prompt)

print(result)