"""
Chat Engine — performs RAG: retrieves relevant chunks then asks the LLM.
Uses Google Gemini 2.5 Flash (free tier — no credit card required).
Set GEMINI_API_KEY in your .env file.
"""

import os
from typing import List, Dict, Tuple
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()


SYSTEM_PROMPT = """You are a personal knowledge assistant. Your job is to answer the user's questions using ONLY the context provided from their personal knowledge base.

Rules:
- Base your answers strictly on the provided context.
- If the context doesn't contain enough information to answer, say so honestly. Don't make things up.
- Cite the source filename(s) inline when you use information from them, like [source: filename.pdf].
- Synthesize across multiple sources when relevant — don't just repeat one chunk.
- Be concise but thorough. Use a friendly, clear tone.
- If asked something completely unrelated to the context, gently redirect."""


class ChatEngine:
    def __init__(self, knowledge_base, model: str = "gemini-2.5-flash"):
        self.kb = knowledge_base
        self.model = model
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Add it to a .env file in the project root.\n"
                "Get a free key at: https://aistudio.google.com/apikey"
            )
        self.client = genai.Client(api_key=api_key)

    def ask(self, question: str, history: List[Dict] = None, n_results: int = 5) -> Tuple[str, List[Dict]]:
        """
        Run the full RAG pipeline:
        1. Retrieve relevant chunks
        2. Build a prompt with context
        3. Call Gemini
        4. Return the answer + sources
        """
        # 1. Retrieve
        hits = self.kb.search(question, n_results=n_results)

        if not hits:
            return (
                "I couldn't find anything relevant in your knowledge base for that question. "
                "Try rephrasing or upload more documents.",
                [],
            )

        # 2. Build context block
        context_parts = []
        for i, hit in enumerate(hits, 1):
            context_parts.append(
                f"[Source {i}: {hit['filename']}]\n{hit['text']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        user_message = f"""Context from my knowledge base:

{context}

---

My question: {question}"""

        # 3. Build conversation history
        contents = []
        if history:
            for msg in history[-6:]:
                role = "user" if msg["role"] == "user" else "model"
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=msg["content"])]
                    )
                )
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_message)]
            )
        )

        # 4. Call Gemini
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=1500,
                    temperature=0.7,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            answer = response.text
        except Exception as e:
            return f"⚠️ Error calling the LLM: {e}", hits

        return answer, hits
