import json
import re
from openai import OpenAI


class Translator:
    def __init__(self, api_key: str, target_language: str = "French"):
        self.client = OpenAI(api_key=api_key)
        self.target_language = target_language

    def translate_blocks(self, blocks: list) -> list:
        if not blocks:
            return []

        payload = [{"id": b["id"], "text": b["text"]} for b in blocks]

        system_prompt = f"""Translate OCR-extracted English text into {self.target_language}.

Rules:
1. Return ONLY a JSON array — no markdown, no commentary
2. Each element must be: {{"id": <int>, "translated": "<string>"}}
3. Fix obvious OCR artefacts (merged words, character substitutions)
4. Preserve codes, numbers, proper nouns, and formatting exactly
5. Every id from the input must appear in the output"""

        response = self.client.responses.create(
            model="gpt-4.1-mini",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )

        return self._parse(response.output_text.strip(), blocks)

    def _parse(self, raw: str, original: list) -> list:
        # Strip markdown fences if the model wrapped the JSON
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw)

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    parsed = []
            else:
                parsed = []

        if not parsed:
            print("[Translator] Could not parse response — keeping original text.")
            return [{"id": b["id"], "translated": b["text"]} for b in original]

        id_map = {
            item["id"]: (item.get("translated") or item.get("text") or "")
            for item in parsed
        }

        return [
            {"id": b["id"], "translated": id_map.get(b["id"], b["text"])}
            for b in original
        ]
