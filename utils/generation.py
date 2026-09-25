"""
Prompt assembly, <think> stream parsing and source formatting.
Streamlit-free so the app and the eval harness build identical prompts.
"""
import os

# Upper bound for Ollama's context window. The actual num_ctx is sized to
# the prompt so small prompts don't pay for a huge KV cache.
NUM_CTX_MIN = 4096
NUM_CTX_MAX = int(os.getenv("NUM_CTX_MAX", "16384"))
ANSWER_TOKEN_BUDGET = 1536


def source_label(metadata):
    """'report.pdf · p.12' style label from chunk metadata."""
    name = metadata.get("source") or "document"
    page = metadata.get("page")
    return f"{name} · p.{page}" if page else name


def source_record(doc):
    """Serializable view of a retrieved chunk, as shown in the UI."""
    return {
        "text": doc.metadata.get("original_text", doc.page_content),
        "label": source_label(doc.metadata),
        "source": doc.metadata.get("source"),
        "page": doc.metadata.get("page"),
    }


def format_context(docs):
    return "\n\n".join(
        f"[Source {i + 1}] ({source_label(d.metadata)}):\n{d.page_content}"
        for i, d in enumerate(docs)
    )


def build_prompt(question, context="", chat_history="", show_thinking=False, crag_low=False):
    ctx_block = f"\nContext:\n{context}\n" if context else ""
    ctx_instruction = (
        "\n- Answer based on the provided context. Cite [Source N] when referencing specific info."
        if context else ""
    )
    think_instruction = (
        "Before answering, reason through the problem step by step inside <think>...</think> tags. "
        "Then give your final answer outside those tags.\n\n"
        if show_thinking else ""
    )
    history_block = f"Chat History:\n{chat_history}\n" if chat_history else ""
    prompt = (
        f"{think_instruction}"
        f"You are a helpful, thorough AI assistant.\n\n"
        f"{history_block}"
        f"{ctx_block}"
        f"Question: {question}\n\n"
        f"Instructions:\n"
        f"- Be concise and well-structured{ctx_instruction}\n"
        f"- If you don't know, say so clearly\n"
    )
    if crag_low:
        prompt += "- The retrieved context may not be relevant; if so, state that the documents don't cover this.\n"
    if show_thinking:
        prompt += "- Put ALL reasoning inside <think>...</think>; the answer goes after\n"
    return prompt


def estimate_num_ctx(prompt):
    """Size the context window to fit the prompt plus room for the answer.
    ~3.5 characters per token is a conservative estimate for English text."""
    needed = int(len(prompt) / 3.5) + ANSWER_TOKEN_BUDGET
    size = NUM_CTX_MIN
    while size < needed and size < NUM_CTX_MAX:
        size *= 2
    return min(size, NUM_CTX_MAX)


class ThinkStreamParser:
    """Split a token stream into <think> reasoning and the visible answer.

    Tags may arrive split across tokens, so a short tail of the buffer is held
    back until it can no longer be the start of a tag.
    """
    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self.thinking = ""
        self.answer = ""
        self.buffer = ""
        self.in_think = False
        self.think_done = False

    def feed(self, token):
        self.buffer += token
        changed = True
        while changed:
            changed = False
            if not self.in_think and not self.think_done:
                idx = self.buffer.find(self.OPEN)
                if idx != -1:
                    self.answer += self.buffer[:idx].strip()
                    self.buffer = self.buffer[idx + len(self.OPEN):]
                    self.in_think = True
                    changed = True
                else:
                    safe = max(0, len(self.buffer) - len(self.OPEN))
                    self.answer += self.buffer[:safe]
                    self.buffer = self.buffer[safe:]
            elif self.in_think:
                idx = self.buffer.find(self.CLOSE)
                if idx != -1:
                    self.thinking += self.buffer[:idx]
                    self.buffer = self.buffer[idx + len(self.CLOSE):]
                    self.in_think = False
                    self.think_done = True
                    changed = True
                else:
                    safe = max(0, len(self.buffer) - len(self.CLOSE))
                    self.thinking += self.buffer[:safe]
                    self.buffer = self.buffer[safe:]
            else:
                self.answer += self.buffer
                self.buffer = ""

    @property
    def live_thinking(self):
        return self.thinking + (self.buffer if self.in_think else "")

    @property
    def live_answer(self):
        return self.answer + (self.buffer if not self.in_think else "")

    def finish(self):
        if self.in_think:
            self.thinking += self.buffer
        else:
            self.answer += self.buffer
        self.buffer = ""
        return self.thinking.strip(), self.answer.strip()
