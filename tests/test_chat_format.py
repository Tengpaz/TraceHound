from traceguard.chat_format import apply_chat_template, messages_to_prompt


MESSAGES = [
    {"role": "system", "content": "system rules"},
    {"role": "user", "content": "judge this"},
]


class TemplateTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        suffix = "<assistant>" if add_generation_prompt else ""
        return "|".join(f"{item['role']}={item['content']}" for item in messages) + suffix


class BrokenTemplateTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        raise RuntimeError("bad template")


def test_apply_chat_template_prefers_tokenizer_template():
    rendered = apply_chat_template(TemplateTokenizer(), MESSAGES)
    assert rendered == "system=system rules|user=judge this<assistant>"


def test_apply_chat_template_falls_back_to_plain_prompt():
    rendered = apply_chat_template(BrokenTemplateTokenizer(), MESSAGES)
    assert "SYSTEM:\nsystem rules" in rendered
    assert "USER:\njudge this" in rendered
    assert rendered.endswith("ASSISTANT:\n")


def test_messages_to_prompt_is_model_agnostic():
    rendered = messages_to_prompt(MESSAGES, add_generation_prompt=False)
    assert "SYSTEM:" in rendered
    assert "USER:" in rendered
    assert "ASSISTANT:" not in rendered
