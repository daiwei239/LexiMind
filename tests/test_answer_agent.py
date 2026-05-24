from app.generation.answer_agent import DeepSeekAnswerAgent


class StubDeepSeekClient:
    def __init__(self):
        self.last_messages = None

    def generate(self, messages):
        self.last_messages = messages
        return "这是基于法条证据生成的回答。"

    def generate_stream(self, messages):
        self.last_messages = messages
        yield "这是"
        yield "流式回答"


def test_deepseek_answer_agent_builds_structured_messages_and_uses_top_two_chunks():
    client = StubDeepSeekClient()
    agent = DeepSeekAnswerAgent(client=client, max_evidence=2)
    evidence = [
        {"chunk_id": "1", "parent_article_id": "p1", "title": "法条一", "content": "内容一"},
        {"chunk_id": "2", "parent_article_id": "p2", "title": "法条二", "content": "内容二"},
        {"chunk_id": "3", "parent_article_id": "p3", "title": "法条三", "content": "内容三"},
    ]

    result = agent.generate("合同解除条件是什么？", evidence)

    assert result["answer"] == "这是基于法条证据生成的回答。"
    assert [item["chunk_id"] for item in result["sources"]] == ["1", "2"]
    assert len(result["prompt_messages"]) == 3
    assert result["prompt_messages"][0]["role"] == "system"
    assert "法条一" in result["prompt_messages"][1]["content"]
    assert "法条二" in result["prompt_messages"][1]["content"]
    assert "法条三" not in result["prompt_messages"][1]["content"]
    assert result["prompt_messages"][2]["content"] == "合同解除条件是什么？"
    assert client.last_messages == result["prompt_messages"]


def test_deepseek_answer_agent_includes_chat_history_in_prompt():
    client = StubDeepSeekClient()
    agent = DeepSeekAnswerAgent(client=client, max_evidence=1)
    evidence = [{"chunk_id": "1", "title": "法条一", "content": "内容一"}]
    history = [
        {"role": "user", "content": "什么是合同？"},
        {"role": "assistant", "content": "合同是民事主体之间的协议。"},
    ]

    result = agent.generate("那解除条件呢？", evidence, chat_history=history)

    assert result["answer"] == "这是基于法条证据生成的回答。"
    assert result["prompt_messages"][2]["content"] == "什么是合同？"
    assert result["prompt_messages"][3]["content"] == "合同是民事主体之间的协议。"
    assert result["prompt_messages"][4]["content"] == "那解除条件呢？"


def test_deepseek_answer_agent_generate_stream():
    client = StubDeepSeekClient()
    agent = DeepSeekAnswerAgent(client=client, max_evidence=1)
    evidence = [{"chunk_id": "1", "title": "法条一", "content": "内容一"}]

    chunks = list(agent.generate_stream("问题", evidence))

    assert chunks == ["这是", "流式回答"]
