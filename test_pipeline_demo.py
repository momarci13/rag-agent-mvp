"""Manual OpenRouter hosted-inference connectivity smoke test.

This file is intentionally outside the automated test suite because it calls a
real configured gateway.
"""
from agents.llm import HostedLLM
from run import load_config, make_llm_config


def main() -> None:
    cfg = load_config()
    llm = HostedLLM(make_llm_config(cfg))
    if not llm.health():
        raise SystemExit("OpenRouter is unavailable or unauthorized")
    reply = llm.chat(
        [{"role": "user", "content": "Reply with the single word: pong"}],
        temperature=0.0,
    )
    print(reply)


if __name__ == "__main__":
    main()
