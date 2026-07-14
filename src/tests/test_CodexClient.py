import subprocess
from pathlib import Path
from unittest.mock import patch

from src.llm.llm import CodexClient, LLMClient


def test_codex_client_uses_final_message_file():
    def fake_run(command, **kwargs):
        output_path = command[command.index("--output-last-message") + 1]
        Path(output_path).write_text("final answer\n", encoding="utf-8")
        assert "<SYSTEM>\nbe concise\n</SYSTEM>" in kwargs["input"]
        assert "<USER>\nhello\n</USER>" in kwargs["input"]
        return subprocess.CompletedProcess(command, 0, "progress noise", "diagnostics")

    previous_log_setting = LLMClient.query_logger.enable_log
    LLMClient.query_logger.enable_log = False
    try:
        with patch("src.llm.llm.shutil.which", return_value="/usr/bin/codex"), patch(
            "src.llm.llm.subprocess.run", side_effect=fake_run
        ) as run:
            client = CodexClient(model="gpt-test", profile="ci", retry_times=1)
            response = client.query_with_messages(
                [
                    {"role": "system", "content": "be concise"},
                    {"role": "user", "content": "hello"},
                ]
            )

        assert response == "final answer"
        command = run.call_args.args[0]
        assert command[:2] == ["codex", "exec"]
        assert ["--sandbox", "read-only"] == command[
            command.index("--sandbox") : command.index("--sandbox") + 2
        ]
        assert ["--model", "gpt-test"] == command[
            command.index("--model") : command.index("--model") + 2
        ]
        assert ["--profile", "ci"] == command[
            command.index("--profile") : command.index("--profile") + 2
        ]
        assert command[-1] == "-"
    finally:
        LLMClient.query_logger.enable_log = previous_log_setting


def test_codex_client_returns_none_on_cli_failure():
    previous_log_setting = LLMClient.query_logger.enable_log
    LLMClient.query_logger.enable_log = False
    try:
        with patch("src.llm.llm.shutil.which", return_value="/usr/bin/codex"), patch(
            "src.llm.llm.subprocess.run",
            return_value=subprocess.CompletedProcess([], 1, "", "not logged in"),
        ) as run:
            client = CodexClient(retry_times=2)
            assert client.query_once("hello") is None
            assert run.call_count == 2
    finally:
        LLMClient.query_logger.enable_log = previous_log_setting
