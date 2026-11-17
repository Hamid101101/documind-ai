
import pytest

from llm.config import LLMConfig


def test_default_configuration():
    config = LLMConfig()

    assert config.provider == "ollama"
    assert config.base_url == "http://localhost:11434"
    assert config.model == "qwen2.5:3b"
    assert config.temperature == 0.1
    assert config.timeout == 120


def test_custom_configuration():
    config = LLMConfig(
        provider="ollama",
        base_url="http://localhost:11435",
        model="qwen2.5:7b",
        temperature=0.5,
        timeout=180,
    )

    assert config.base_url == "http://localhost:11435"
    assert config.model == "qwen2.5:7b"
    assert config.temperature == 0.5
    assert config.timeout == 180


def test_configuration_is_immutable():
    config = LLMConfig()

    with pytest.raises(AttributeError):
        config.model = "another-model"


@pytest.mark.parametrize("provider", ["", " ", "\t"])
def test_rejects_empty_provider(provider):
    with pytest.raises(ValueError, match="provider cannot be empty"):
        LLMConfig(provider=provider)


@pytest.mark.parametrize("base_url", ["", " ", "\t"])
def test_rejects_empty_base_url(base_url):
    with pytest.raises(ValueError, match="base URL cannot be empty"):
        LLMConfig(base_url=base_url)


@pytest.mark.parametrize("model", ["", " ", "\t"])
def test_rejects_empty_model(model):
    with pytest.raises(ValueError, match="model cannot be empty"):
        LLMConfig(model=model)


@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature must be between"):
        LLMConfig(temperature=temperature)


@pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0, 2.0])
def test_accepts_valid_temperature(temperature):
    config = LLMConfig(temperature=temperature)

    assert config.temperature == temperature


@pytest.mark.parametrize("timeout", [0, -1])
def test_rejects_invalid_timeout(timeout):
    with pytest.raises(ValueError, match="timeout must be greater than zero"):
        LLMConfig(timeout=timeout)


def test_from_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:11435/")
    monkeypatch.setenv("LLM_MODEL", "qwen2.5:7b")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.4")
    monkeypatch.setenv("LLM_TIMEOUT", "180")

    config = LLMConfig.from_env()

    assert config.provider == "ollama"
    assert config.base_url == "http://localhost:11435"
    assert config.model == "qwen2.5:7b"
    assert config.temperature == 0.4
    assert config.timeout == 180


def test_from_env_uses_defaults(monkeypatch):
    for variable in (
        "LLM_PROVIDER",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "LLM_TEMPERATURE",
        "LLM_TIMEOUT",
    ):
        monkeypatch.delenv(variable, raising=False)

    config = LLMConfig.from_env()

    assert config == LLMConfig()
