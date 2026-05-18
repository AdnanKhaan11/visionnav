"""
VisionNav typed settings.
All configuration lives here — no hardcoded values anywhere else.

Priority (highest → lowest):
  1. Environment variables   VISIONNAV_MODEL__BACKEND=vllm
  2. .env file
  3. Defaults defined below
"""
from __future__ import annotations
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class VLLMSettings(BaseSettings):
    base_url: str = "http://localhost:8001"
    model_name: str = "visionnav-3b"
    timeout_seconds: int = 30


class ModelSettings(BaseSettings):
    backend: str = "local"
    name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    dtype: str = "bfloat16"
    device_map: str = "auto"
    vllm: VLLMSettings = VLLMSettings()


class APISettings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    valid_keys: list[str] = Field(default_factory=list)


class AgentSettings(BaseSettings):
    max_steps: int = 50
    screenshot_dir: str = "/tmp/visionnav/screenshots"
    change_threshold: float = 0.01


class DBSettings(BaseSettings):
    url: str = "sqlite+aiosqlite:///./visionnav.db"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VISIONNAV_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
    )
    env: str = "development"
    model: ModelSettings = ModelSettings()
    api: APISettings = APISettings()
    agent: AgentSettings = AgentSettings()
    db: DBSettings = DBSettings()


def get_settings() -> Settings:
    return Settings()
