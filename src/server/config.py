"""Application settings for the Word-to-LaTeX API."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "wordtolatex-server"
    api_prefix: str = "/v1"
    environment: str = "local"
    log_level: str = "INFO"

    default_use_local_llm: bool = True
    default_use_gemini: bool = False
    default_llm_model: str | None = None

    temp_dir_prefix: str = "wordtolatex-"
    rate_limit_enabled: bool = True
    rate_limit_cooldown_seconds: int = 1800
    rate_limit_trust_forwarded: bool = True
    allow_user_options: bool = False
    expose_metadata: bool = False
    auto_reference_pdf: bool = True
    reference_pdf_command: str = "soffice"
    reference_pdf_args: str = "--headless --convert-to pdf --outdir {outdir} {input}"
    reference_pdf_timeout_seconds: int = 120

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
