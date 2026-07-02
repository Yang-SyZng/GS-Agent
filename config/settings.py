from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
import os

class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    LLM_API_KEY: str = model_config.get("LLM_API_KEY") or None
    LLM_BASE_URL: str = model_config.get("LLM_BASE_URL") or None

    # model config
    LLM_MODEL_ID: str = model_config.get("LLM_MODEL_ID") or None

    EMBEDDING_MODEL_ID: str = ""


    # DATA save config
    _mainly_save_dir: str = "./database"

    # ChromaDB config
    chroma_save_dir: str = os.path.join(_mainly_save_dir, "Chroma")
    chroma_collection_name: str = "arxiv_papers"
    
    # pdf save config
    pdf_save_dir: str = os.path.join(_mainly_save_dir, "Papers")


    # rag config
    chunk_size: int = 3000
    chunk_overlap: int = 500


    # api config
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_reload: bool = True


    # log config
    log_level: str = "INFO"


    # arxiv api config
    rate_limit: float = 3.0
    max_retries: int = 3


    @property
    def chroma(self) -> Path:
        return Path(self.chroma_save_dir)


    @property
    def pdf(self) -> Path:
        return Path(self.pdf_save_dir)


    def __init__(self):
        super().__init__()
        self.chroma.mkdir(parents=True, exist_ok=True)
        self.pdf.mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self._mainly_save_dir, "Cache")).mkdir(parents=True, exist_ok=True)

setting = Settings()

if __name__ == "__main__":
    settings = Settings()

    