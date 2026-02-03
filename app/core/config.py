from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import ClassVar, Literal, Set
from urllib.parse import quote_plus
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

class Settings(BaseSettings):
    APP_ENV: Literal["prod", "dev"] = "dev"

    DEV_MODEL_VOCAB_PATH: str
    DEV_MODEL_CHECKPOINT_PATH: str
    PROD_MODEL_VOCAB_PATH: str
    PROD_MODEL_CHECKPOINT_PATH: str

    DEV_HOST: str
    DEV_PORT: int
    DEV_USER: str
    DEV_PASSWORD: str

    PROD_HOST: str
    PROD_PORT: int
    PROD_USER: str
    PROD_PASSWORD: str

    HF_MODEL: str
    HF_TOKEN: str | None = None
    LLM_URL: str
    LLM_SYSTEM_PROMPT: str | None = None
    LLM_USER_SESSION_ID: str | None = None
    LLM_USER_ID: str | None = None
    LLM_SYSTEM_CORRECT_PROMPT: str | None = None
    MISTRAL_URL: str

    SEMANTIC_KEYWORD_MODEL: str
    EMBED_MODEL: str
    DATA_PATH: str

    MIN_SAMPLES_PER_CLASS: ClassVar[int] = 5

    STOPWORDS: ClassVar[Set[str]] = { "و", "در", "به", "از", "با", "برای", "که", "را", "این", "آن", "آنها", "من", "تو", "او", "ما", "شما", "آنچه", "چگونه", "چه", "هم", "نیز", "بود", "بودن", "است", "هست", "می", "تا", "هر", "هرگز", "همه", "همین", "ولی", "اگر", "یا", "چون", "اما", "چرا", "چند", "چیزی", "کسی", "کجا", "چگونه", "چطور", "هنگام", "زمان", "هنوز", "بعد", "قبل", "پیش", "همزمان", "بعدها", "هنگامی", "چیز", "هیچ", "همان", "چیزهایی", "یکی", "دومی", "سومی", "خود", "خودش", "خودم", "خودت", "خودمان", "اینجا", "آنجا", "بالا", "پایین", "زیاد", "کم", "خوب", "بد", "چندین", "اکثر", "کمتر", "بیشتر", "نیست", "ندارد", "داشت", "داشتن", "می‌شود", "می‌کند", "می‌توان", "شد", "شدن", "شده", "وارد", "خارج", "همچنین", "بهتر", "بدتر", "سلام", "تیکت" }
    
    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        extra="ignore",
    )

    @property
    def MODEL_VOCAB_PATH(self) -> str:
        return self.PROD_MODEL_VOCAB_PATH if self.APP_ENV == "prod" else self.DEV_MODEL_VOCAB_PATH

    @property
    def MODEL_CHECKPOINT_PATH(self) -> str:
        return self.PROD_MODEL_CHECKPOINT_PATH if self.APP_ENV == "prod" else self.DEV_MODEL_CHECKPOINT_PATH

    @property
    def DATABASE_URL(self) -> str:
        if self.APP_ENV == "prod":
            host, port, user, pw = self.PROD_HOST, self.PROD_PORT, self.PROD_USER, self.PROD_PASSWORD
        else:
            host, port, user, pw = self.DEV_HOST, self.DEV_PORT, self.DEV_USER, self.DEV_PASSWORD

        return f"mysql+pymysql://{user}:{quote_plus(pw)}@{host}:{port}/smart"


settings = Settings()