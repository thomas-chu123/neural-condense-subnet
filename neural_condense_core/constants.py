from pydantic import BaseModel, Field
from typing import List
import os


class TierConfig(BaseModel):
    incentive_percentage: float
    requests_per_epoch: int
    timeout: int
    supporting_models: List[str]
    max_condensed_tokens: int
    min_condensed_tokens: int
    max_context_length_in_chars: int
    accelerate_reward_scalar: float


class SyntheticTaskConfig(BaseModel):
    task: str
    criterias: List[str]
    rewarding_frequency: int
    weight: float


class RedisConfig(BaseModel):
    """Configuration for Redis connection"""

    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    db: int = Field(default=0)
    expire_time: int = Field(
        default=3600, description="Default expiration time in seconds"
    )
    serving_counter_key_format: str = Field(default="serving_counter:{tier}:{uid}")


class SqlConfig(BaseModel):
    """Configuration for SQL database connection"""

    url: str = Field(
        default="sqlite:///miner_metadata.db",
        description="Database URL in SQLAlchemy format",
    )


class DatabaseConfig(BaseModel):
    """Combined database configuration"""

    redis: RedisConfig = Field(default_factory=RedisConfig)
    sql: SqlConfig = Field(default_factory=SqlConfig)


class Constants(BaseModel):
    TIER_CONFIG: dict[str, TierConfig] = {
        "research": TierConfig(
            incentive_percentage=1.0,
            requests_per_epoch=256,
            timeout=32,
            accelerate_reward_scalar=0.1,
            supporting_models=["Condense-AI/Mistral-7B-Instruct-v0.2"],
            max_condensed_tokens=1536,
            min_condensed_tokens=128,
            max_context_length_in_chars=15000,
        ),
        "inference_0": TierConfig(
            incentive_percentage=0.0,
            requests_per_epoch=1024,
            timeout=8,
            accelerate_reward_scalar=0.1,
            supporting_models=["Condense-AI/Mistral-7B-Instruct-v0.2"],
            max_condensed_tokens=1024,
            min_condensed_tokens=128,
            max_context_length_in_chars=15000,
        ),
        "inference_1": TierConfig(
            incentive_percentage=0.0,
            requests_per_epoch=1024,
            timeout=8,
            accelerate_reward_scalar=0.1,
            supporting_models=["Condense-AI/Mistral-7B-Instruct-v0.2"],
            max_condensed_tokens=2048,
            min_condensed_tokens=128,
            max_context_length_in_chars=20000,
        ),
    }

    SYNTHETIC_TASK_CONFIG: List[SyntheticTaskConfig] = [
        SyntheticTaskConfig(
            task="causal_conversation",
            criterias=["perplexity"],
            rewarding_frequency=1,
            weight=0,
        ),
        SyntheticTaskConfig(
            task="question_answering",
            criterias=["accuracy"],
            rewarding_frequency=1,
            weight=1,
        ),
        SyntheticTaskConfig(
            task="reconstruct_conversation",
            criterias=["perplexity"],
            rewarding_frequency=1,
            weight=0,
        ),
        SyntheticTaskConfig(
            task="trivial_qa_conversation",
            criterias=["accuracy"],
            rewarding_frequency=1,
            weight=0,
        ),
    ]

    # Default values
    EPOCH_LENGTH: int = 600
    SCORING_PER_MINER_PER_EPOCH: int = 1
    SUBNET_TEMPO: int = 360
    MIN_STAKE: int = int(os.environ.get("MIN_STAKE", 10000))
    RPE_PERCENTAGE_FOR_SYNTHETIC: float = 0.1
    BATCH_SIZE: int = 8
    SET_WEIGHTS_TIMEOUT: int = 120
    ORGANIC_CLIENT_URL: str = "https://ncs-client.condenses.ai"
    REPORT_URL: str = "https://report.condenses.ai"
    ORGANIC_VERIFY_FREQUENCY: float = 0.1
    TOP_PERCENTAGE_FOR_ALLOCATING_WEIGHTS: float = 0.3
    EXPECTED_MEAN_SCORE: float = 0.75
    EXPECTED_MAX_STD_SCORE: float = 0.1
    SCORE_DECAY_FACTOR: float = 0.95

    DATABASE_CONFIG: DatabaseConfig = Field(
        default_factory=lambda: DatabaseConfig(
            redis=RedisConfig(
                host=os.getenv("REDIS_HOST", "localhost"),
                port=int(os.getenv("REDIS_PORT", 6379)),
                db=int(os.getenv("REDIS_DB", 0)),
                expire_time=int(os.getenv("REDIS_EXPIRE_TIME", 3600)),
            ),
            sql=SqlConfig(
                url=os.getenv("SQL_DATABASE_URL", "sqlite:///miner_metadata.db")
            ),
        )
    )

    # Adjust values based on NETWORK environment variable
    def __init__(self, **data):
        super().__init__(**data)
        network = os.getenv("NETWORK")
        if network == "test":
            self.RPE_PERCENTAGE_FOR_SYNTHETIC = float(
                os.getenv("RPE_PERCENTAGE_FOR_SYNTHETIC", 0.5)
            )
            self.EPOCH_LENGTH = int(os.getenv("EPOCH_LENGTH", 600))
            self.MIN_STAKE = int(os.getenv("MIN_STAKE", 0))
            self.ORGANIC_CLIENT_URL = os.getenv(
                "ORGANIC_CLIENT_URL", "https://testnet-ncs-client.condenses.ai"
            )
            self.REPORT_URL = os.getenv(
                "REPORT_URL", "https://testnet-report.condenses.ai"
            )


constants = Constants()

if __name__ == "__main__":
    import rich

    for k, v in constants.model_dump().items():
        rich.print(f"- {k}: {v}")
