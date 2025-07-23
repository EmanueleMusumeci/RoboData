import yaml
import os
from pathlib import Path

class LLMSettings:
    def __init__(self, provider="openai", api_key=None, model="openai-pro", temperature=0.7, max_tokens=4096):
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

class WikidataSettings:
    def __init__(self, timeout=30, max_results=100, default_language="en"):
        self.timeout = timeout
        self.max_results = max_results
        self.default_language = default_language

class Neo4jSettings:
    def __init__(self, uri="bolt://localhost:7687", username="neo4j", password="password", 
                 database="neo4j", connection_timeout=30, max_connection_lifetime=3600):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.connection_timeout = connection_timeout
        self.max_connection_lifetime = max_connection_lifetime

class ToolboxSettings:
    def __init__(self, auto_register_wikidata_tools=True, max_tool_execution_time=60):
        self.auto_register_wikidata_tools = auto_register_wikidata_tools
        self.max_tool_execution_time = max_tool_execution_time

class InteractiveSettings:
    def __init__(self, show_tool_calls=True, show_intermediate_steps=True, max_history_length=100):
        self.show_tool_calls = show_tool_calls
        self.show_intermediate_steps = show_intermediate_steps
        self.max_history_length = max_history_length

class Settings:
    def __init__(self, llm=None, wikidata=None, neo4j=None, toolbox=None, interactive=None):
        self.llm = llm or LLMSettings()
        self.wikidata = wikidata or WikidataSettings()
        self.neo4j = neo4j or Neo4jSettings()
        self.toolbox = toolbox or ToolboxSettings()
        self.interactive = interactive or InteractiveSettings()

    def to_dict(self):
        return {
            "llm": self.llm.__dict__,
            "wikidata": self.wikidata.__dict__,
            "neo4j": self.neo4j.__dict__,
            "toolbox": self.toolbox.__dict__,
            "interactive": self.interactive.__dict__,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            llm=LLMSettings(**data.get("llm", {})),
            wikidata=WikidataSettings(**data.get("wikidata", {})),
            neo4j=Neo4jSettings(**data.get("neo4j", {})),
            toolbox=ToolboxSettings(**data.get("toolbox", {})),
            interactive=InteractiveSettings(**data.get("interactive", {})),
        )

class SettingsManager:
    """Manages application settings from config file and environment variables."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.settings = self._load_settings()

    def _load_settings(self) -> Settings:
        settings_dict = {}
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                if config_data:
                    settings_dict.update(config_data)
        else:
            raise FileNotFoundError(f"Configuration file {self.config_path} not found.")
        self._load_env_overrides(settings_dict)
        return Settings.from_dict(settings_dict)

    def _load_env_overrides(self, settings_dict):
        if 'llm' not in settings_dict:
            settings_dict['llm'] = {}

        # LLM environment overrides
        if os.getenv('OPENAI_API_KEY'):
            settings_dict['llm']['api_key'] = os.getenv('OPENAI_API_KEY')
        if os.getenv('LLM_PROVIDER'):
            settings_dict['llm']['provider'] = os.getenv('LLM_PROVIDER')
        if os.getenv('LLM_MODEL'):
            settings_dict['llm']['model'] = os.getenv('LLM_MODEL')

        # Neo4j environment overrides
        if 'neo4j' not in settings_dict:
            settings_dict['neo4j'] = {}
        if os.getenv('NEO4J_URI'):
            settings_dict['neo4j']['uri'] = os.getenv('NEO4J_URI')
        if os.getenv('NEO4J_USERNAME'):
            settings_dict['neo4j']['username'] = os.getenv('NEO4J_USERNAME')
        if os.getenv('NEO4J_PASSWORD'):
            settings_dict['neo4j']['password'] = os.getenv('NEO4J_PASSWORD')
        if os.getenv('NEO4J_DATABASE'):
            settings_dict['neo4j']['database'] = os.getenv('NEO4J_DATABASE')

    def save_settings(self):
        with open(self.config_path, 'w') as f:
            yaml.dump(self.settings.to_dict(), f, default_flow_style=False)

    def get_settings(self) -> Settings:
        return self.settings

    def update_settings(self, **kwargs):
        settings_dict = self.settings.to_dict()
        for key, value in kwargs.items():
            if '.' in key:
                keys = key.split('.')
                current = settings_dict
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
            else:
                settings_dict[key] = value
        self.settings = Settings.from_dict(settings_dict)

# Global settings manager instance
settings_manager = SettingsManager(config_path="config.yaml")
