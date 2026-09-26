from abc import ABC, abstractmethod
from typing import Any, Dict, List

class BaseStoryCompiler(ABC):
    """
    Abstract Interface for Story / Prompt Compiler component.
    Parses scripts, storyboards, or multi-shot scene prompts into structured scene graphs and shot lists.
    """

    @abstractmethod
    def compile_script(self, script_text: str, **kwargs: Any) -> List[Dict[str, Any]]:
        """
        Compiles script into structured shot sequence specifications.
        """
        pass
