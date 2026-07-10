from __future__ import annotations

import logging

from prompts import AnalyzerPrompt, AnalyzerDescription

from .Baser import BaseFunctionAgent

logger = logging.getLogger(__name__)


class Analyzer(BaseFunctionAgent):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("name", "Planner")
        kwargs.setdefault("description", AnalyzerDescription)
        kwargs.setdefault("system_prompt", AnalyzerPrompt)
        super().__init__(*args, **kwargs)
    
    
