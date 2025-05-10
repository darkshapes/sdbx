from typing import Dict, List, Optional, Any

from pydantic import BaseModel

class TaskUpdate(BaseModel):
    id: str  # task ID
    results: Optional[Dict[str, List[Any]]] = None
    completed: Optional[bool] = False
    error: Optional[str] = None

    def dict(self, *args, **kwargs):
        # exclude_none and exclude_defaults are used to remove empty fields from the dictionary
        kwargs.setdefault('exclude_none', True)
        kwargs.setdefault('exclude_defaults', True)
        return super().dict(*args, **kwargs)