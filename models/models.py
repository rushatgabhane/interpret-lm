from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class Example(BaseModel):
    sentence_good: str
    sentence_bad: str
    UID: str
    
    one_prefix_method: bool = Field(False, alias="one_prefix_method")
    two_prefix_method: bool = Field(False, alias="two_prefix_method")
    
    one_prefix_prefix: str | None = None
    one_prefix_word_good: str | None = None
    one_prefix_word_bad: str | None = None
    
    two_prefix_prefix_good: str | None = None
    two_prefix_prefix_bad: str | None = None
    two_prefix_word: str | None = None

    @property
    def is_one_prefix(self):
        return self.one_prefix_method

    @property
    def is_two_prefix(self):
        return self.two_prefix_method
