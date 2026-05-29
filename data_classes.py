import datetime
from typing import Optional
from dataclasses import dataclass

@dataclass
class RawComment:
  comment_id: str
  video_id: str
  parent_id: Optional[str]
  raw_text: str
  language: Optional[str]
  likes: int
  created_at: datetime.datetime

  # function to allow insertion into mongo
  def to_dict(self) -> dict:
    return {
      "_id": self.comment_id,
      "video_id": self.video_id,
      "parent_id": self.parent_id,
      "raw_text": self.raw_text,
      "language": self.language,
      "likes": self.likes,
      "created_at": self.created_at
    }