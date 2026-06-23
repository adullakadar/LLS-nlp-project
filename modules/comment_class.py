import datetime
from typing import Optional
from dataclasses import dataclass
import math

@dataclass
class RawComment:
  comment_id: str
  video_id: str
  parent_id: Optional[str]
  is_reply: bool
  author: str
  text: str
  like_count: int
  public: bool

  # function to allow insertion into mongo
  def to_dict(self) -> dict:
    return {
      "_id": self.comment_id,
      "video_id": self.video_id,
      "parent_id": self.parent_id,
      "is_reply": self.is_reply,
      "author": self.author,
      "text": self.text,
      "like_count": self.like_count,
      "public": self.public
    }
  
  @classmethod
  def df_to_comment(comment, row: dict) -> "RawComment":
    parent = row['parent_id']
    if parent is None or (isinstance(parent, float) and math.isnan(parent)):
      parent = None
    
    return comment(
      comment_id=str(row["comment_id"]),
      video_id=str(row["video_id"]),
      parent_id=parent,
      is_reply=bool(row["is_reply"]),
      author=str(row["author"]),
      text=str(row["text"]),
      like_count=int(row["like_count"]),
      public=bool(row["public"]),
    )
  