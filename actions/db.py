import os
import shutil
import tempfile
from typing import Any, List
from typing import Optional, List, Dict
from pydantic import Field

from pydantic import BaseModel

from rasa.nlu.utils import write_json_to_file
from rasa.shared.utils.io import read_json_file

ORIGIN_DB_PATH = "db"
USER_PROFILES = "user_profiles.json"


class UserProfile(BaseModel):
    user_id: str
    name: Optional[str] = "Yajman"
    pain_points: List[str] = Field(default_factory=list)
    active_remedy: Optional[Dict] = None
    topic_change_count: int = 0
    last_barnum_turn: int = 0
    interaction_count: int = 0

def get_session_db_path(session_id: str) -> str:
    tempdir = tempfile.gettempdir()
    project_name = "ai_astro_omkar"
    return os.path.join(tempdir, project_name, session_id)

def prepare_db_file(session_id: str, db: str) -> str:
    session_db_path = get_session_db_path(session_id)
    os.makedirs(session_db_path, exist_ok=True)
    destination_file = os.path.join(session_db_path, db)
    if not os.path.exists(destination_file):
        origin_file = os.path.join(ORIGIN_DB_PATH, db)
        if os.path.exists(origin_file):
            shutil.copy(origin_file, destination_file)
        else:
            # Initialize empty list if file doesn't exist
            write_json_to_file(destination_file, [])
    return destination_file

def get_user_profile(session_id: str, user_id: str) -> UserProfile:
    profiles = read_json_file(prepare_db_file(session_id, USER_PROFILES))
    for p in profiles:
        if p['user_id'] == user_id:
            return UserProfile(**p)
    return UserProfile(user_id=user_id)

def update_user_profile(session_id: str, profile: UserProfile) -> None:
    db_file = prepare_db_file(session_id, USER_PROFILES)
    profiles = read_json_file(db_file)
    # Update existing or append new
    updated_profiles = [p for p in profiles if p['user_id'] != profile.user_id]
    updated_profiles.append(profile.dict())
    write_json_to_file(db_file, updated_profiles)