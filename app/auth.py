import json
import os
from datetime import datetime, timedelta
from typing import Annotated
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

if not SECRET_KEY or len(SECRET_KEY) < 32:
    raise RuntimeError("JWT_SECRET_KEY debe estar configurada y tener al menos 32 caracteres.")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

def _load_users() -> dict:
    """Load bcrypt password hashes from RAG_USERS_JSON, never plaintext defaults."""
    raw_users = os.getenv("RAG_USERS_JSON")
    if not raw_users:
        raise RuntimeError("Falta RAG_USERS_JSON con usuarios y hashes bcrypt.")
    try:
        users = json.loads(raw_users)
    except json.JSONDecodeError as exc:
        raise RuntimeError("RAG_USERS_JSON no contiene JSON válido.") from exc

    if not isinstance(users, dict) or not users:
        raise RuntimeError("RAG_USERS_JSON debe contener al menos un usuario.")
    for username, user in users.items():
        if not isinstance(username, str) or not isinstance(user, dict):
            raise RuntimeError("Formato de usuario inválido en RAG_USERS_JSON.")
        if user.get("role") not in {"admin", "user"} or not user.get("hashed_password"):
            raise RuntimeError("Cada usuario requiere role (admin/user) y hashed_password.")
    return users


USERS_DB = _load_users()

class Token(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

class TokenData(BaseModel):
    username: str | None = None

class User(BaseModel):
    username: str
    role: str

def authenticate_user(username, password):
    user = USERS_DB.get(username)
    if not user or not pwd_context.verify(password, user["hashed_password"]):
        return None
    return User(username=username, role=user["role"])

def create_access_token(data):
    to_encode = data.copy()
    to_encode.update({"exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> User:
    exc = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token inválido o expirado", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise exc
    except JWTError:
        raise exc
    user = USERS_DB.get(username)
    if not user:
        raise exc
    return User(username=username, role=user["role"])


async def require_admin(current_user: Annotated[User, Depends(get_current_user)]) -> User:
    if current_user.role != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Se requieren permisos de administrador.")
    return current_user
