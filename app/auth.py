"""
Authentication blueprint: SQLite + JWT signup / login / me / logout.

Storage: ``app/users.db`` SQLite file (auto-created on first run).
Tokens : HS256 JWT, 7-day expiry, signed with ``Config.SECRET_KEY``.
Hashing: bcrypt via flask-bcrypt.

The blueprint is mounted at ``/api/auth`` by ``app/app.py``. The model
routes (``/`` ``/predict`` ``/health`` ``/about``) are untouched.

Endpoints
---------
POST /api/auth/signup    {email, password, name?}              -> {token, user}
POST /api/auth/login     {email, password}                     -> {token, user}
GET  /api/auth/me        Authorization: Bearer <token>         -> {user}
POST /api/auth/logout    (client-side; here for completeness)  -> {ok}
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from functools import wraps
from pathlib import Path
from typing import Optional

import jwt
from flask import Blueprint, current_app, g, jsonify, request
from flask_bcrypt import Bcrypt
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError

log = logging.getLogger("app.auth")

db = SQLAlchemy()
bcrypt = Bcrypt()
auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
TOKEN_TTL_DAYS = 7


class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    name = db.Column(db.String(120), nullable=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow, nullable=False)

    def to_public(self) -> dict:
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name or self.email.split("@")[0],
            "created_at": self.created_at.isoformat() + "Z",
        }


def _make_token(user: User) -> str:
    payload = {
        # RFC 7519 requires `sub` to be a string. PyJWT >= 2.10 enforces this.
        "sub": str(user.id),
        "email": user.email,
        "iat": dt.datetime.utcnow(),
        "exp": dt.datetime.utcnow() + dt.timedelta(days=TOKEN_TTL_DAYS),
    }
    secret = current_app.config["SECRET_KEY"]
    token = jwt.encode(payload, secret, algorithm="HS256")
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    return token


def _decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(
            token, current_app.config["SECRET_KEY"], algorithms=["HS256"]
        )
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def _err(http_status: int, code: str, message: str):
    return jsonify({"ok": False, "error": {"code": code, "message": message}}), http_status


def auth_required(fn):
    """Decorator: require a valid bearer token; populate ``g.current_user``."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return _err(401, "missing_token", "Authentication required.")
        token = header[len("Bearer ") :].strip()
        payload = _decode_token(token)
        if payload is None:
            return _err(401, "invalid_token", "Token is invalid or expired.")
        try:
            sub_int = int(payload["sub"])
        except (KeyError, ValueError, TypeError):
            return _err(401, "invalid_token", "Token is invalid or expired.")
        user = db.session.get(User, sub_int)
        if user is None:
            return _err(401, "user_gone", "Account no longer exists.")
        g.current_user = user
        return fn(*args, **kwargs)
    return wrapper


@auth_bp.post("/signup")
def signup():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    name = (data.get("name") or "").strip() or None

    if not EMAIL_RE.match(email):
        return _err(400, "invalid_email", "Please enter a valid email address.")
    if len(password) < 8:
        return _err(400, "weak_password", "Password must be at least 8 characters.")
    if name is not None and len(name) > 120:
        return _err(400, "invalid_name", "Name is too long.")

    pw_hash = bcrypt.generate_password_hash(password).decode("utf-8")
    user = User(email=email, password_hash=pw_hash, name=name)
    try:
        db.session.add(user)
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        return _err(409, "email_taken", "An account with this email already exists.")

    token = _make_token(user)
    log.info("signup ok user_id=%s email=%s", user.id, user.email)
    return jsonify({"ok": True, "token": token, "user": user.to_public()}), 201


@auth_bp.post("/login")
def login():
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return _err(400, "missing_fields", "Email and password are required.")

    user = User.query.filter_by(email=email).first()
    if user is None or not bcrypt.check_password_hash(user.password_hash, password):
        return _err(401, "invalid_credentials", "Email or password is incorrect.")

    token = _make_token(user)
    log.info("login ok user_id=%s email=%s", user.id, user.email)
    return jsonify({"ok": True, "token": token, "user": user.to_public()}), 200


@auth_bp.get("/me")
@auth_required
def me():
    return jsonify({"ok": True, "user": g.current_user.to_public()}), 200


@auth_bp.post("/logout")
def logout():
    # Stateless JWT — client just discards the token. Endpoint exists for
    # symmetry and so that frontend code has a single, named auth flow.
    return jsonify({"ok": True}), 200


def init_auth(app, db_path: Optional[Path] = None) -> None:
    """Initialise the auth subsystem on an existing Flask app."""
    if db_path is None:
        db_path = Path(app.root_path) / "users.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)
    bcrypt.init_app(app)
    app.register_blueprint(auth_bp)

    with app.app_context():
        db.create_all()

    log.info("Auth initialised (sqlite at %s)", db_path)
