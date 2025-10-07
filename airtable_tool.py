# =========================
# airtable_tool.py
# =========================
import os
from typing import Optional, Tuple, Dict, Any

from pyairtable import Table
from langchain.tools import tool


# ----------------------------
# Airtable connection
# ----------------------------
def _airtable() -> Table:
    token = os.getenv("AIRTABLE_TOKEN")
    base_id = os.getenv("AIRTABLE_BASE_ID")
    table_name = os.getenv("AIRTABLE_TABLE", "customers")

    if os.getenv("AIRTABLE_DEBUG") == "1":
        safe_token = (token[:4] + "…") if token else "None"
        print(f"[airtable] base={base_id or 'None'} table={table_name} token={safe_token}")

    if not (token and base_id and table_name):
        raise RuntimeError("Missing AIRTABLE_TOKEN, AIRTABLE_BASE_ID, or AIRTABLE_TABLE env vars.")
    return Table(token, base_id, table_name)


def _escape_formula(value: str) -> str:
    # Escape single quotes for Airtable formulas
    return (value or "").replace("'", "\\'")


# ----------------------------
# Core ops
# ----------------------------
def _lookup_customer(
    name: Optional[str],
    email: Optional[str],
    username: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Lookup priority: Email (exact) -> Username (exact) -> Name (exact).
    If AIRTABLE_VIEW is set, restrict search to that view for stable ordering.
    """
    table = _airtable()
    view = os.getenv("AIRTABLE_VIEW") or None

    # 1) Email (unique)
    if email:
        formula = f"LOWER({{Email}})=LOWER('{_escape_formula(email)}')"
        rec = table.first(formula=formula, view=view)
        if rec:
            return rec

    # 2) Username (if you have a 'Username' field in Airtable)
    if username:
        formula = f"LOWER({{Username}})=LOWER('{_escape_formula(username)}')"
        rec = table.first(formula=formula, view=view)
        if rec:
            return rec

    # 3) Name (exact match)
    if name:
        formula = f"LOWER({{Name}})=LOWER('{_escape_formula(name)}')"
        rec = table.first(formula=formula, view=view)
        if rec:
            return rec

    return None


def _create_customer(
    name: str,
    email: str,
    description: Optional[str],
    username: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a new customer row. Requires name + email.
    """
    table = _airtable()
    fields = {"Name": name, "Email": email}
    if description:
        fields["Description"] = description
    if username:
        # Only set if your table actually has the 'Username' column
        fields["Username"] = username
    return table.create(fields)


def _context_from_record(rec: Dict[str, Any], created: bool) -> str:
    f = rec.get("fields", {})
    header = "Customer Profile (NEWLY CREATED):" if created else "Customer Profile:"
    return (
        f"{header}\n"
        f"- Name: {f.get('Name', '')}\n"
        f"- Email: {f.get('Email', '')}\n"
        f"- Username: {f.get('Username', '')}\n"
        f"- Description: {f.get('Description', '')}\n"
        f"- AirtableID: {rec.get('id')}\n"
    )


def upsert_customer_and_get_context(
    name: Optional[str],
    email: Optional[str],
    username: Optional[str] = None,
    description: Optional[str] = None,
    create_if_missing: bool = True,
) -> Tuple[str, bool, Optional[Dict[str, Any]]]:
    """
    Returns: (context_string, created_flag, record_dict_or_None)

    Behavior:
      - Try Email → Username → Name (exact) to find a record.
      - If found: return context, created=False.
      - If not found:
          * If create_if_missing AND BOTH name & email are provided → create, return context, created=True.
          * Else → return a 'not found' note, created=False, record=None.
    """
    existing = _lookup_customer(name=name, email=email, username=username)
    if existing:
        return _context_from_record(existing, created=False), False, existing

    if create_if_missing and name and email:
        created = _create_customer(name=name, email=email, description=description, username=username)
        return _context_from_record(created, created=True), True, created

    # Not found and we won't create
    ctx = (
        "Customer not found.\n"
        f"- Name: {name or ''}\n"
        f"- Email: {email or ''}\n"
        f"- Username: {username or ''}\n"
        "Provide both name and email to create a new record."
    )
    return ctx, False, None


# ----------------------------
# LangChain Tool
# ----------------------------
# @tool("customer_context", return_direct=False)
def customer_context(customer_json: str) -> str:
    """
    Look up a customer by email OR username OR name; optionally create when both name & email are present.

    Input JSON (any subset of lookup keys is fine):
      {
        "name": "Ada Lovelace",          # optional if email or username present
        "email": "ada@example.com",      # optional if name or username present
        "username": "ada_therapy",       # optional if name or email present
        "description": "therapist ...",  # optional; used only if creating
        "create_if_missing": true        # optional; default true
      }

    Output:
      Human-readable context block + STATUS line:
      "STATUS: created=<true|false>; found=<true|false>"
    """
    import json
    try:
        data = json.loads(customer_json)
    except Exception:
        return (
            "Invalid JSON. Expected keys: name (opt), email (opt), username (opt), "
            "description (opt), create_if_missing (opt)."
        )

    name = (data.get("name") or "").strip() or None
    email = (data.get("email") or "").strip() or None
    username = (data.get("username") or "").strip() or None
    description = (data.get("description") or "").strip() or None
    create_if_missing = bool(data.get("create_if_missing", True))

    if not (name or email or username):
        return "Missing lookup keys: provide at least one of name, email, or username."

    try:
        ctx, created, rec = upsert_customer_and_get_context(
            name=name,
            email=email,
            username=username,
            description=description,
            create_if_missing=create_if_missing,
        )
        status = f"created={'true' if created else 'false'}; found={'true' if rec else 'false'}"
        return f"{ctx}\nSTATUS: {status}"
    except Exception as e:
        return f"ERROR: {e}"
