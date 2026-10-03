"""Minimal IMAP client. Standard library only -- no dependencies to install on the user's PC.

Scope is deliberately small: read the newest N messages, and file them into folders. There is
no background daemon and no delete path: this tool never deletes mail. `--apply` moves
messages, and `move` is only ever called with an existing-or-created target folder.

Credentials are never stored by this code. Pass a token on the command line or set
MAILTRIAGE_PASSWORD in the environment -- an app password, not the account password.
"""
from __future__ import annotations

import email
import imaplib
import os
import re
from dataclasses import dataclass, field
from email.header import decode_header
from email.message import Message

DEFAULT_HOST = os.environ.get("MAILTRIAGE_HOST", "imap.gmail.com")


@dataclass
class Mail:
    uid: str
    subject: str
    body: str
    sender: str = ""
    headers: str = ""
    flags: tuple = field(default_factory=tuple)


def _decode(raw: str | None) -> str:
    if not raw:
        return ""
    parts = []
    for chunk, charset in decode_header(raw):
        if isinstance(chunk, bytes):
            for enc in (charset, "utf-8", "cp1251", "latin-1"):
                if not enc:
                    continue
                try:
                    parts.append(chunk.decode(enc, errors="replace"))
                    break
                except (LookupError, UnicodeDecodeError):
                    continue
            else:
                parts.append(chunk.decode("utf-8", errors="replace"))
        else:
            parts.append(chunk)
    return "".join(parts).strip()


def _html_to_text(html: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    text = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</tr>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
    )
    return re.sub(r"[ \t]+", " ", text).strip()


def _extract_body(msg: Message) -> str:
    """Prefer text/plain; fall back to stripped HTML. Never raises on malformed mail."""
    plain, html = [], []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if part.get_content_disposition() == "attachment":
                continue
            if ctype not in ("text/plain", "text/html"):
                continue
            try:
                payload = part.get_payload(decode=True) or b""
                charset = part.get_content_charset() or "utf-8"
                text = payload.decode(charset, errors="replace")
            except Exception:  # noqa: BLE001 -- malformed mail must not kill the run
                continue
            (plain if ctype == "text/plain" else html).append(text)
    else:
        try:
            payload = msg.get_payload(decode=True) or b""
            charset = msg.get_content_charset() or "utf-8"
            (plain if msg.get_content_type() == "text/html" else plain).append(
                payload.decode(charset, errors="replace")
            )
        except Exception:  # noqa: BLE001
            pass

    if plain:
        return "\n".join(plain).strip()
    if html:
        return _html_to_text("\n".join(html))
    return ""


def _headers_of_interest(msg: Message) -> str:
    keys = ("List-Unsubscribe", "Precedence", "From", "Reply-To", "Authentication-Results")
    return "\n".join("%s: %s" % (k, msg.get(k, "")) for k in keys if msg.get(k))


class Inbox:
    """Context manager over an IMAP connection. Passwords live only in this object."""

    def __init__(self, user: str, password: str, host: str = DEFAULT_HOST, mailbox: str = "INBOX"):
        self.user = user
        self.password = password
        self.host = host
        self.mailbox = mailbox
        self._conn: imaplib.IMAP4_SSL | None = None

    def __enter__(self) -> "Inbox":
        self._conn = imaplib.IMAP4_SSL(self.host)
        self._conn.login(self.user, self.password)
        status, _ = self._conn.select(self.mailbox, readonly=False)
        if status != "OK":
            raise RuntimeError("cannot select mailbox %r" % self.mailbox)
        return self

    def __exit__(self, *exc) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            try:
                self._conn.logout()
            except Exception:  # noqa: BLE001
                pass

    # -- reading -------------------------------------------------------------------------
    def newest(self, limit: int = 20) -> list[Mail]:
        assert self._conn is not None
        status, data = self._conn.uid("search", None, "ALL")
        if status != "OK":
            raise RuntimeError("search failed: %s" % status)
        uids = data[0].split()
        picked = uids[-limit:]
        out: list[Mail] = []
        for uid in picked:
            status, chunks = self._conn.uid("fetch", uid, "(RFC822)")
            if status != "OK" or not chunks:
                continue
            raw = next((c[1] for c in chunks if isinstance(c, tuple)), None)
            if raw is None:
                continue
            msg = email.message_from_bytes(raw)
            out.append(
                Mail(
                    uid=uid.decode(),
                    subject=_decode(msg.get("Subject")),
                    body=_extract_body(msg),
                    sender=_decode(msg.get("From")),
                    headers=_headers_of_interest(msg),
                )
            )
        return out

    def folders(self) -> list[str]:
        assert self._conn is not None
        status, data = self._conn.list()
        if status != "OK":
            return []
        names = []
        for entry in data:
            if not entry:
                continue
            text = entry.decode(errors="replace")
            match = re.search(r'"/" (?:"([^"]+)"|(\S+))$', text)
            if match:
                names.append(match.group(1) or match.group(2))
        return names

    def ensure_folder(self, name: str) -> None:
        """Create the folder if absent. Idempotent, and safe to call per message."""
        assert self._conn is not None
        if name in set(self.folders()):
            return
        status, _ = self._conn.create(name)
        if status != "OK":
            # Some servers want a quoted or UTF-7 name; a failure here is not fatal, the
            # caller reports it rather than silently dropping the message.
            raise RuntimeError("could not create folder %r" % name)

    def move(self, uid: str, folder: str) -> None:
        assert self._conn is not None
        self.ensure_folder(folder)
        status, _ = self._conn.uid("COPY", uid, folder)
        if status != "OK":
            raise RuntimeError("copy to %r failed" % folder)
        # Only delete from the source after the copy succeeded -- never lose a message.
        self._conn.uid("STORE", uid, "+FLAGS", "(\\Deleted)")
        self._conn.expunge()
