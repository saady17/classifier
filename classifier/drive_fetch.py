"""Downloads resumes from a Google Drive folder shared with the service account (Viewer), into
eval/resumes/. Uses GOOGLE_SA_EMAIL / GOOGLE_SA_PRIVATE_KEY; nothing is uploaded anywhere.

    python -m classifier.drive_fetch <folder_id>"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
from email.utils import formatdate  # noqa: F401  (keeps stdlib-only imports obvious)

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

OUT = os.environ.get("RESUMES_OUT", "eval/resumes")
TYPES = {"application/pdf": ".pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx", "text/plain": ".txt"}


def token() -> str:
    email = os.environ["GOOGLE_SA_EMAIL"]
    key = serialization.load_pem_private_key(os.environ["GOOGLE_SA_PRIVATE_KEY"].replace("\\n", "\n").encode(), password=None)
    now = int(time.time())
    enc = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()  # noqa: E731
    unsigned = f"{enc({'alg': 'RS256', 'typ': 'JWT'})}.{enc({'iss': email, 'scope': 'https://www.googleapis.com/auth/drive.readonly', 'aud': 'https://oauth2.googleapis.com/token', 'iat': now, 'exp': now + 3600})}"
    sig = base64.urlsafe_b64encode(key.sign(unsigned.encode(), padding.PKCS1v15(), hashes.SHA256())).rstrip(b"=").decode()
    r = httpx.post("https://oauth2.googleapis.com/token", data={"grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer", "assertion": f"{unsigned}.{sig}"})
    r.raise_for_status()
    return r.json()["access_token"]


def main(folder: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    h = {"Authorization": f"Bearer {token()}"}
    files, page = [], None
    while True:
        params = {"q": f"'{folder}' in parents and trashed = false", "fields": "nextPageToken, files(id, name, mimeType, size)",
                  "pageSize": 1000, "supportsAllDrives": "true", "includeItemsFromAllDrives": "true"}
        if page:
            params["pageToken"] = page
        r = httpx.get("https://www.googleapis.com/drive/v3/files", params=params, headers=h)
        r.raise_for_status()
        data = r.json()
        files += data.get("files", [])
        page = data.get("nextPageToken")
        if not page:
            break
    if not files:
        print("No files visible. Share the folder with", os.environ["GOOGLE_SA_EMAIL"], "as Viewer.")
        return
    got = skipped = 0
    for f in files:
        ext = TYPES.get(f["mimeType"])
        if not ext:
            skipped += 1
            continue
        name = re.sub(r"[^\w.-]+", "_", f["name"].strip())
        name = name if name.lower().endswith(ext) else name + ext
        path = os.path.join(OUT, f"{f['id'][:8]}_{name}")
        if os.path.exists(path):
            continue
        r = httpx.get(f"https://www.googleapis.com/drive/v3/files/{f['id']}", params={"alt": "media", "supportsAllDrives": "true"}, headers=h, timeout=60)
        r.raise_for_status()
        open(path, "wb").write(r.content)
        got += 1
    print(f"{len(files)} files in folder: {got} downloaded, {skipped} skipped (not PDF/DOCX/TXT)")


if __name__ == "__main__":
    main(sys.argv[1])
