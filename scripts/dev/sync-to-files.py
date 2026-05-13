from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
from datetime import datetime

# config
DATA_DIR = "data"
REMOTE_DEST = "icgroup:/var/www/files.ischemist.com/retrocast/data/"

# only these paths (relative to data/) get hashed and uploaded
WHITELIST = ["1-benchmarks/definitions", "1-benchmarks/stocks", "2-raw", "3-processed", "4-scored", "5-results"]

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta name="color-scheme" content="light dark">
    <title>RetroCast Data Index</title>
    <style>
        :root {{
            color-scheme: light dark;
            --page-background: #f4f4f4;
            --surface-background: #ffffff;
            --text-color: #111111;
            --muted-text-color: #666666;
            --meta-text-color: #888888;
            --border-color: #dddddd;
            --header-background: #333333;
            --header-text-color: #ffffff;
            --row-hover-background: #f5f5f5;
            --link-color: #005fcc;
            --visited-link-color: #5d2ca0;
        }}

        @media (prefers-color-scheme: dark) {{
            :root {{
                --page-background: #101418;
                --surface-background: #161b22;
                --text-color: #e6edf3;
                --muted-text-color: #9da7b3;
                --meta-text-color: #8b949e;
                --border-color: #30363d;
                --header-background: #21262d;
                --header-text-color: #f0f6fc;
                --row-hover-background: #1f2630;
                --link-color: #7ab7ff;
                --visited-link-color: #c297ff;
            }}
        }}

        body {{
            font-family: monospace;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: var(--page-background);
            color: var(--text-color);
        }}
        h1 {{ border-bottom: 2px solid var(--header-background); }}
        table {{ width: 100%; border-collapse: collapse; background: var(--surface-background); }}
        th, td {{ padding: 8px; border: 1px solid var(--border-color); text-align: left; }}
        th {{ background-color: var(--header-background); color: var(--header-text-color); }}
        tr:hover {{ background-color: var(--row-hover-background); }}
        a {{ color: var(--link-color); }}
        a:visited {{ color: var(--visited-link-color); }}
        .hash {{ font-size: 0.8em; color: var(--muted-text-color); }}
        .meta {{ font-size: 0.8em; color: var(--meta-text-color); margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>RetroCast Data Artifacts</h1>
    <div class="meta">
        Generated: {timestamp}<br>
        Verified Files Only.
    </div>
    <table>
        <tr><th>File</th><th>Size</th><th>SHA256</th></tr>
        {rows}
    </table>
</body>
</html>
"""


def hash_file(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upload", action="store_true", help="rsync whitelist to vps")
    args = parser.parse_args()

    file_list = []  # (relpath, hash, size)

    print(f"scanning whitelist in {DATA_DIR}...")

    # 1. build index from whitelist
    for path in WHITELIST:
        full_path = os.path.join(DATA_DIR, path)
        if not os.path.exists(full_path):
            print(f"warning: whitelisted path not found: {path}")
            continue

        # walk the subdirs of the whitelisted path
        for root, _, files in os.walk(full_path):
            for name in files:
                if name.startswith("."):
                    continue  # ignore .DS_Store

                filepath = os.path.join(root, name)
                relpath = os.path.relpath(filepath, DATA_DIR)

                print(f"hashing {relpath}...")
                fhash = hash_file(filepath)
                fsize = os.path.getsize(filepath)

                file_list.append((relpath, fhash, fsize))

    # 2. write SHA256SUMS
    sums_content = "".join([f"{h}  {p}\n" for p, h, _ in file_list])
    sums_path = os.path.join(DATA_DIR, "SHA256SUMS")
    with open(sums_path, "w") as f:
        f.write(sums_content)

    # 3. write index.html
    rows = ""
    for relpath, fhash, fsize in sorted(file_list):
        size_str = f"{fsize / 1024 / 1024:.2f} MB" if fsize > 1024 * 1024 else f"{fsize / 1024:.2f} KB"
        rows += f"<tr><td><a href='{relpath}'>{relpath}</a></td><td>{size_str}</td><td class='hash'>{fhash}</td></tr>"

    html_path = os.path.join(DATA_DIR, "index.html")
    with open(html_path, "w") as f:
        f.write(HTML_TEMPLATE.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), rows=rows))

    print(f"index built. {len(file_list)} files.")

    # 4. upload if requested
    if args.upload:
        print("syncing to vps...")
        # we change dir to DATA_DIR so --relative works cleanly
        # we include SHA256SUMS and index.html explicitly
        cmd = ["rsync", "-avz", "--progress", "--relative", "SHA256SUMS", "index.html"] + WHITELIST + [REMOTE_DEST]

        # execution context needs to be inside DATA_DIR for relative paths to match
        subprocess.run(cmd, cwd=DATA_DIR, check=True)


if __name__ == "__main__":
    main()
