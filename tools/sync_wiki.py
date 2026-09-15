"""
Helper script to synchronize the local docs/wiki/ directory with the GitHub Wiki repository.
Usage:
    python tools/sync_wiki.py [--repo <github-wiki-git-url>] [--dry-run]
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

def sync_wiki(repo_url: str, dry_run: bool = False):
    repo_root = Path(__file__).resolve().parent.parent
    wiki_src = repo_root / "docs" / "wiki"
    
    if not wiki_src.exists():
        print(f"Error: Source directory {wiki_src} does not exist.")
        sys.exit(1)
        
    temp_dir = repo_root / "build" / "wiki_clone"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        
    print(f"Cloning GitHub Wiki from {repo_url}...")
    try:
        subprocess.run(["git", "clone", repo_url, str(temp_dir)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to clone wiki repository: {e}")
        print("Tip: Ensure the GitHub Wiki feature is enabled on the repository settings and that at least one page has been created.")
        sys.exit(1)
        
    print(f"Copying files from {wiki_src} to {temp_dir}...")
    copied_count = 0
    for f in wiki_src.glob("*.md"):
        dst = temp_dir / f.name
        shutil.copy2(f, dst)
        copied_count += 1
        print(f"  Copied: {f.name}")
        
    print(f"Total files copied: {copied_count}")
    
    if dry_run:
        print("Dry-run mode active. Skipping git commit and push.")
        return
        
    print("Committing and pushing changes...")
    try:
        subprocess.run(["git", "add", "."], cwd=temp_dir, check=True)
        status = subprocess.run(["git", "status", "--porcelain"], cwd=temp_dir, capture_output=True, text=True, check=True)
        if not status.stdout.strip():
            print("No changes detected in Wiki. Everything is up to date.")
            return
            
        subprocess.run(["git", "commit", "-m", "Sync Wiki documentation from docs/wiki/"], cwd=temp_dir, check=True)
        subprocess.run(["git", "push"], cwd=temp_dir, check=True)
        print("GitHub Wiki successfully updated.")
    except subprocess.CalledProcessError as e:
        print(f"Git operation failed: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Sync docs/wiki/ markdown files to GitHub Wiki")
    parser.add_argument("--repo", default="https://github.com/harshitkhandelwal208/hk.wiki.git", help="Git URL of GitHub Wiki repository")
    parser.add_argument("--dry-run", action="store_true", help="Perform file copying without git commit and push")
    args = parser.parse_args()
    
    sync_wiki(args.repo, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
