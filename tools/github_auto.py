"""GitHub 프로젝트 자동 관리 스크립트.

토큰은 파일에서 실행 시점에만 읽고, 화면/로그에 절대 출력하지 않습니다.

토큰 파일 경로 우선순위
  1) --token-file 인자
  2) 환경변수 PROMPTMAKER_TOKEN_FILE
  3) <저장소>/token/git_token.txt
  4) F:\\AI_Dev\\PromptMaker\\token\\git_token.txt

사용 예
  python tools/github_auto.py status
  python tools/github_auto.py push
  python tools/github_auto.py pr --title "기능 추가" --body "설명" --base main
  python tools/github_auto.py auto -m "요소 목록 추가" --pr
  python tools/github_auto.py issue --title "버그" --body "내용"
  python tools/github_auto.py prs
  python tools/github_auto.py issues
"""
from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REPO = os.environ.get("PROMPTMAKER_REPO", "wowjjang83-sketch/PromptMaker")
TOKEN_CANDIDATES = [
    os.path.join(REPO_ROOT, "token", "git_token.txt"),
    r"F:\AI_Dev\PromptMaker\token\git_token.txt",
]
API = "https://api.github.com"


class TokenError(RuntimeError):
    pass


def find_token_file(explicit: str | None = None) -> str:
    candidates = [explicit, os.environ.get("PROMPTMAKER_TOKEN_FILE")] + TOKEN_CANDIDATES
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    raise TokenError("토큰 파일을 찾을 수 없습니다. --token-file 또는 PROMPTMAKER_TOKEN_FILE 을 지정하세요.")


def read_token(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        token = f.read().strip()
    if not token:
        raise TokenError("토큰 파일이 비어 있습니다.")
    return token


def _mask(text: str, token: str) -> str:
    return text.replace(token, "***") if token else text


# ------------------------------------------------------------------ git
def git(*args: str, token: str | None = None, check: bool = True) -> str:
    env = os.environ.copy()
    askpass = None
    if token:
        # GIT_ASKPASS 로 토큰을 전달 (명령줄/URL 에 토큰이 남지 않음)
        env["PM_GIT_TOKEN"] = token
        env["GIT_TERMINAL_PROMPT"] = "0"
        if os.name == "nt":
            fd, askpass = tempfile.mkstemp(suffix=".bat")
            content = "@echo off\r\necho %1 | findstr /i \"username\" >nul && (echo x-access-token) || (echo %PM_GIT_TOKEN%)\r\n"
        else:
            fd, askpass = tempfile.mkstemp(suffix=".sh")
            content = '#!/bin/sh\ncase "$1" in *sername*) echo x-access-token;; *) echo "$PM_GIT_TOKEN";; esac\n'
        with os.fdopen(fd, "w") as f:
            f.write(content)
        os.chmod(askpass, stat.S_IRWXU)
        env["GIT_ASKPASS"] = askpass
    try:
        proc = subprocess.run(["git", *args], cwd=REPO_ROOT, env=env, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
    finally:
        if askpass and os.path.exists(askpass):
            os.remove(askpass)
    out = _mask((proc.stdout or "") + (proc.stderr or ""), token or "")
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} 실패:\n{out}")
    return out.strip()


def current_branch() -> str:
    return git("rev-parse", "--abbrev-ref", "HEAD")


# ------------------------------------------------------------------ api
def api(method: str, path: str, token: str, data: dict | None = None):
    req = urllib.request.Request(
        API + path, method=method,
        data=json.dumps(data).encode("utf-8") if data is not None else None,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "PromptMaker-auto",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = _mask(exc.read().decode("utf-8", "replace"), token)
        raise RuntimeError(f"GitHub API 오류 {exc.code}: {detail}") from None


# ------------------------------------------------------------- commands
def cmd_status(a, token):
    print(git("status", "-sb"))
    me = api("GET", "/user", token)
    print(f"GitHub 계정: {me.get('login')}  /  저장소: {a.repo}")


def cmd_pull(a, token):
    print(git("pull", "origin", a.branch or current_branch(), token=token))


def cmd_push(a, token):
    branch = a.branch or current_branch()
    print(git("push", "-u", "origin", branch, token=token))
    print(f"푸시 완료: {branch}")


def cmd_pr(a, token):
    head = a.head or current_branch()
    pr = api("POST", f"/repos/{a.repo}/pulls", token,
             {"title": a.title, "body": a.body or "", "head": head, "base": a.base, "draft": a.draft})
    print(f"PR 생성: #{pr['number']} {pr['html_url']}")


def cmd_prs(a, token):
    for pr in api("GET", f"/repos/{a.repo}/pulls?state=open", token):
        print(f"#{pr['number']} [{pr['head']['ref']} -> {pr['base']['ref']}] {pr['title']}")


def cmd_issue(a, token):
    issue = api("POST", f"/repos/{a.repo}/issues", token, {"title": a.title, "body": a.body or ""})
    print(f"이슈 생성: #{issue['number']} {issue['html_url']}")


def cmd_issues(a, token):
    for it in api("GET", f"/repos/{a.repo}/issues?state=open", token):
        if "pull_request" not in it:
            print(f"#{it['number']} {it['title']}")


def cmd_auto(a, token):
    """add -> commit -> push (-> PR). main 브랜치에서는 새 브랜치를 만듭니다."""
    branch = current_branch()
    if branch in ("main", "master"):
        if not a.branch:
            raise SystemExit("main 에 직접 푸시하지 않습니다. --branch 로 작업 브랜치 이름을 지정하세요.")
        git("checkout", "-b", a.branch)
        branch = a.branch
    git("add", "-A")
    if git("status", "--porcelain"):
        print(git("commit", "-m", a.message))
    else:
        print("커밋할 변경 사항이 없습니다.")
    print(git("push", "-u", "origin", branch, token=token))
    if a.pr:
        pr = api("POST", f"/repos/{a.repo}/pulls", token,
                 {"title": a.message, "body": a.body or "", "head": branch, "base": a.base, "draft": True})
        print(f"PR 생성: #{pr['number']} {pr['html_url']}")


# ------------------------------------------------------------- release
VERSION_FILES = {
    "pyproject.toml": re.compile(r'^(version\s*=\s*")([^"]+)(")', re.M),
    os.path.join("promptmaker", "__init__.py"): re.compile(r'^(__version__\s*=\s*")([^"]+)(")', re.M),
}


def read_version() -> str:
    versions = set()
    for rel, rx in VERSION_FILES.items():
        with open(os.path.join(REPO_ROOT, rel), encoding="utf-8") as f:
            m = rx.search(f.read())
        if not m:
            raise RuntimeError(f"{rel} 에서 버전을 찾지 못했습니다.")
        versions.add(m.group(2))
    if len(versions) != 1:
        raise RuntimeError(f"버전 파일들이 서로 다릅니다: {sorted(versions)}")
    return versions.pop()


def bump_version(version: str, part: str) -> str:
    major, minor, patch = (list(map(int, version.split(".")[:3])) + [0, 0, 0])[:3]
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def write_version(version: str) -> None:
    for rel, rx in VERSION_FILES.items():
        path = os.path.join(REPO_ROOT, rel)
        with open(path, encoding="utf-8", newline="") as f:
            text = f.read()
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(rx.sub(lambda m: m.group(1) + version + m.group(3), text, count=1))


def last_tag() -> str | None:
    tags = [t for t in git("tag", "--list", "v*", "--sort=-v:refname").split() if t]
    return tags[0] if tags else None


def release_notes(since: str | None) -> str:
    """Merged PR titles since the previous tag (GitHub merge commits: subject + PR title as body)."""
    rng = f"{since}..HEAD" if since else "HEAD"
    log = git("log", rng, "--first-parent", "--merges", "--format=%s%x1f%b%x1e")
    lines = []
    for entry in log.split("\x1e"):
        if not entry.strip():
            continue
        subject, _, body = entry.strip().partition("\x1f")
        m = re.match(r"Merge pull request #(\d+)", subject)
        title = body.strip().splitlines()[0] if body.strip() else subject
        lines.append(f"- {title} (#{m.group(1)})" if m else f"- {title}")
    if not lines:  # no PR merges: plain commits
        lines = [f"- {s}" for s in git("log", rng, "--first-parent", "--format=%s").splitlines() if s]
    return "\n".join(reversed(lines))


def cmd_release(a, token):
    """Versioning rule: every PR to main bumps the version ("release prepare"),
    and after the merge "release publish" tags main and creates the GitHub Release."""
    if a.action == "check":
        print(f"버전 파일: {read_version()}  /  마지막 태그: {last_tag() or '(없음)'}")
        try:
            r = api("GET", f"/repos/{a.repo}/releases/latest", token)
            print(f"최신 GitHub Release: {r['tag_name']} {r['html_url']}")
        except RuntimeError:
            print("최신 GitHub Release: (없음)")
        return
    if a.action == "prepare":
        if current_branch() in ("main", "master"):
            raise SystemExit("main 에서는 하지 않습니다. 작업 브랜치(PR)에서 버전을 올리세요.")
        old = read_version()
        new = a.version or bump_version(old, a.bump)
        write_version(new)
        git("add", *VERSION_FILES)
        print(git("commit", "-m", f"버전 {new}"))
        print(f"버전 {old} -> {new}. 푸시 후 PR 을 병합하고 'release publish' 를 실행하세요.")
        return
    # publish: tag the current main and create the GitHub Release
    if current_branch() not in ("main", "master"):
        raise SystemExit("main 에서 실행하세요 (병합 후).")
    git("fetch", "origin", token=token)
    if git("rev-parse", "HEAD") != git("rev-parse", "origin/" + current_branch()):
        raise SystemExit("로컬 main 이 origin 과 다릅니다. 먼저 pull 하세요.")
    version = read_version()
    tag = f"v{version}"
    prev = last_tag()
    if tag in git("tag", "--list", tag).split():
        raise SystemExit(f"{tag} 태그가 이미 있습니다. PR 에서 'release prepare' 로 버전을 올리세요.")
    notes = a.notes or release_notes(prev)
    git("tag", "-a", tag, "-m", f"PromptMaker {version}\n\n{notes}")
    print(git("push", "origin", tag, token=token))
    r = api("POST", f"/repos/{a.repo}/releases", token,
            {"tag_name": tag, "name": f"PromptMaker {version}",
             "body": notes + (f"\n\n변경 비교: https://github.com/{a.repo}/compare/{prev}...{tag}" if prev else "")})
    print(f"릴리스 생성: {r['html_url']}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="PromptMaker GitHub 자동 관리")
    p.add_argument("--token-file", default=None)
    p.add_argument("--repo", default=DEFAULT_REPO)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    for name in ("pull", "push"):
        s = sub.add_parser(name)
        s.add_argument("--branch", default=None)
    s = sub.add_parser("pr")
    s.add_argument("--title", required=True)
    s.add_argument("--body", default="")
    s.add_argument("--head", default=None)
    s.add_argument("--base", default="main")
    s.add_argument("--draft", action="store_true")
    sub.add_parser("prs")
    s = sub.add_parser("issue")
    s.add_argument("--title", required=True)
    s.add_argument("--body", default="")
    sub.add_parser("issues")
    s = sub.add_parser("auto")
    s.add_argument("-m", "--message", required=True)
    s.add_argument("--branch", default=None)
    s.add_argument("--base", default="main")
    s.add_argument("--body", default="")
    s.add_argument("--pr", action="store_true")
    s = sub.add_parser("release", help="버전 관리: check / prepare (PR 에서 버전 올리기) / publish (병합 후 태그+Release)")
    s.add_argument("action", choices=["check", "prepare", "publish"])
    s.add_argument("--bump", choices=["major", "minor", "patch"], default="minor")
    s.add_argument("--version", default=None, help="올릴 버전을 직접 지정 (예: 0.3.0)")
    s.add_argument("--notes", default=None, help="릴리스 설명 (생략 시 병합된 PR 제목으로 자동 작성)")
    a = p.parse_args(argv)

    try:
        token = read_token(find_token_file(a.token_file))
        globals()[f"cmd_{a.cmd}"](a, token)
    except (TokenError, RuntimeError) as exc:
        print(f"오류: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
