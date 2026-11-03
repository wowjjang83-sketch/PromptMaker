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
        proc = subprocess.run(["git", *args], cwd=REPO_ROOT, env=env, capture_output=True, text=True)
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
