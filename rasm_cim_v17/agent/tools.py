"""Инструменты агента: поиск и загрузка страниц.

Правила, которые важны для честности:

* сеть **выключена по умолчанию** — включается флагом `--online` или полем
  `online: true` в конфиге; без него агент отвечает «нет доступа к сети»;
* хост проверяется по списку разрешённых, редиректы не разыменовываются
  (иначе список обходится одним 302);
* таймаут и лимит байт обязательны, ответы кэшируются в доме;
* любая ошибка возвращается как `ToolResult(ok=False, error=…)` — агент
  обязан сказать об этом пользователю, а не придумать ответ.

Провайдеры:

* `GenericEndpoint` — любой поисковый бэкенд, отдающий JSON вида
  `{"results": [{"title": …, "url": …, "text": …}]}` (SearxNG, свой шлюз,
  локальный стенд в тестах). Задаётся `RASM_CIM_SEARCH_ENDPOINT`;
* `Wikipedia` — публичный API Википедии (`ru.wikipedia.org`), по умолчанию;
* `GithubSearch` — поиск репозиториев через `api.github.com`, без токена.

Каждый результат несёт `source` — URL, который агент обязан показать.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser

USER_AGENT = "rasm-cim-agent/0.18 (+https://github.com/podshoevbunyod16-sketch/Bunyod)"
DEFAULT_TIMEOUT = 10.0
MAX_BYTES = 400_000
CACHE_TTL = 60 * 60 * 24

#: домены, к которым сетевому инструменту разрешено ходить
ALLOWED_HOSTS = (
    "ru.wikipedia.org", "wikipedia.org", "ru.m.wikipedia.org",
    "api.github.com", "github.com", "raw.githubusercontent.com",
    "localhost", "127.0.0.1",
)


@dataclass
class ToolResult:
    ok: bool
    query: str = ""
    title: str = ""
    source: str = ""
    text: str = ""
    error: str = ""
    cached: bool = False
    extra: dict = field(default_factory=dict)
    raw: str = ""

    def as_dict(self) -> dict:
        return {"ok": self.ok, "query": self.query, "title": self.title,
                "source": self.source, "text": self.text, "error": self.error,
                "cached": self.cached}


class _Text(HTMLParser):
    """Вытащить текст из HTML без сторонних библиотек."""

    SKIP = {"script", "style", "noscript", "sup", "table"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag in ("p", "div", "br", "li", "h1", "h2", "h3") and not self._skip:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip and data.strip():
            self.parts.append(data.strip() + " ")

    def text(self) -> str:
        raw = "".join(self.parts)
        raw = re.sub(r"[ \t]+", " ", raw)
        return re.sub(r"\n{3,}", "\n\n", raw).strip()


def html_to_text(html: str) -> str:
    parser = _Text()
    try:
        parser.feed(html)
    except Exception:                     # битая разметка — берём как есть
        return re.sub(r"<[^>]+>", " ", html)
    return parser.text()


class Cache:
    """Кэш ответов в доме: одна JSON-запись на URL."""

    def __init__(self, home: str):
        self.path = os.path.join(home, "agent_cache.json")
        self._data: dict = {}
        try:
            with open(self.path, encoding="utf-8") as handle:
                self._data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            self._data = {}

    def get(self, key: str) -> dict | None:
        item = self._data.get(key)
        if not item:
            return None
        if time.time() - item.get("at", 0) > CACHE_TTL:
            return None
        return item

    def put(self, key: str, value: dict) -> None:
        self._data[key] = {"at": time.time(), "value": value}
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as handle:
                json.dump(self._data, handle, ensure_ascii=False)
        except OSError:
            pass


def host_allowed(url: str, allowed: tuple[str, ...] = ALLOWED_HOSTS) -> bool:
    try:
        host = urllib.parse.urlparse(url).hostname or ""
    except ValueError:
        return False
    host = host.casefold()
    return any(host == item or host.endswith("." + item) for item in allowed)


class Web:
    """Сетевой инструмент: `search()` и `fetch()`."""

    def __init__(self, home: str, *, online: bool = False,
                 endpoint: str | None = None,
                 allowed: tuple[str, ...] = ALLOWED_HOSTS):
        self.home = home
        self.online = online
        self.endpoint = endpoint or os.environ.get("RASM_CIM_SEARCH_ENDPOINT", "")
        self.allowed = allowed
        self.cache = Cache(home)
        self.last_error = ""

    # ---------------------------------------------------------------- fetch
    def fetch(self, url: str) -> ToolResult:
        if not self.online:
            return ToolResult(False, source=url,
                              error="сеть выключена (включите --online)")
        if not host_allowed(url, self.allowed):
            return ToolResult(False, source=url,
                              error=f"хост не разрешён: {urllib.parse.urlparse(url).hostname}")
        cached = self.cache.get("fetch:" + url)
        if cached:
            value = cached["value"]
            return ToolResult(True, title=value.get("title", ""), source=url,
                              text=value.get("text", ""),
                              raw=value.get("raw", ""), cached=True)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                       "Accept": "*/*"})
        try:
            with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT) as response:
                raw = response.read(MAX_BYTES + 1)
                if len(raw) > MAX_BYTES:
                    return ToolResult(False, source=url, error="ответ слишком большой")
                charset = response.headers.get_content_charset() or "utf-8"
                ctype = response.headers.get_content_type() or "text/html"
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                OSError, ValueError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return ToolResult(False, source=url, error=f"нет доступа к {url} ({self.last_error})")
        body = raw.decode(charset, errors="replace")
        ctype = ""
        try:
            ctype = response.headers.get_content_type()      # type: ignore[possibly-undefined]
        except Exception:
            ctype = "text/html"
        if "json" in ctype or body.lstrip()[:1] in "{[":
            text = _json_text(body)
            title = ""
        elif "html" in ctype or "<html" in body[:400].casefold():
            text = html_to_text(body)
            title = _html_title(body)
        else:
            text, title = body, ""
        self.cache.put("fetch:" + url, {"title": title, "text": text,
                                       "raw": body[:20000]})
        return ToolResult(True, title=title, source=url, text=text, raw=body)

    # --------------------------------------------------------------- search
    def search(self, query: str, *, limit: int = 3) -> list[ToolResult]:
        query = (query or "").strip()
        if not query:
            return [ToolResult(False, error="пустой запрос")]
        if not self.online:
            return [ToolResult(False, query=query,
                               error="сеть выключена (включите --online)")]
        cached = self.cache.get("search:" + query)
        if cached:
            return [ToolResult(True, query=query, title=item.get("title", ""),
                               source=item.get("url", ""), text=item.get("text", ""),
                               cached=True) for item in cached["value"]["results"]]
        if self.endpoint == "github":
            results = self._search_github(query, limit)
        elif self.endpoint:
            results = self._search_endpoint(query, limit)
        else:
            results = self._search_wikipedia(query, limit)
        if results and results[0].ok:
            self.cache.put("search:" + query, {"results": [
                {"title": r.title, "url": r.source, "text": r.text} for r in results
                if r.ok]})
        return results

    def _search_endpoint(self, query: str, limit: int) -> list[ToolResult]:
        url = self.endpoint.replace("{query}", urllib.parse.quote(query))
        if "{query}" not in self.endpoint:
            joiner = "&" if "?" in url else "?"
            url = f"{url}{joiner}q={urllib.parse.quote(query)}"
        fetched = self.fetch(url)
        if not fetched.ok:
            return [ToolResult(False, query=query, error=fetched.error)]
        try:
            payload = json.loads(fetched.raw or fetched.text)
        except json.JSONDecodeError:
            return [ToolResult(False, query=query, error="поисковый бэкенд вернул не JSON")]
        items = payload.get("results") or []
        out = []
        for item in items[:limit]:
            out.append(ToolResult(True, query=query, title=item.get("title", ""),
                                  source=item.get("url", ""), text=item.get("text", "")))
        if not out:
            return [ToolResult(True, query=query, error="ничего не найдено")]
        return out

    def _search_github(self, query: str, limit: int) -> list[ToolResult]:
        """Поиск репозиториев через api.github.com — работает без токена."""
        api = ("https://api.github.com/search/repositories?per_page="
               f"{limit}&q=" + urllib.parse.quote(query))
        found = self.fetch(api)
        if not found.ok:
            return [ToolResult(False, query=query, error=found.error)]
        try:
            payload = json.loads(found.raw or found.text)
            items = payload.get("items") or []
        except (json.JSONDecodeError, AttributeError):
            return [ToolResult(False, query=query,
                               error="GitHub вернул неожиданный ответ")]
        out = [ToolResult(True, query=query, title=item.get("full_name", ""),
                          source=item.get("html_url", ""),
                          text=item.get("description") or "",
                          extra={"stars": item.get("stargazers_count", 0)})
               for item in items[:limit]]
        if not out:
            return [ToolResult(True, query=query, error="ничего не найдено")]
        return out

    def _search_wikipedia(self, query: str, limit: int) -> list[ToolResult]:
        api = ("https://ru.wikipedia.org/w/api.php?action=query&list=search"
               "&srlimit={limit}&format=json&utf8=1&srsearch="
               + urllib.parse.quote(query)).replace("{limit}", str(limit))
        found = self.fetch(api)
        if not found.ok:
            return [ToolResult(False, query=query, error=found.error)]
        try:
            payload = json.loads(found.raw or found.text)
            hits = payload["query"]["search"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return [ToolResult(False, query=query, error="Википедия вернула неожиданный ответ")]
        out = []
        for hit in hits[:limit]:
            title = hit.get("title", "")
            snippet = html_to_text(hit.get("snippet", ""))
            out.append(ToolResult(True, query=query, title=title,
                                  source="https://ru.wikipedia.org/wiki/"
                                         + urllib.parse.quote(title.replace(" ", "_")),
                                  text=snippet, extra={"wikipedia_title": title}))
        if not out:
            return [ToolResult(True, query=query, error="ничего не найдено")]
        return out


def _json_text(body: str) -> str:
    """Из JSON-ответа вытащить читаемый текст (extract/description/summary)."""
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return body
    if isinstance(payload, dict):
        for key in ("extract", "description", "summary", "text", "content"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        if "results" in payload:
            return "\n".join(str(item.get("text", "")) for item in payload["results"]
                             if isinstance(item, dict))
    return body


def _html_title(html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    return html_to_text(match.group(1)) if match else ""
