"""Server-owned transport policy for a self-hosted Ollama service.

Loopback HTTP supports the development SSH tunnel. Remote requests require an
explicit host, verified HTTPS, and either a client certificate or a gateway token.
This module never runs model-generated commands or accepts browser credentials.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import ssl
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, HTTPSHandler, ProxyHandler, build_opener


LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


@dataclass(frozen=True)
class OllamaConnection:
    remote_host: str | None = None
    ca_file: str | None = None
    client_cert: str | None = None
    client_key: str | None = None
    token_file: str | None = None

    @classmethod
    def from_environment(cls) -> "OllamaConnection":
        names = ("REMOTE_HOST", "CA_FILE", "CLIENT_CERT", "CLIENT_KEY", "API_KEY_FILE")
        return cls(*(os.getenv("LOCAL_LLM_" + name) or None for name in names))

    def endpoint_error(self, base_url: str | None) -> str | None:
        error = ("Use the local Ollama endpoint http://127.0.0.1:11434/v1 for an SSH tunnel, or configure "
                 "LOCAL_LLM_REMOTE_HOST and authenticated HTTPS for your own Ollama server.")
        try:
            parsed = urlparse(base_url or "")
            host, port = parsed.hostname, parsed.port
        except ValueError:
            return error
        if (not host or parsed.username is not None or parsed.password is not None
                or parsed.query or parsed.fragment or parsed.path.rstrip("/") != "/v1"):
            return error
        if parsed.scheme == "http" and host in LOOPBACK_HOSTS and port == 11434:
            if any((self.client_cert, self.client_key, self.token_file)):
                return "LLM authentication credentials must only be sent over verified HTTPS."
            return None
        if parsed.scheme != "https" or not self.remote_host:
            return error
        configured = self.remote_host.lower()
        # Exact server-owned host pinning, not a wildcard or a request-supplied URL.
        if configured != host or any(c in configured for c in "/:@?#\\ \t\n"):
            return "HTTPS endpoint must match the exact LOCAL_LLM_REMOTE_HOST."
        if configured == "ollama.com" or configured.endswith(".ollama.com"):
            return "Ollama cloud inference is disabled; configure your own server."
        if bool(self.client_cert) != bool(self.client_key):
            return "LOCAL_LLM_CLIENT_CERT and LOCAL_LLM_CLIENT_KEY must be configured together."
        if not self.client_cert and not self.token_file:
            return "Remote Ollama requires a client certificate or a gateway API key file."
        return None

    def ssl_context(self) -> ssl.SSLContext:
        """Always verify server identity; there is deliberately no insecure mode."""
        context = ssl.create_default_context(cafile=self.ca_file)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        if self.client_cert and self.client_key:
            context.load_cert_chain(self.client_cert, self.client_key)
        return context

    def headers(self) -> dict[str, str]:
        """Read a mounted gateway secret when needed, never expose it in health JSON."""
        if not self.token_file:
            return {}
        with Path(self.token_file).open("rb") as stream:
            content = stream.read(8193)
        if len(content) > 8192:
            raise ValueError("Invalid gateway secret file.")
        token = content.decode("ascii").strip()
        if not token or any(ord(char) <= 32 or ord(char) >= 127 for char in token):
            raise ValueError("Invalid gateway secret file.")
        return {"Authorization": "Bearer " + token}


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        raise HTTPError(request.full_url, code, "LLM redirects are disabled", headers, fp)


def open_request(request, timeout=None, *, context=None):
    """No redirects or ambient HTTP proxies: credentials stay at the pinned service."""
    handlers = [ProxyHandler({}), _NoRedirect()]
    if context is not None:
        handlers.append(HTTPSHandler(context=context))
    opener = build_opener(*handlers)
    return opener.open(request, timeout=timeout)
