# Extends an existing administrator-selected NaaVRE single-user image.
# Does NOT start a second Jupyter server, MCP container or Ollama service.
ARG NAAVRE_BASE_IMAGE
FROM ${NAAVRE_BASE_IMAGE}
COPY Releases/naavre_ai_assistant-0.2.2-py3-none-any.whl /tmp/naavre_ai_assistant-0.2.2-py3-none-any.whl
RUN python -m pip install --no-cache-dir /tmp/naavre_ai_assistant-0.2.2-py3-none-any.whl
# Keep the base image's USER, ENTRYPOINT and CMD. Mount lab files and TLS secrets
# and inject environment configuration through JupyterHub, not image layers.
