FROM ubuntu:24.04

# 1bit.systems — single-binary inference server (zaya_server)
# Ships the staged release binaries; runtime deps only.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libstdc++6 libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

COPY build/stage/1bit-systems-*/usr/bin/ /usr/local/bin/
COPY build/stage/1bit-systems-*/usr/lib/1bit/ /usr/local/lib/1bit/
ENV LD_LIBRARY_PATH=/usr/local/lib/1bit

EXPOSE 8088

# zaya_server: OpenAI-compatible multi-backend inference server
ENTRYPOINT ["zaya_server"]
CMD ["--port", "8088"]
