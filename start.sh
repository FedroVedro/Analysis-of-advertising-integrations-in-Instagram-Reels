#!/usr/bin/env bash
# Web и воркер в одном контейнере: файл SQLite на volume должен быть общим.
# Если любой из процессов умер — гасим второй и выходим с ошибкой, платформа перезапустит контейнер.

python -m app.worker &
uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" \
  --proxy-headers --forwarded-allow-ips='*' &

trap 'kill -TERM $(jobs -p) 2>/dev/null' TERM INT

wait -n
kill -TERM $(jobs -p) 2>/dev/null
wait
exit 1
