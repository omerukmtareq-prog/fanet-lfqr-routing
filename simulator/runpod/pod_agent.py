"""Minimal control agent for a RunPod CPU pod (stdlib only).

Started by the pod's start command. Listens on :8080 (exposed through the
RunPod HTTPS proxy as https://<pod_id>-8080.proxy.runpod.net). Every endpoint
except /health requires the header X-Token equal to $AGENT_TOKEN.

  GET  /health                         -> "ok"
  POST /upload?dest=<rel dir>          body = .tar.gz, extracted under /workspace/<dest>
  POST /exec   {"id": "...", "cmd": "..."}  starts `bash -lc cmd` in the background (one per id)
  GET  /status?id=<id>                 -> {running, returncode, started, log_tail, done_files}
  GET  /download?path=<rel dir>        -> .tar.gz of /workspace/<path>
Nothing here changes experiment code or results; it only moves files and
starts the declared job runner.
"""
import io, json, os, subprocess, tarfile, time, urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.environ.get("WORKSPACE", "/workspace")
TOKEN = os.environ.get("AGENT_TOKEN", "")
PROCS = {}


def safe(rel):
    p = os.path.realpath(os.path.join(ROOT, rel))
    if not (p == ROOT or p.startswith(ROOT + os.sep)):
        raise ValueError("path outside workspace")
    return p


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def _auth(self):
        if not TOKEN or self.headers.get("X-Token") != TOKEN:
            self._send(403, {"error": "forbidden"}); return False
        return True

    def do_GET(self):
        u = urllib.parse.urlparse(self.path); q = dict(urllib.parse.parse_qsl(u.query))
        if u.path == "/health":
            return self._send(200, "ok", "text/plain")
        if not self._auth():
            return
        try:
            if u.path == "/status":
                pid = q.get("id", "")
                info = PROCS.get(pid)
                log = os.path.join(ROOT, "logs", pid + ".log")
                tail = ""
                if os.path.exists(log):
                    with open(log, "rb") as f:
                        f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 6000)); tail = f.read().decode(errors="replace")
                done = sum(1 for dp, _, fs in os.walk(ROOT) if "DONE" in fs)
                res = {"id": pid, "known": info is not None, "log_tail": tail, "done_files": done,
                       "load": os.getloadavg(), "cpus": os.cpu_count(), "time": time.time()}
                if info:
                    rc = info["p"].poll()
                    res.update(running=rc is None, returncode=rc, started=info["started"])
                return self._send(200, res)
            if u.path == "/download":
                src = safe(q.get("path", ""))
                buf = io.BytesIO()
                with tarfile.open(fileobj=buf, mode="w:gz") as tf:
                    tf.add(src, arcname=os.path.relpath(src, ROOT))
                return self._send(200, buf.getvalue(), "application/gzip")
            self._send(404, {"error": "unknown"})
        except Exception as e:
            self._send(500, {"error": repr(e)})

    def do_POST(self):
        u = urllib.parse.urlparse(self.path); q = dict(urllib.parse.parse_qsl(u.query))
        if not self._auth():
            return
        n = int(self.headers.get("Content-Length", "0")); data = self.rfile.read(n)
        try:
            if u.path == "/upload":
                dest = safe(q.get("dest", "upload")); os.makedirs(dest, exist_ok=True)
                with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
                    for m in tf.getmembers():
                        safe(os.path.relpath(os.path.join(dest, m.name), ROOT))
                    tf.extractall(dest)
                return self._send(200, {"ok": True, "dest": dest, "bytes": n})
            if u.path == "/exec":
                req = json.loads(data)
                pid = req["id"]
                if pid in PROCS and PROCS[pid]["p"].poll() is None:
                    return self._send(409, {"error": "already running", "id": pid})
                os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
                log = open(os.path.join(ROOT, "logs", pid + ".log"), "ab")
                p = subprocess.Popen(["bash", "-lc", req["cmd"]], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                PROCS[pid] = {"p": p, "started": time.time()}
                return self._send(200, {"ok": True, "id": pid, "pid": p.pid})
            self._send(404, {"error": "unknown"})
        except Exception as e:
            self._send(500, {"error": repr(e)})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    os.makedirs(ROOT, exist_ok=True)
    ThreadingHTTPServer(("0.0.0.0", 8080), H).serve_forever()
