import os, sys, time, socket, subprocess, webbrowser, signal
from contextlib import closing

DEFAULT_PORT = 8501
APP = ["streamlit", "run", "app.py", "--server.headless", "true", "--server.port"]
AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN", "").strip()

def find_free_port(start=DEFAULT_PORT, limit=20):
    for p in range(start, start+limit):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError("No free port found")

def main():
    port = find_free_port()
    print(f"[serve] starting Streamlit on port {port}…")
    app_proc = subprocess.Popen(APP + [str(port)])

    time.sleep(3)

    try:
        from pyngrok import ngrok, conf
    except Exception as e:
        print("[serve] pyngrok not installed. Run: pip install pyngrok")
        app_proc.terminate()
        sys.exit(1)

    if not AUTHTOKEN:
        print("[serve] NGROK_AUTHTOKEN not set. Set it and re-run.")
        app_proc.terminate()
        sys.exit(1)

    conf.get_default().auth_token = AUTHTOKEN

    print("[serve] creating ngrok tunnel…")
    tunnel = ngrok.connect(port, "http")
    url = tunnel.public_url
    print("\n==================== PUBLIC URL ====================")
    print(url)
    print("====================================================\n")

    with open("PUBLIC_URL.txt", "w", encoding="utf-8") as f:
        f.write(url + "\n")

    try:
        import qrcode
        img = qrcode.make(url)
        img.save("public_url_qr.png")
        print("[serve] saved QR: public_url_qr.png")
    except Exception as e:
        print(f"[serve] QR generation skipped: {e}")

    try:
        webbrowser.open(url, new=2)
    except Exception:
        pass

    print("[serve] press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
            if app_proc.poll() is not None:
                raise KeyboardInterrupt
    except KeyboardInterrupt:
        print("\n[serve] shutting down…")
    finally:
        try:
            ngrok.disconnect(tunnel.public_url)
            ngrok.kill()
        except Exception:
            pass
        try:
            if app_proc.poll() is None:
                if os.name == "nt":
                    app_proc.send_signal(signal.CTRL_BREAK_EVENT)
                    time.sleep(1)
                app_proc.terminate()
        except Exception:
            pass

if __name__ == "__main__":
    main()