"""
Entry point for Second Eye Web Server (FastAPI + HTTPS/HTTP).
Supports auto-port binding, self-signed SSL certificate generation for mobile camera access,
and clear LAN URL announcement.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def get_local_ip() -> str:
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '127.0.0.1'

def ensure_ssl_certs():
    cert_path = PROJECT_ROOT / "cert.pem"
    key_path = PROJECT_ROOT / "key.pem"
    if not (cert_path.exists() and key_path.exists()):
        import subprocess
        try:
            print("[SSL] Đang tạo chứng chỉ SSL bảo mật cho kết nối Camera di động...")
            subprocess.run([
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", str(key_path), "-out", str(cert_path),
                "-days", "365", "-nodes", "-subj", "/CN=SecondEye"
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("[SSL] Đã tạo chứng chỉ SSL cert.pem & key.pem thành công.")
        except Exception as e:
            print(f"[SSL Error] Không thể tạo chứng chỉ tự động: {e}")

def find_free_port(start_port: int = 8000) -> int:
    import socket
    port = start_port
    while port < start_port + 50:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('0.0.0.0', port)) != 0:
                return port
        port += 1
    return start_port

def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Second Eye Web Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    parser.add_argument("--auto-port", action="store_true", help="Auto-select free port if occupied")
    parser.add_argument("--ssl", action="store_true", default=True, help="Enable HTTPS (Required for phone camera)")
    parser.add_argument("--no-ssl", action="store_false", dest="ssl", help="Disable HTTPS and run plain HTTP")
    parser.add_argument("--reload", action="store_true", default=True, help="Enable auto-reload")
    args = parser.parse_args()

    target_port = args.port
    if args.auto_port:
        target_port = find_free_port(args.port)

    local_ip = get_local_ip()
    use_ssl = args.ssl
    cert_path = PROJECT_ROOT / "cert.pem"
    key_path = PROJECT_ROOT / "key.pem"

    if use_ssl:
        ensure_ssl_certs()
        if not (cert_path.exists() and key_path.exists()):
            print("[Warning] Không tìm thấy cert.pem / key.pem, chuyển về chế độ HTTP thông thường.")
            use_ssl = False

    proto = "https" if use_ssl else "http"
    print("\n" + "=" * 70)
    print("      SECOND EYE - HỆ THỐNG TRỢ LÝ THỊ GIÁC CHO NGƯỜI KHIẾM THỊ")
    print("=" * 70)
    print(f"💻 MÁY TÍNH:   {proto}://localhost:{target_port}")
    print(f"📱 ĐIỆN THOẠI: {proto}://{local_ip}:{target_port}")
    print("-" * 70)
    if use_ssl:
        print("📌 LƯU Ý KHI MỞ TRÊN ĐIỆN THOẠI (iOS Safari / Android Chrome):")
        print("   Vì dùng chứng chỉ bảo mật nội bộ, trình duyệt sẽ hỏi xác nhận 1 lần:")
        print("   -> Bấm 'Nâng cao' (Advanced) -> Chọn 'Tiếp tục truy cập' (Proceed).")
        print("   -> Sau đó bấm 'Bật Camera' để cấp quyền sử dụng camera điện thoại.")
    print("=" * 70 + "\n")

    if use_ssl:
        uvicorn.run("src.web.app:app", host=args.host, port=target_port,
                    ssl_keyfile=str(key_path), ssl_certfile=str(cert_path), reload=args.reload)
    else:
        uvicorn.run("src.web.app:app", host=args.host, port=target_port, reload=args.reload)

if __name__ == "__main__":
    main()
