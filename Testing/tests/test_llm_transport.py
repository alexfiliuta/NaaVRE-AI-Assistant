"""Transport policy units plus real local HTTPS/mTLS connections; no cloud/model calls."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import ssl
from threading import Thread
from urllib.error import HTTPError, URLError
from urllib.request import Request

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
import pytest

from naavre_ai_assistant.llm_service import LLMCodeGenerator, LLMConfig, LLMServiceError
from naavre_ai_assistant.llm_transport import OllamaConnection, open_request


@pytest.mark.parametrize('url', [
    'http://nis07.lab.uvalight.net:11434/v1', 'https://another.example/v1',
    'https://nis07.lab.uvalight.net/v1?key=secret',
    'https://user:secret@nis07.lab.uvalight.net/v1',
    'https://nis07.lab.uvalight.net/v1#fragment', 'https://nis07.lab.uvalight.net/other',
])
def test_untrusted_endpoint_rejected(url):
    assert OllamaConnection(remote_host='nis07.lab.uvalight.net', token_file='secret').endpoint_error(url)


def test_remote_requires_explicit_host_and_authentication():
    assert OllamaConnection(token_file='secret').endpoint_error('https://localhost/v1')
    assert OllamaConnection(remote_host='localhost').endpoint_error('https://localhost/v1')
    assert OllamaConnection(remote_host='localhost', client_cert='cert').endpoint_error('https://localhost/v1')
    assert OllamaConnection(remote_host='localhost', client_cert='cert', client_key='key').endpoint_error('https://localhost/v1') is None
    assert OllamaConnection(remote_host='ollama.com', token_file='secret').endpoint_error('https://ollama.com/v1')


def test_loopback_cannot_receive_credentials():
    assert OllamaConnection().endpoint_error('http://127.0.0.1:11434/v1') is None
    assert OllamaConnection(token_file='secret').endpoint_error('http://127.0.0.1:11434/v1')


@pytest.mark.parametrize('content', [b'', b'x\nInjected: secret', b'x'*8193, b'\xff'],
                         ids=['empty', 'header-injection', 'oversize', 'non-ascii'])
def test_secret_file_validation(tmp_path, content):
    path = tmp_path/'token'
    path.write_bytes(content)
    with pytest.raises(ValueError):
        OllamaConnection(token_file=str(path)).headers()


def test_dotenv_is_not_implicitly_loaded(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('NAAVRE_ENV_FILE', raising=False)
    monkeypatch.setenv('LOCAL_LLM_MODEL', 'qwen3-coder:30b')
    monkeypatch.delenv('LOCAL_LLM_MODEL', raising=False)
    (tmp_path/'.env').write_text('LOCAL_LLM_MODEL=cloud:evil\n')
    assert LLMConfig.from_environment().model == 'qwen3-coder:30b'
    monkeypatch.setenv('NAAVRE_ENV_FILE', str(tmp_path/'.env'))
    assert LLMConfig.from_environment().configuration_error


@pytest.fixture
def certificates(tmp_path):
    now = datetime.now(timezone.utc)
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'Test-only CA')])
    ca = (x509.CertificateBuilder().subject_name(ca_name).issuer_name(ca_name)
          .public_key(ca_key.public_key()).serial_number(x509.random_serial_number())
          .not_valid_before(now-timedelta(minutes=1)).not_valid_after(now+timedelta(days=1))
          .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
          .sign(ca_key, hashes.SHA256()))
    result = {}
    ca_path = tmp_path/'ca.pem'
    ca_path.write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    result['ca'] = str(ca_path)
    for name, usage in [('server', ExtendedKeyUsageOID.SERVER_AUTH), ('client', ExtendedKeyUsageOID.CLIENT_AUTH)]:
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        cert = (x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)]))
                .issuer_name(ca_name).public_key(key.public_key()).serial_number(x509.random_serial_number())
                .not_valid_before(now-timedelta(minutes=1)).not_valid_after(now+timedelta(days=1))
                .add_extension(x509.ExtendedKeyUsage([usage]), critical=False)
                .add_extension(x509.SubjectAlternativeName([x509.DNSName('localhost')]), critical=False)
                .sign(ca_key, hashes.SHA256()))
        cert_path, key_path = tmp_path/(name+'.pem'), tmp_path/(name+'.key')
        cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                              serialization.NoEncryption()))
        result[name+'_cert'], result[name+'_key'] = str(cert_path), str(key_path)
    return result


@contextmanager
def gateway(certificates, *, mutual=False, redirect=False):
    records = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            records.append((self.path, self.headers.get('Authorization'), body, self.connection.getpeercert()))
            if redirect:
                self.send_response(307)
                self.send_header('Location', '/must-not-follow')
                self.end_headers()
            else:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"message":{"content":"transport verified"}}')
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificates['server_cert'], certificates['server_key'])
    if mutual:
        context.load_verify_locations(certificates['ca'])
        context.verify_mode = ssl.CERT_REQUIRED
    server.socket = context.wrap_socket(server.socket, server_side=True)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f'https://localhost:{server.server_port}/v1', records
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def service(url, connection):
    return LLMCodeGenerator(LLMConfig(url, 'qwen3-coder:30b', 5, 100, connection=connection))


def test_actual_https_gateway_token(certificates, tmp_path):
    secret = tmp_path/'token'
    secret.write_text('test-only-secret\n')
    connection = OllamaConnection(remote_host='localhost', ca_file=certificates['ca'], token_file=str(secret))
    with gateway(certificates) as (url, records):
        assert service(url, connection)._request_content('Test', system_message='Test', temperature=0) == 'transport verified'
    assert records[0][0] == '/api/chat'
    assert records[0][1] == 'Bearer test-only-secret'
    assert records[0][2]['model'] == 'qwen3-coder:30b'


def test_actual_mutual_tls(certificates):
    connection = OllamaConnection(remote_host='localhost', ca_file=certificates['ca'],
        client_cert=certificates['client_cert'], client_key=certificates['client_key'])
    with gateway(certificates, mutual=True) as (url, records):
        assert service(url, connection)._request_content('Test', system_message='Test', temperature=0) == 'transport verified'
    assert records[0][3]['subject'][0][0] == ('commonName', 'client')
    assert records[0][1] is None


def test_untrusted_certificate_fails_closed(certificates, tmp_path):
    secret = tmp_path/'token'
    secret.write_text('test-only-secret')
    connection = OllamaConnection(remote_host='localhost', token_file=str(secret))
    with gateway(certificates) as (url, records):
        with pytest.raises(LLMServiceError, match='TLS/authentication'):
            service(url, connection)._request_content('Test', system_message='Test', temperature=0)
    assert records == []


def test_hostname_mismatch_fails_closed(certificates):
    connection = OllamaConnection(ca_file=certificates['ca'])
    with gateway(certificates) as (url, records):
        with pytest.raises(URLError):
            open_request(Request(url.replace('localhost', '127.0.0.1')), timeout=5, context=connection.ssl_context())
    assert records == []


def test_missing_client_identity_is_rejected(certificates, tmp_path):
    secret = tmp_path/'token'
    secret.write_text('test-only-secret')
    connection = OllamaConnection(remote_host='localhost', ca_file=certificates['ca'], token_file=str(secret))
    with gateway(certificates, mutual=True) as (url, records):
        with pytest.raises(LLMServiceError):
            service(url, connection)._request_content('Test', system_message='Test', temperature=0)
    assert records == []


def test_redirect_not_followed_and_credentials_not_leaked(certificates, tmp_path):
    secret = tmp_path/'token'
    secret.write_text('test-only-secret')
    connection = OllamaConnection(remote_host='localhost', ca_file=certificates['ca'], token_file=str(secret))
    with gateway(certificates, redirect=True) as (url, records):
        with pytest.raises(LLMServiceError):
            service(url, connection)._request_content('Test', system_message='Test', temperature=0)
    assert len(records) == 1


def test_transport_error_does_not_expose_secret_or_file_path(tmp_path):
    path = tmp_path/'missing-private-token'
    connection = OllamaConnection(remote_host='localhost', token_file=str(path))
    with pytest.raises(LLMServiceError) as caught:
        service('https://localhost/v1', connection)._request_content('Test', system_message='Test', temperature=0)
    assert str(path) not in str(caught.value)
