"""Model controls, rejected inputs, and real HTTP streaming/cleanup boundaries."""
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import unittest
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thinking import default_effort, levels, responses_adapter, template_kwargs, translate


class ThinkingTests(unittest.TestCase):
    def test_all_profiles_match_native_controls(self):
        for path in (Path(__file__).resolve().parents[2] / 'config').glob('opencode-*.json'):
            config = json.loads(path.read_text())
            model = config['model'].split('/')[1]
            entry = config['provider']['pinnacles']['models'][model]
            enabled = {k: v for k, v in entry['variants'].items() if not v.get('disabled')}
            self.assertEqual(tuple(enabled), levels(model))
            for inherited in {'low', 'medium', 'high'} - set(levels(model)):
                self.assertTrue(entry['variants'][inherited]['disabled'])
            self.assertEqual(entry['options']['chat_template_kwargs'], template_kwargs(model, default_effort(model)))
            for effort, variant in enabled.items():
                request = {'model': model, 'input': 'hello', 'stream': True, 'reasoning': {'effort': effort}}
                adapted = translate(request, model)
                self.assertEqual(adapted['chat_template_kwargs'], variant['chat_template_kwargs'])
                self.assertEqual(adapted['input'], 'hello')
                self.assertTrue(adapted['stream'])
                self.assertEqual(request['reasoning']['effort'], effort)

    def test_super_low_mode_and_reset(self):
        self.assertEqual(levels('nemotron-3-super'), ('none', 'low', 'high'))
        self.assertEqual(template_kwargs('nemotron-3-super', 'low'),
                         {'enable_thinking': True, 'low_effort': True})
        self.assertEqual(template_kwargs('nemotron-3-super', 'high'),
                         {'enable_thinking': True, 'low_effort': False})
        self.assertEqual(template_kwargs('nemotron-3-super', 'none'),
                         {'enable_thinking': False, 'low_effort': False})
        with self.assertRaises(ValueError):
            template_kwargs('nemotron-3.5-lightning', 'low')

    def test_invalid_requests(self):
        for request in ([], {}, {'model': 'other'},
                        {'model': 'gemma4-31b', 'reasoning': 'high'},
                        {'model': 'gemma4-31b', 'reasoning': []},
                        {'model': 'gemma4-31b', 'reasoning': {'effort': ''}},
                        {'model': 'gemma4-31b', 'reasoning': {'effort': 'medium'}},
                        {'model': 'gemma4-31b', 'chat_template_kwargs': 'bad'}):
            with self.subTest(request=request), self.assertRaises(ValueError):
                translate(request, 'gemma4-31b')
        with self.assertRaises(ValueError):
            levels('unknown')

    def test_selected_effort_overrides_template_default(self):
        result = translate({'model': 'gemma4-31b', 'reasoning': {'effort': 'high'},
                            'chat_template_kwargs': {'enable_thinking': False, 'other': 1}}, 'gemma4-31b')
        self.assertEqual(result['chat_template_kwargs'], {'enable_thinking': True, 'other': 1})

    def test_exit_closes_an_inflight_upstream_stream(self):
        release = threading.Event()
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                self.wfile.write(b'data: first\n\n')
                self.wfile.flush()
                release.wait(10)
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        conn = None
        try:
            with responses_adapter(f'http://127.0.0.1:{server.server_port}/v1', 'gemma4-31b') as url:
                conn = http.client.HTTPConnection('127.0.0.1', urlsplit(url).port, timeout=3)
                conn.request('POST', '/v1/responses', json.dumps({'model': 'gemma4-31b'}))
                response = conn.getresponse()
                self.assertEqual(response.readline(), b'data: first\n')
                self.assertEqual(response.readline(), b'\n')
            # Exit must interrupt the upstream read, not wait for its 180s timeout.
            self.assertEqual(response.read(), b'')
            response.close()
        finally:
            release.set()
            if conn is not None:
                conn.close()
            server.shutdown()
            server.server_close()
            thread.join()

    def test_http_stream_errors_and_socket_cleanup(self):
        captured = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                captured.append((self.path, json.loads(self.rfile.read(int(self.headers['Content-Length'])))))
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                self.wfile.write(b'data: {"ok":true}\n\ndata: [DONE]\n\n')
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            with responses_adapter(f'http://127.0.0.1:{server.server_port}/v1', 'muse-glimmer-30b') as url:
                port = urlsplit(url).port
                for effort in levels('muse-glimmer-30b'):
                    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
                    try:
                        conn.request('POST', '/v1/responses', json.dumps({'model': 'muse-glimmer-30b', 'reasoning': {'effort': effort}}))
                        response = conn.getresponse()
                        self.assertEqual(response.status, 200)
                        self.assertEqual(response.read(), b'data: {"ok":true}\n\ndata: [DONE]\n\n')
                        self.assertEqual(captured[-1][0], '/v1/responses')
                        self.assertEqual(captured[-1][1]['chat_template_kwargs'], {'reasoning_strength': effort})
                    finally:
                        conn.close()
                for path, body, headers, status in (
                    ('/v1/responses', 'bad', {}, 400),
                    ('/v1/responses', '{}', {}, 400),
                    ('/v1/responses', '{}', {'Content-Encoding': 'gzip'}, 400),
                    ('/elsewhere', '{}', {}, 404),
                ):
                    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
                    try:
                        conn.request('POST', path, body, headers)
                        response = conn.getresponse()
                        self.assertEqual(response.status, status)
                        response.read()
                    finally:
                        conn.close()
            conn = http.client.HTTPConnection('127.0.0.1', port, timeout=1)
            try:
                with self.assertRaises(OSError):
                    conn.connect()
            finally:
                conn.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
