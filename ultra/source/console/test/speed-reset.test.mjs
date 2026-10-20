import test from 'node:test';
import assert from 'node:assert/strict';
import { speedResetScript } from '../original-tests.mjs';

function run(engineUrl, saved, key) {
  const store = new Map();
  if (saved !== undefined) store.set('llmPerfTestApiUrl', saved);
  if (key !== undefined) store.set('sm75-speed-proxy-v2', key);
  global.localStorage = {
    getItem: (name) => (store.has(name) ? store.get(name) : null),
    setItem: (name, value) => store.set(name, String(value)),
    removeItem: (name) => store.delete(name),
  };
  const html = speedResetScript(engineUrl);
  const code = html.replace(/^<script[^>]*>/, '').replace(/<\/script>$/, '');
  eval(code);
  return store;
}

test('clears a stale 40xx API URL even when the engine key is current', () => {
  const store = run(
    'http://172.17.100.10:40046',
    'http://172.17.100.10:40062/v1/chat/completions',
    'http://172.17.100.10:40046',
  );
  assert.equal(store.has('llmPerfTestApiUrl'), false);
  assert.equal(store.get('sm75-speed-proxy-v2'), 'http://172.17.100.10:40046');
});

test('keeps a custom non-40xx API URL when the engine key is current', () => {
  const store = run(
    'http://172.17.100.10:40046',
    'http://172.17.100.10:8180/v1/chat/completions',
    'http://172.17.100.10:40046',
  );
  assert.equal(store.get('llmPerfTestApiUrl'), 'http://172.17.100.10:8180/v1/chat/completions');
});

test('clears the saved URL when the engine URL changes', () => {
  const store = run(
    'http://172.17.100.10:40046',
    'http://172.17.100.10:8180/v1/chat/completions',
    'http://172.17.100.10:40009',
  );
  assert.equal(store.has('llmPerfTestApiUrl'), false);
});

test('clears a saved 40xx URL on a different host', () => {
  const store = run(
    'http://172.17.100.10:40046',
    'http://127.0.0.1:40062/v1/chat/completions',
    'http://172.17.100.10:40046',
  );
  assert.equal(store.has('llmPerfTestApiUrl'), false);
});

test('escapes script-closing text in the injected engine URL', () => {
  const html = speedResetScript('http://172.17.100.10:40046/</script>');
  assert.equal(html.split('</script>').length - 1, 1);
});
