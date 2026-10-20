import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');

test('session response does not expose engine URL', () => {
  const server = fs.readFileSync(path.join(root, 'server.mjs'), 'utf8');
  const block = server.match(/u\.pathname === "\/api\/session"[\s\S]*?return json\(res, \{[\s\S]*?\}\);/);
  assert.ok(block, 'session route not found');
  assert.equal(/engineUrl/.test(block[0]), false);
  const app = fs.readFileSync(path.join(root, 'public', 'app.js'), 'utf8');
  assert.equal(/engineEndpoint/.test(app), false);
  assert.equal(/iframeBust/.test(app), true);
});
