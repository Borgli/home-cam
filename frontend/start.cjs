const path = require('path');
process.chdir(__dirname);
// Extra arguments go to Vite (e.g. --port 5174 --strictPort); --backend <url> sets the
// backend the dev server proxies to (default http://localhost:5001, see vite.config.js)
const args = process.argv.slice(2);
const env = { ...process.env };
const backendArg = args.indexOf('--backend');
if (backendArg !== -1) {
  env.BACKEND_URL = args[backendArg + 1];
  args.splice(backendArg, 2);
}
const child = require('child_process').spawn(
  process.execPath,
  [path.join(__dirname, 'node_modules', 'vite', 'bin', 'vite.js'), '--host', ...args],
  { stdio: 'inherit', cwd: __dirname, env }
);
child.on('exit', (code) => process.exit(code || 0));
