const path = require('path');
process.chdir(__dirname);
const child = require('child_process').spawn(
  process.execPath,
  [path.join(__dirname, 'node_modules', 'vite', 'bin', 'vite.js'), '--host'],
  { stdio: 'inherit', cwd: __dirname }
);
child.on('exit', (code) => process.exit(code || 0));
