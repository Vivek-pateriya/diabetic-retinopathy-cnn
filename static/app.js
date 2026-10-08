// Copyright (c) 2026 Vivek Pateriya
const themeButton = document.getElementById('theme-toggle');
function updateThemeButton() {
  const dark = document.documentElement.dataset.theme === 'dark';
  themeButton.textContent = dark ? 'Light mode' : 'Dark mode';
  themeButton.setAttribute('aria-pressed', String(dark));
  themeButton.setAttribute('aria-label', dark ? 'Turn dark mode off' : 'Turn dark mode on');
}
themeButton.addEventListener('click', () => {
  const theme = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem('retinacheck-theme', theme); } catch (_) {}
  updateThemeButton();
});
updateThemeButton();
const input = document.getElementById('image-input');
const preview = document.getElementById('preview');
const status = document.getElementById('status');
let previewUrl;
input.addEventListener('change', () => {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  const file = input.files[0];
  document.getElementById('filename').textContent = file ? file.name : 'No image selected';
  document.getElementById('result').hidden = true;
  document.getElementById('empty-result').hidden = false;
  preview.hidden = !file;
  if (file) { previewUrl = URL.createObjectURL(file); preview.src = previewUrl; }
  status.textContent = '';
});
document.getElementById('upload-form').addEventListener('submit', async event => {
  event.preventDefault();
  const file = input.files[0];
  if (!file) return;
  status.className = '';
  if (file.size > 10 * 1024 * 1024) { status.textContent = 'Choose an image smaller than 10 MB.'; status.className = 'error'; return; }
  const button = document.getElementById('analyse');
  button.disabled = true; input.disabled = true;
  document.getElementById('result').hidden = true;
  status.textContent = 'Analysing your image…';
  const form = new FormData(); form.append('image', file);
  try {
    const response = await fetch('/predict', { method: 'POST', body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Analysis failed. Please try again.');
    document.getElementById('grade').textContent = data.label;
    const scores = document.getElementById('scores'); scores.replaceChildren();
    for (const [label, probability] of Object.entries(data.probabilities)) {
      const percent = Math.max(0, Math.min(100, probability * 100));
      const row = document.createElement('div'); row.className = 'score';
      const caption = document.createElement('div'); caption.className = 'score-label';
      const name = document.createElement('span'); name.textContent = label;
      const value = document.createElement('span'); value.textContent = percent.toFixed(1) + '%';
      caption.append(name, value);
      const track = document.createElement('div'); track.className = 'track';
      const fill = document.createElement('div'); fill.className = 'fill'; fill.style.width = percent + '%';
      track.append(fill); row.append(caption, track); scores.append(row);
    }
    document.getElementById('empty-result').hidden = true;
    document.getElementById('result').hidden = false;
    status.textContent = 'Analysis complete.';
  } catch (error) { status.textContent = error.message; status.className = 'error'; }
  finally { button.disabled = false; input.disabled = false; }
});
