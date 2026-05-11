function getCurrentTheme() {
  return localStorage.getItem('theme') || 'dark';
}

function setTheme(theme) {
  const link = document.getElementById('theme-stylesheet');
  if (theme === 'light') {
    link.href = "/static/css/style_white.css";
  } else {
    link.href = "/static/css/styles.css";
  }
  localStorage.setItem('theme', theme);
}

function toggleTheme() {
  const current = getCurrentTheme();
  const newTheme = current === 'dark' ? 'light' : 'dark';
  setTheme(newTheme);
}

document.addEventListener('DOMContentLoaded', () => {
  setTheme(getCurrentTheme());
});