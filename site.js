document.addEventListener('DOMContentLoaded', () => {
  const { pathname, search, hash } = window.location;
  if (pathname.endsWith('.html')) {
    const cleanPath = pathname.replace(/\/index\.html$/, '/').replace(/\.html$/, '/');
    window.history.replaceState(null, '', `${cleanPath}${search}${hash}`);
  }

  const nav = document.querySelector('.navbar');
  const toggle = document.querySelector('.nav-toggle');

  if (nav && toggle) {
    toggle.addEventListener('click', () => {
      const isOpen = nav.classList.toggle('is-open');
      toggle.setAttribute('aria-expanded', String(isOpen));
    });

    nav.querySelectorAll('.nav-links a').forEach((link) => {
      link.addEventListener('click', () => {
        if (window.innerWidth <= 768) {
          nav.classList.remove('is-open');
          toggle.setAttribute('aria-expanded', 'false');
        }
      });
    });
  }

  const year = document.querySelector('[data-current-year]');
  if (year) {
    year.textContent = new Date().getFullYear();
  }
});
