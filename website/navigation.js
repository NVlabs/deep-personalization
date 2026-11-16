// Keep the contents aligned with the section at the top of the reading area.
(() => {
  const contentsToggle = document.querySelector('.contents-toggle');
  const sidebar = document.querySelector('.sidebar');
  function closeContents(returnFocus = false) {
    sidebar?.classList.remove('is-open');
    contentsToggle?.setAttribute('aria-expanded', 'false');
    if (returnFocus) contentsToggle?.focus();
  }
  contentsToggle?.addEventListener('click', () => {
    const open = contentsToggle.getAttribute('aria-expanded') !== 'true';
    contentsToggle.setAttribute('aria-expanded', String(open));
    sidebar?.classList.toggle('is-open', open);
  });
  document.addEventListener('click', event => {
    if (!sidebar?.contains(event.target) && !contentsToggle?.contains(event.target)) closeContents();
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && contentsToggle?.getAttribute('aria-expanded') === 'true') closeContents(true);
  });
  sidebar?.querySelectorAll('a').forEach(link => link.addEventListener('click', () => closeContents(true)));
  window.matchMedia('(max-width: 760px)').addEventListener('change', () => closeContents());
  const exploreLink = document.querySelector('.hero-link');
  const article = document.getElementById('article');
  exploreLink?.addEventListener('click', event => {
    // Preserve normal browser behavior for modified clicks.
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || !article) return;
    event.preventDefault();
    const offset = parseFloat(getComputedStyle(document.documentElement).scrollPaddingTop) || 0;
    if (window.location.hash !== '#article') {
      history.pushState(null, '', '#article');
    }
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    window.scrollTo({
      top: window.scrollY + article.getBoundingClientRect().top - offset,
      behavior: reducedMotion ? 'instant' : 'smooth',
    });
  });
  const entries = Array.from(document.querySelectorAll('.contents a[href^="#"]'))
    .map(link => ({ link, section: document.getElementById(link.hash.slice(1)) }))
    .filter(entry => entry.section);
  if (!entries.length) return;

  let scheduled = false;
  let previousScrollY = window.scrollY;
  function updateCurrentSection() {
    scheduled = false;
    const scrollingUp = window.scrollY < previousScrollY;
    previousScrollY = window.scrollY;
    // Update the URL without triggering another anchor navigation.
    if (article && scrollingUp && window.location.hash === '#article' &&
        article.getBoundingClientRect().top >= window.innerHeight * 0.5) {
      history.replaceState(history.state, '', window.location.pathname + window.location.search);
    }
    const readingLine = Math.min(160, window.innerHeight * 0.25);
    let current = null;
    for (const entry of entries) {
      if (entry.section.getBoundingClientRect().top <= readingLine) current = entry;
    }
    // The final section may be too short to reach the reading line.
    if (window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 2) {
      current = entries[entries.length - 1];
    }
    for (const entry of entries) {
      if (entry === current) entry.link.setAttribute('aria-current', 'location');
      else entry.link.removeAttribute('aria-current');
    }
  }
  function scheduleUpdate() {
    if (scheduled) return;
    scheduled = true;
    window.requestAnimationFrame(updateCurrentSection);
  }
  window.addEventListener('scroll', scheduleUpdate, { passive: true });
  window.addEventListener('resize', scheduleUpdate);
  window.addEventListener('pageshow', scheduleUpdate);
  window.addEventListener('load', scheduleUpdate);
  // Figures and web fonts can move section boundaries after the page loads.
  if ('ResizeObserver' in window) {
    new ResizeObserver(scheduleUpdate).observe(document.querySelector('.blog-content'));
  }
  updateCurrentSection();
})();
