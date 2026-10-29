(() => {
  const root = document.querySelector('.frame-explainer');
  if (!root) return;
  const button = root.querySelector('.frame-trigger');
  const panel = root.querySelector('.frame-popover');
  let pinned = false;
  function show() { panel.hidden = false; button.setAttribute('aria-expanded', 'true'); }
  function hide() { panel.hidden = true; button.setAttribute('aria-expanded', 'false'); pinned = false; }
  button.addEventListener('pointerenter', event => { if (event.pointerType !== 'touch') show(); });
  button.addEventListener('focus', show);
  button.addEventListener('click', () => { pinned = !pinned; if (pinned) show(); else hide(); });
  root.addEventListener('pointerleave', () => { if (!pinned && !root.contains(document.activeElement)) hide(); });
  root.addEventListener('focusout', event => { if (!root.contains(event.relatedTarget)) hide(); });
  document.addEventListener('pointerdown', event => { if (!root.contains(event.target)) hide(); });
  function dismiss() { if (panel.contains(document.activeElement)) button.focus(); hide(); }
  root.querySelector('.frame-close').addEventListener('click', dismiss);
  document.addEventListener('keydown', event => { if (event.key === 'Escape' && !panel.hidden) dismiss(); });
})();
