(() => {
  const root = document.querySelector('[data-interactive-overview]');
  if (!root) return;
  const controls = [...root.querySelectorAll('.overview-hotspot')];
  const overlay = root.querySelector('.overview-overlay');
  let active = null;

  function hide() {
    overlay.hidden = true;
    controls.forEach(button => button.setAttribute('aria-expanded', 'false'));
    active = null;
  }
  function show(button) {
    active = button;
    controls.forEach(control => control.setAttribute('aria-expanded', String(control === button)));
    root.querySelector('.overview-detail-title').textContent = button.dataset.title;
    root.querySelector('.overview-detail-label').textContent = button.dataset.label;
    const copy = root.querySelector('.overview-detail-copy');
    if (button.dataset.part === 'memory') {
      copy.replaceChildren(root.querySelector('#overview-memory-copy').content.cloneNode(true));
    } else {
      copy.textContent = button.dataset.detail;
    }
    // Keep the card away from the component being explored when space permits.
    overlay.dataset.side = button.dataset.part === 'context' ? 'right' : 'left';
    overlay.hidden = false;
    overlay.scrollTop = 0;
  }
  controls.forEach(button => {
    button.addEventListener('pointerenter', event => {
      if (event.pointerType !== 'touch') show(button);
    });
    button.addEventListener('focus', () => show(button));
    button.addEventListener('click', () => show(button));
  });
  root.addEventListener('pointerleave', event => {
    if (event.pointerType !== 'touch' && !root.contains(document.activeElement)) hide();
  });
  root.addEventListener('focusout', event => {
    if (!root.contains(event.relatedTarget)) hide();
  });
  document.addEventListener('pointerdown', event => {
    if (!root.contains(event.target)) hide();
  });
  function dismiss() {
    // Return focus before hiding so the focus handler cannot reopen the card.
    if (overlay.contains(document.activeElement) && active) active.focus();
    hide();
  }
  root.querySelector('.overview-close').addEventListener('click', dismiss);
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && !overlay.hidden) dismiss();
  });
  root.querySelector('.overview-hotspots').hidden = false;
})();
