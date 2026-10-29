(() => {
  const root = document.querySelector('.pipeline-preview');
  if (!root) return;
  const trigger = root.querySelector('.pipeline-trigger');
  const panel = root.querySelector('.pipeline-example');
  let dismissed = false;
  function show() {
    panel.hidden = false;
    trigger.setAttribute('aria-expanded', 'true');
  }
  function hide() {
    panel.hidden = true;
    trigger.setAttribute('aria-expanded', 'false');
  }
  trigger.addEventListener('pointerenter', event => {
    if (event.pointerType !== 'touch' && !dismissed) show();
  });
  trigger.addEventListener('focus', show);
  trigger.addEventListener('click', show);
  root.addEventListener('pointerleave', event => {
    dismissed = false;
    if (event.pointerType !== 'touch' && !root.contains(document.activeElement)) hide();
  });
  root.addEventListener('focusout', event => {
    if (!root.contains(event.relatedTarget)) hide();
  });
  document.addEventListener('pointerdown', event => {
    if (!root.contains(event.target)) hide();
  });
  function dismiss() {
    if (panel.contains(document.activeElement)) trigger.focus();
    hide();
    dismissed = true;
  }
  root.querySelector('.pipeline-close').addEventListener('click', dismiss);
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && !panel.hidden) dismiss();
  });
})();
