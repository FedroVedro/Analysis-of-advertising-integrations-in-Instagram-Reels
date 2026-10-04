/* Skycoach Reels Analyzer — фронтенд без сборки.
 *
 * Состояние: список задач (job) с сервера + ID задач в localStorage, чтобы таблица
 * переживала перезагрузку. Пока есть незавершённые задачи — опрос /api/jobs каждые 3 с.
 * Режим ?mock=1 показывает все состояния на тестовых данных без бэкенда.
 */
(() => {
  'use strict';

  const MAX_URLS = 20;
  const POLL_MS = 3000;
  const MAX_STORED = 200;
  const IDS_PER_REQUEST = 100; // сервер принимает не больше 100 ID за запрос
  const STORAGE_IDS = 'ra.jobIds';
  const STORAGE_LEGEND = 'ra.legendOpen';
  const ACTIVE = ['queued', 'fetching', 'analyzing'];
  const PROBLEM = ['unavailable', 'invalid_url', 'failed'];
  const MOCK = new URLSearchParams(location.search).has('mock');

  const STATUS = {
    queued:      { label: 'В очереди',         cls: 'pill-neutral', pulse: true },
    fetching:    { label: 'Получаем данные',   cls: 'pill-blue',    spin: true },
    analyzing:   { label: 'Анализируем видео', cls: 'pill-blue',    spin: true },
    done:        { label: 'Готово',            cls: 'pill-green',   icon: 'ph-check' },
    unavailable: { label: 'Недоступен',        cls: 'pill-orange',  icon: 'ph-eye-slash' },
    invalid_url: { label: 'Неверная ссылка',   cls: 'pill-red',     icon: 'ph-link-break' },
    failed:      { label: 'Ошибка',            cls: 'pill-red',     icon: 'ph-warning-octagon' },
  };
  const CLASSES = [
    { label: 'Нет',        cls: 'pill-neutral' },
    { label: 'Упоминание', cls: 'pill-yellow' },
    { label: 'Реклама',    cls: 'pill-green' },
  ];
  const FILTERS = [
    { key: 'all', label: 'Все',        test: () => true },
    { key: '2',   label: 'Реклама',    test: r => r.status === 'done' && r.cls === 2 },
    { key: '1',   label: 'Упоминание', test: r => r.status === 'done' && r.cls === 1 },
    { key: '0',   label: 'Нет',        test: r => r.status === 'done' && r.cls === 0 },
    { key: 'p',   label: 'Проблемы',   test: r => PROBLEM.includes(r.status) },
  ];

  const NF = new Intl.NumberFormat('ru-RU');
  const DF = new Intl.DateTimeFormat('ru-RU', { day: 'numeric', month: 'short', year: 'numeric' });

  const state = {
    rows: [],               // модели строк, новые сверху
    expanded: new Set(),
    filter: 'all',
    submitting: false,
    pollTimer: null,
    lastRendered: '',
  };

  const $ = id => document.getElementById(id);
  const el = {
    urls: $('urls'), submit: $('submit'), counter: $('counter'), tooMany: $('too-many'), hint: $('hint'),
    legendToggle: $('legend-toggle'), legendBody: $('legend-body'), legendChevron: $('legend-chevron'),
    banner: $('banner'), bannerIcon: $('banner-icon'), bannerTitle: $('banner-title'), bannerBody: $('banner-body'), bannerClose: $('banner-close'),
    summary: $('summary'), polling: $('polling'), filters: $('filters'), csv: $('csv'), clear: $('clear'),
    empty: $('empty'), tableWrap: $('table-wrap'), rows: $('rows'), filteredEmpty: $('filtered-empty'),
    fillExample: $('fill-example'),
  };

  // ---------- утилиты ----------

  const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const fmtDate = s => (s ? DF.format(new Date(s)).replace(/\./g, '').replace(/\s?г$/, '') : '—');

  function plural(n, one, few, many) {
    const m10 = n % 10, m100 = n % 100;
    if (m10 === 1 && m100 !== 11) return one;
    if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few;
    return many;
  }

  const storage = {
    get(key, fallback) {
      try { const v = localStorage.getItem(key); return v == null ? fallback : JSON.parse(v); } catch { return fallback; }
    },
    set(key, value) {
      try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* приватный режим и т.п. */ }
    },
  };

  const inputLines = () => el.urls.value.split('\n').map(s => s.trim()).filter(Boolean);

  // ---------- модель ----------

  /** Ответ API (job) -> модель строки таблицы. */
  function toRow(job) {
    const reel = job.reel || {};
    return {
      id: job.job_id,
      url: reel.url || job.input_url,
      noReel: !job.reel,
      status: job.status,
      cached: !!job.cached,
      error: job.error,
      author: reel.author ?? null,
      published: reel.published_at ?? null,
      views: reel.views ?? null,
      likes: reel.likes ?? null,
      comments: reel.comments ?? null,
      cls: reel.integration_class ?? null,
      vis: reel.visibility_score ?? null,
      hasAudio: reel.has_audio ?? null,
      just: reel.justification ?? null,
      transcript: reel.transcript ?? null,
      caption: reel.caption ?? null,
    };
  }

  function mergeJobs(jobs) {
    const byId = new Map(jobs.map(j => [j.job_id, toRow(j)]));
    state.rows = state.rows.map(r => byId.get(r.id) || r);
  }

  const saveIds = () => storage.set(STORAGE_IDS, state.rows.map(r => r.id).slice(0, MAX_STORED));
  const activeIds = () => state.rows.filter(r => ACTIVE.includes(r.status)).map(r => r.id);

  // ---------- API ----------

  class ApiError extends Error {}

  async function api(path, options) {
    let resp;
    try {
      resp = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...options });
    } catch {
      throw new ApiError('network');
    }
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) {
      const detail = data.detail;
      const msg = typeof detail === 'string' ? detail
        : Array.isArray(detail) ? detail.map(d => d.msg).join('; ')
        : `Сервер ответил ошибкой ${resp.status}`;
      const err = new ApiError(msg);
      err.status = resp.status;
      throw err;
    }
    return data;
  }

  async function fetchJobs(ids) {
    const jobs = [];
    for (let i = 0; i < ids.length; i += IDS_PER_REQUEST) {
      const chunk = ids.slice(i, i + IDS_PER_REQUEST);
      const data = await api('/api/jobs?ids=' + encodeURIComponent(chunk.join(',')));
      jobs.push(...data.jobs);
    }
    return jobs;
  }

  // ---------- баннер ----------

  function showBanner(kind, body) {
    const network = kind === 'network';
    el.bannerIcon.className = 'ph banner-icon ' + (network ? 'ph-wifi-slash' : 'ph-warning-circle');
    el.bannerTitle.textContent = network ? 'Не удалось связаться с сервером' : 'Ссылки не отправлены';
    el.bannerBody.textContent = network ? 'Проверьте подключение — обновление статусов продолжится автоматически.' : body;
    el.banner.dataset.kind = kind;
    el.banner.hidden = false;
  }

  function hideBanner(kind) {
    if (!kind || el.banner.dataset.kind === kind) el.banner.hidden = true;
  }

  // ---------- действия ----------

  async function submit() {
    const urls = inputLines();
    if (!urls.length || urls.length > MAX_URLS || state.submitting) return;
    state.submitting = true;
    renderForm();
    try {
      const data = MOCK ? mockSubmit(urls) : await api('/api/jobs', { method: 'POST', body: JSON.stringify({ urls }) });
      const fresh = data.jobs.map(toRow);
      const freshIds = new Set(fresh.map(r => r.id));
      state.rows = [...fresh, ...state.rows.filter(r => !freshIds.has(r.id))];
      el.urls.value = '';
      hideBanner();
      saveIds();
      schedulePoll(true);
    } catch (e) {
      showBanner(e.message === 'network' ? 'network' : 'request', e.message);
    } finally {
      state.submitting = false;
      render();
    }
  }

  async function poll() {
    state.pollTimer = null;
    const ids = activeIds();
    if (!ids.length) return render();
    try {
      mergeJobs(MOCK ? mockPoll(ids) : await fetchJobs(ids));
      hideBanner('network');
    } catch (e) {
      if (e.message === 'network') showBanner('network');
    }
    render();
    schedulePoll();
  }

  function schedulePoll(soon = false) {
    if (state.pollTimer) clearTimeout(state.pollTimer);
    state.pollTimer = activeIds().length ? setTimeout(poll, soon ? 1000 : POLL_MS) : null;
  }

  async function loadStored() {
    if (MOCK) {
      state.rows = mockJobs().map(toRow);
      state.expanded.add(state.rows[0].id);
      return;
    }
    const ids = storage.get(STORAGE_IDS, []);
    if (!Array.isArray(ids) || !ids.length) return;
    try {
      const jobs = await fetchJobs(ids);
      const byId = new Map(jobs.map(j => [j.job_id, toRow(j)]));
      // Задачи, которых больше нет на сервере (например, БД пересоздана), выбрасываем
      state.rows = ids.filter(id => byId.has(id)).map(id => byId.get(id));
      saveIds();
    } catch (e) {
      if (e.message === 'network') showBanner('network');
    }
  }

  function toggleRow(id) {
    state.expanded.has(id) ? state.expanded.delete(id) : state.expanded.add(id);
    state.lastRendered = '';
    renderRows();
  }

  function clearRows() {
    if (!confirm('Убрать все строки из таблицы? Результаты на сервере сохранятся, повторная проверка ссылки вернёт их из кэша.')) return;
    state.rows = [];
    state.expanded.clear();
    saveIds();
    render();
  }

  function downloadCsv() {
    const head = ['Автор', 'Ссылка', 'Статус', 'Из кэша', 'Дата публикации', 'Просмотры', 'Лайки', 'Комментарии', 'Интеграция', 'Заметность', 'Обоснование', 'Ошибка'];
    const q = v => '"' + String(v ?? '').replace(/"/g, '""') + '"';
    const body = state.rows.map(r => [
      r.author ? '@' + r.author : '', r.url, STATUS[r.status]?.label ?? r.status, r.cached ? 'да' : '',
      r.published ? fmtDate(r.published) : '', r.views, r.likes, r.comments,
      r.status === 'done' && r.cls != null ? CLASSES[r.cls].label : '', r.vis, r.just, r.error,
    ].map(q).join(';'));
    // BOM + ";" — чтобы Excel с русской локалью открыл файл без мастера импорта
    const blob = new Blob(['﻿' + [head.map(q).join(';'), ...body].join('\n')], { type: 'text/csv;charset=utf-8' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'skycoach-reels-' + new Date().toISOString().slice(0, 10) + '.csv';
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  // ---------- отрисовка ----------

  function render() {
    renderForm();
    renderHead();
    renderRows();
  }

  function renderForm() {
    const n = inputLines().length, over = n > MAX_URLS;
    el.counter.textContent = `${n} / ${MAX_URLS}`;
    el.counter.classList.toggle('over', over);
    el.urls.classList.toggle('over', over);
    el.tooMany.hidden = !over;
    el.hint.hidden = over;
    el.submit.disabled = !n || over || state.submitting;
    el.submit.innerHTML = state.submitting
      ? '<i class="ph ph-circle-notch spin"></i><span>Отправляем…</span>'
      : '<i class="ph ph-magnifying-glass"></i><span>Проверить</span>';
  }

  function renderHead() {
    const rows = state.rows, n = rows.length;
    const active = activeIds().length, done = rows.filter(r => r.status === 'done').length;
    el.summary.textContent = n
      ? `${n} ${plural(n, 'ролик', 'ролика', 'роликов')} · готово ${done}` + (active ? ` · в работе ${active}` : '')
      : 'Пока пусто';
    el.polling.hidden = !active;
    el.csv.disabled = !n;
    el.clear.disabled = !n;
    el.empty.hidden = n > 0;
    el.tableWrap.hidden = n === 0;
    el.filters.innerHTML = FILTERS.map(f => {
      const count = f.key === 'all' ? '' : rows.filter(f.test).length;
      return `<label class="seg-opt"><input type="radio" name="ra-filter" value="${f.key}"${state.filter === f.key ? ' checked' : ''}>` +
        `<span>${f.label}</span><span class="seg-n">${count}</span></label>`;
    }).join('');
  }

  function renderRows() {
    const test = FILTERS.find(f => f.key === state.filter).test;
    const shown = state.rows.filter(test);
    el.filteredEmpty.hidden = !(state.rows.length && !shown.length);
    const html = shown.map(rowHtml).join('');
    // Не перерисовываем без изменений: не сбрасываем hover/фокус каждые 3 с
    if (html === state.lastRendered) return;
    state.lastRendered = html;
    el.rows.innerHTML = html;
  }

  function metricHtml(v, r) {
    if (v != null) return `<span class="metric">${NF.format(v)}</span>`;
    // «Скрыто автором» имеет смысл, только если ролик получен, а метрики нет
    const expected = r.author && !PROBLEM.includes(r.status);
    return expected
      ? '<span class="metric na hidden-by-author" title="Скрыто автором или недоступно">—</span>'
      : '<span class="metric na">—</span>';
  }

  function rowHtml(r) {
    const st = STATUS[r.status] || { label: r.status, cls: 'pill-neutral' };
    const open = state.expanded.has(r.id);
    const isDone = r.status === 'done';
    const code = (r.url.match(/reel\/([^/?#]+)/) || [])[1];
    const shortUrl = code ? 'reel/' + code : r.url;

    const authorHtml = r.author
      ? `<span class="author">@${esc(r.author)}</span>`
      : `<span class="author unknown">${r.noReel ? 'Ссылка не распознана' : 'Автор пока неизвестен'}</span>`;
    const linkHtml = r.noReel
      ? `<span class="raw-url">${esc(r.url)}</span>`
      : `<a class="reel-link" href="${esc(r.url)}" target="_blank" rel="noopener">${esc(shortUrl)}<i class="ph ph-arrow-up-right"></i></a>`;

    const badgeIcon = st.spin ? '<i class="ph ph-circle-notch spin"></i>'
      : st.pulse ? '<span class="pulse"></span>'
      : st.icon ? `<i class="ph ${st.icon}"></i>` : '';
    const cachedHtml = r.cached
      ? '<span class="cached" title="Ролик уже проверялся, показан сохранённый результат"><i class="ph ph-clock-counter-clockwise"></i>из кэша</span>'
      : '';

    const clsHtml = isDone && r.cls != null && CLASSES[r.cls]
      ? `<span class="pill ${CLASSES[r.cls].cls}"><span class="dot"></span>${CLASSES[r.cls].label}</span>`
      : '<span class="dash">—</span>';
    const visHtml = isDone && r.vis != null
      ? `<span class="vis" aria-label="Заметность ${r.vis} из 5"><span class="segs">${[1, 2, 3, 4, 5].map(i => `<span${i <= r.vis ? ' class="on"' : ''}></span>`).join('')}</span><span class="vis-n">${r.vis}</span></span>`
      : '<span class="dash">—</span>';

    const justFirst = r.just ? r.just.split('\n')[0]
      : r.error ? 'Анализ не выполнен'
      : ACTIVE.includes(r.status) ? 'Появится после анализа'
      : isDone ? 'Анализ ещё не подключён' : '—';
    const noAudio = r.hasAudio === false;

    return `<div class="rgroup${open ? ' open' : ''}" role="rowgroup">
  <div class="rrow" role="row" tabindex="0" data-id="${esc(r.id)}" aria-expanded="${open}">
    <div class="cell-reel">${authorHtml}${linkHtml}</div>
    <div class="cell-status">
      <div class="status-line"><span class="pill ${st.cls}">${badgeIcon}${st.label}</span>${cachedHtml}</div>
      ${r.error ? `<span class="status-err">${esc(r.error)}</span>` : ''}
    </div>
    <span class="date">${fmtDate(r.published)}</span>
    ${metricHtml(r.views, r)}${metricHtml(r.likes, r)}${metricHtml(r.comments, r)}
    <div>${clsHtml}</div>
    <div>${visHtml}</div>
    <div class="cell-just">
      <span class="just-first${r.just ? '' : ' empty-text'}">${esc(justFirst)}</span>
      ${noAudio ? '<i class="ph ph-speaker-slash no-audio-icon" title="Без звука"></i>' : ''}
      <i class="ph ${open ? 'ph-caret-up' : 'ph-caret-down'} row-chevron"></i>
    </div>
  </div>
  ${open ? detailsHtml(r, st, noAudio) : ''}
</div>`;
  }

  function detailsHtml(r, st, noAudio) {
    const errCls = st.cls === 'pill-orange' ? 'error-orange' : 'error-red';
    const parts = [];
    if (r.error) parts.push(`<div class="detail wide"><span class="detail-label">Ошибка</span><span class="detail-text ${errCls}">${esc(r.error)}</span></div>`);
    if (r.just) parts.push(`<div class="detail wide"><span class="detail-label">Обоснование</span><p class="detail-text lead" style="margin:0">${esc(r.just)}</p></div>`);
    parts.push(`<div class="detail"><span class="detail-label">Транскрипт</span>
      ${noAudio ? '<span class="pill pill-orange"><i class="ph ph-speaker-slash"></i>Без звука — анализ только по видео</span>' : ''}
      ${r.transcript ? `<span class="detail-text">${esc(r.transcript)}</span>` : noAudio ? '' : '<span class="detail-text muted">Нет транскрипта</span>'}
    </div>`);
    parts.push(`<div class="detail"><span class="detail-label">Подпись к ролику</span>
      <span class="detail-text${r.caption ? '' : ' muted'}">${esc(r.caption || 'Нет подписи')}</span></div>`);
    return `<div class="details">${parts.join('')}</div>`;
  }

  // ---------- события ----------

  el.urls.addEventListener('input', renderForm);
  el.urls.addEventListener('keydown', e => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) submit();
  });
  el.submit.addEventListener('click', submit);
  el.fillExample.addEventListener('click', () => {
    el.urls.value = 'https://www.instagram.com/reel/DceO7gsR0w-/\n';
    el.urls.focus();
    renderForm();
  });
  el.bannerClose.addEventListener('click', () => hideBanner());
  el.csv.addEventListener('click', downloadCsv);
  el.clear.addEventListener('click', clearRows);
  el.filters.addEventListener('change', e => {
    state.filter = e.target.value;
    renderRows();
  });
  el.rows.addEventListener('click', e => {
    if (e.target.closest('a')) return; // ссылка на ролик открывается, строка не раскрывается
    const row = e.target.closest('.rrow');
    if (row) toggleRow(row.dataset.id);
  });
  el.rows.addEventListener('keydown', e => {
    const row = e.target.closest('.rrow');
    if (row && (e.key === 'Enter' || e.key === ' ')) {
      e.preventDefault();
      toggleRow(row.dataset.id);
      el.rows.querySelector(`.rrow[data-id="${CSS.escape(row.dataset.id)}"]`)?.focus();
    }
  });

  function setLegend(open) {
    el.legendBody.hidden = !open;
    el.legendChevron.className = 'ph legend-chevron ' + (open ? 'ph-caret-up' : 'ph-caret-down');
    el.legendToggle.setAttribute('aria-expanded', String(open));
    storage.set(STORAGE_LEGEND, open);
  }
  el.legendToggle.addEventListener('click', () => setLegend(el.legendBody.hidden));
  setLegend(storage.get(STORAGE_LEGEND, true));

  // ---------- мок-режим (?mock=1) ----------

  function mockJobs() {
    const reel = (code, extra) => ({ url: `https://www.instagram.com/reel/${code}/`, ...extra });
    const job = (id, status, extra = {}, reelData = null, input) => ({
      job_id: id, status, cached: false, error: null, ...extra,
      input_url: input || (reelData && reelData.url), reel: reelData,
    });
    return [
      job('m1', 'done', {}, reel('DceO7gsR0w-', { author: 'valorant_funzone', published_at: '2026-08-25T18:01:57+00:00', views: 221738, likes: 10954, comments: 67, integration_class: 2, visibility_score: 4, has_audio: true,
        justification: '4/5. Логотип в кадре 9 с из 32 (28 %), до ≈12 % площади кадра. Голосовой призыв на 0:14. Промокод WOW20 в подписи.',
        transcript: '…и если хотите апнуть ранг без нервов — ребята из Skycoach реально помогли, ссылка в описании, промокод WOW20.',
        caption: 'They will cower! 🔥\nПромокод WOW20 — скидка на бустинг в Skycoach' })),
      job('m2', 'done', {}, reel('Cx81kQpLm2a', { author: 'dota_mid_or_feed', published_at: '2026-09-02T11:20:00+00:00', views: 48210, likes: null, comments: 212, integration_class: 1, visibility_score: 2, has_audio: true,
        justification: '2/5. Название Skycoach на футболке стримера, в кадре 4 с из 41. Призыва и промокода нет.', caption: 'Когда тиммейт снова пикнул Пуджа' })),
      job('m3', 'done', {}, reel('DaQ7s0vT11b', { author: 'wow_classic_daily', published_at: '2026-09-14T08:45:10+00:00', views: 15320, likes: 902, comments: 31, integration_class: 0, visibility_score: null, has_audio: true,
        justification: 'Skycoach не найден ни в кадре, ни в речи, ни в подписи.', transcript: 'Сегодня разбираем новый рейд, всё по таймингам…', caption: 'Гайд по первому боссу' })),
      job('m4', 'analyzing', {}, reel('DbK2yhR9uXw', { author: 'cs_clutchmoments', published_at: '2026-09-28T19:00:00+00:00', views: 9021, likes: 640, comments: 18 })),
      job('m5', 'queued', {}, reel('DdP0aa1Lq7Z', {})),
      job('m6', 'done', { cached: true }, reel('CzT5wNn4pQe', { author: 'lol_rankup', published_at: '2026-07-30T15:12:00+00:00', views: 1034502, likes: 88410, comments: 1204, integration_class: 2, visibility_score: 5, has_audio: true,
        justification: '5/5. Интеграция на весь ролик: логотип в углу 100 % времени, два голосовых призыва, промокод RANK15 в кадре и в подписи.',
        transcript: 'Skycoach — лучший способ поднять ранг, жми на ссылку в шапке!', caption: 'Реклама. Skycoach — промокод RANK15' })),
      job('m7', 'unavailable', { error: 'Ролик недоступен: приватный аккаунт, удалён или не существует' }, reel('DcX00privat', {})),
      job('m8', 'invalid_url', { error: 'Это не ссылка на Instagram Reel. Пример: https://www.instagram.com/reel/ABC123/' }, null, 'hello'),
      job('m9', 'failed', { error: 'Сервис получения данных недоступен, попробуйте позже (попыток: 3)' }, reel('Dd12FailX9q', {})),
      job('m10', 'done', {}, reel('DcM8nOsnd4r', { author: 'valorant_silentplays', published_at: '2026-09-20T21:30:00+00:00', views: 67345, likes: 4120, comments: 88, integration_class: 1, visibility_score: 1, has_audio: false,
        justification: '1/5. Логотип Skycoach мелькнул на 0:03 на 0,8 с. Звуковой дорожки нет — анализ только по кадрам и подписи.', caption: 'no comms needed' })),
    ];
  }

  const mockProgress = new Map();
  function mockSubmit(urls) {
    return {
      jobs: urls.map((u, i) => {
        const code = (u.match(/instagram\.com\/(?:[\w.]+\/)?(?:reels?|p|tv)\/([\w-]{5,})/i) || [])[1];
        const id = 'n' + Date.now() + i;
        if (!code) return { job_id: id, input_url: u, status: 'invalid_url', cached: false, reel: null, error: 'Это не ссылка на Instagram Reel. Пример: https://www.instagram.com/reel/ABC123/' };
        mockProgress.set(id, 0);
        return { job_id: id, input_url: u, status: 'queued', cached: false, error: null, reel: { url: `https://www.instagram.com/reel/${code}/` } };
      }),
    };
  }
  function mockPoll(ids) {
    return ids.filter(id => mockProgress.has(id)).map(id => {
      const step = mockProgress.get(id) + 1;
      mockProgress.set(id, step);
      const row = state.rows.find(r => r.id === id);
      const status = step < 2 ? 'fetching' : step < 4 ? 'analyzing' : 'done';
      const reel = { url: row.url };
      if (step >= 2) Object.assign(reel, { author: 'new_blogger', published_at: '2026-09-29T12:00:00+00:00', views: 30412, likes: 2210, comments: 54 });
      if (status === 'done') Object.assign(reel, { integration_class: 2, visibility_score: 3, has_audio: true,
        justification: '3/5. Логотип в кадре 6 с из 24 (25 %). Промокод в подписи, голосового призыва нет.',
        transcript: '…заходите в Skycoach, ссылка ниже.', caption: 'Промокод в описании' });
      return { job_id: id, input_url: row.url, status, cached: false, error: null, reel };
    });
  }

  // ---------- старт ----------

  (async () => {
    render();
    await loadStored();
    render();
    schedulePoll(true);
  })();
})();
