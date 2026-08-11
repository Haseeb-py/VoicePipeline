const els = {
  start: document.querySelector('#start-call'),
  end: document.querySelector('#end-call'),
  enableAudio: document.querySelector('#enable-audio'),
  persona: document.querySelector('#persona-select'),
  microphone: document.querySelector('#microphone-select'),
  micMeter: document.querySelector('#mic-meter-bar'),
  micLevel: document.querySelector('#mic-level-label'),
  connection: document.querySelector('#connection-status'),
  listening: document.querySelector('#listening-status'),
  speaking: document.querySelector('#speaking-status'),
  interim: document.querySelector('#interim-transcript'),
  final: document.querySelector('#final-transcript'),
  confidence: document.querySelector('#confidence strong'),
  response: document.querySelector('#ai-response'),
  responseTurn: document.querySelector('#response-turn'),
  audioState: document.querySelector('#audio-state'),
  audioProgress: document.querySelector('#audio-progress-bar'),
  latency: document.querySelector('#latency-metrics'),
  latencyTurn: document.querySelector('#latency-turn'),
  log: document.querySelector('#event-log'),
  clearLog: document.querySelector('#clear-log'),
  room: document.querySelector('#room-label'),
  remoteAudio: document.querySelector('#remote-audio'),
  toast: document.querySelector('#toast'),
};

let room = null;
let eventSource = null;
let toastTimer = null;
let browserReady = false;
let callStarting = false;
let microphoneTesting = false;
let audioContext = null;
let meterSource = null;
let meterAnimation = null;
let audibleInputResolve = null;
let audibleInputTimer = null;
let audibleFrames = 0;

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
  });
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try { message = (await response.json()).detail || message; } catch (_) { /* no body */ }
    throw new Error(message);
  }
  return response.json();
}

function pretty(value) {
  return String(value || '').replaceAll('_', ' ');
}

function setState(element, state) {
  element.dataset.state = state;
  element.textContent = pretty(state);
}

function applySession(data) {
  const waitingForBrowser = Boolean(data.running) && (!browserReady || callStarting);
  setState(els.connection, waitingForBrowser ? 'connecting' : (data.connection || 'disconnected'));
  if (microphoneTesting) {
    els.listening.dataset.state = 'processing';
    els.listening.textContent = 'Testing microphone';
  } else {
    setState(els.listening, waitingForBrowser ? 'idle' : (data.listening || 'idle'));
  }
  setState(els.speaking, data.speaking || 'silent');
  els.start.disabled = Boolean(data.running);
  els.end.disabled = !data.running;
  els.room.textContent = `Room: ${data.room_name || '—'}`;
  if (data.persona?.id && els.persona.value !== data.persona.id) els.persona.value = data.persona.id;
}

function addLog(type, detail) {
  const row = document.createElement('li');
  row.className = 'log-row';
  const time = document.createElement('span');
  time.className = 'log-time';
  time.textContent = new Date().toLocaleTimeString([], { hour12: false });
  const label = document.createElement('span');
  label.className = 'log-type';
  label.textContent = type;
  const copy = document.createElement('span');
  copy.className = 'log-detail';
  copy.textContent = detail;
  row.append(time, label, copy);
  els.log.append(row);
  while (els.log.children.length > 180) els.log.firstElementChild.remove();
  els.log.scrollTop = els.log.scrollHeight;
}

function showToast(message) {
  els.toast.textContent = message;
  els.toast.classList.add('visible');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => els.toast.classList.remove('visible'), 4500);
}

function renderLatency(data) {
  els.latencyTurn.textContent = `Turn ${data.turn_id}`;
  els.latency.replaceChildren();
  for (const [name, value] of Object.entries(data.metrics_ms || {})) {
    const metric = document.createElement('div');
    metric.className = 'metric';
    const label = document.createElement('span');
    label.className = 'metric-name';
    label.textContent = pretty(name);
    const number = document.createElement('strong');
    number.className = 'metric-value';
    number.textContent = `${Number(value).toFixed(1)} ms`;
    metric.append(label, number);
    els.latency.append(metric);
  }
}

function listenForPipelineEvents() {
  eventSource = new EventSource('/api/events');
  eventSource.addEventListener('session', ({ data }) => applySession(JSON.parse(data)));
  eventSource.addEventListener('lifecycle', ({ data }) => {
    const event = JSON.parse(data);
    addLog('lifecycle', event.state);
  });
  eventSource.addEventListener('persona', ({ data }) => {
    const event = JSON.parse(data);
    addLog('persona', `${event.name} · ${event.voice} @ ${event.speech_rate}x`);
  });
  eventSource.addEventListener('transcript', ({ data }) => {
    const event = JSON.parse(data);
    if (event.is_final) {
      els.final.textContent = event.text;
      els.interim.textContent = 'Listening for the next turn…';
      els.confidence.textContent = `${(Number(event.confidence) * 100).toFixed(1)}%`;
    } else {
      els.interim.textContent = event.text;
    }
    addLog(event.is_final ? 'stt final' : 'stt interim', `T${event.turn_id} · ${event.text}`);
  });
  eventSource.addEventListener('response_chunk', ({ data }) => {
    const event = JSON.parse(data);
    if (event.sequence === 0 && !event.is_final) els.response.textContent = '';
    if (event.text) els.response.textContent += event.text;
    els.responseTurn.textContent = `Turn ${event.turn_id} · streaming`;
  });
  eventSource.addEventListener('response', ({ data }) => {
    const event = JSON.parse(data);
    els.response.textContent = event.text;
    els.responseTurn.textContent = `Turn ${event.turn_id} · final`;
    addLog('ai response', `T${event.turn_id} · ${event.text}`);
  });
  eventSource.addEventListener('tts', ({ data }) => {
    const event = JSON.parse(data);
    const active = event.state === 'audio_started';
    els.audioProgress.classList.toggle('active', active);
    els.audioState.textContent = active ? 'Streaming real Aura audio to LiveKit…' : 'Audio output completed.';
    addLog('tts', `T${event.turn_id} · ${pretty(event.state)}`);
  });
  eventSource.addEventListener('barge_in', () => {
    els.audioProgress.classList.remove('active');
    els.audioState.textContent = 'Playback interrupted by customer speech.';
    addLog('barge-in', 'TTS and queued LiveKit audio cleared');
  });
  eventSource.addEventListener('latency', ({ data }) => {
    const event = JSON.parse(data);
    renderLatency(event);
    addLog('latency', `T${event.turn_id} · ${pretty(event.checkpoint)} · ${event.elapsed_ms} ms`);
  });
  eventSource.addEventListener('error', ({ data }) => {
    if (!data) return;
    const event = JSON.parse(data);
    showToast(event.message || 'Pipeline error');
    addLog('error', event.message || 'Pipeline error');
  });
}

async function connectBrowserAudio(credentials) {
  if (!window.LivekitClient) throw new Error('LiveKit browser SDK could not be loaded.');
  const { Room, RoomEvent, Track } = window.LivekitClient;
  room = new Room({ adaptiveStream: true, dynacast: true });
  room.on(RoomEvent.TrackSubscribed, (track, publication, participant) => {
    const isAiResponse = participant.identity === 'echoai-server'
      && publication.trackName === 'echoai-response';
    if (track.kind === Track.Kind.Audio && isAiResponse) {
      const element = track.attach();
      element.autoplay = true;
      els.remoteAudio.append(element);
      addLog('livekit', 'AI outbound audio track subscribed');
    } else if (track.kind === Track.Kind.Audio) {
      addLog('livekit', `Ignored non-AI audio track from ${participant.identity}`);
    }
  });
  room.on(RoomEvent.TrackUnsubscribed, (track) => track.detach().forEach((element) => element.remove()));
  room.on(RoomEvent.Disconnected, () => addLog('livekit', 'Browser participant disconnected'));
  room.on(RoomEvent.AudioPlaybackStatusChanged, () => {
    els.enableAudio.hidden = room?.canPlaybackAudio !== false;
  });
  room.on(RoomEvent.MediaDevicesError, () => {
    showToast('The browser could not access the microphone. Check its site permission and device availability.');
    addLog('microphone', 'Browser media-device error');
  });
  await room.connect(credentials.livekit_url, credentials.participant_token);
  try { await room.startAudio(); } catch (_) { els.enableAudio.hidden = false; }
  const selectedDevice = els.microphone.value;
  const publication = await room.localParticipant.setMicrophoneEnabled(true, {
    deviceId: selectedDevice || undefined,
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true,
  });
  const mediaTrack = publication?.track?.mediaStreamTrack;
  if (!mediaTrack) throw new Error('Chrome did not provide a usable microphone track.');
  await loadMicrophones(mediaTrack.getSettings().deviceId || selectedDevice);
  microphoneTesting = true;
  applySession({ running: true, connection: 'connected', listening: 'idle', speaking: 'silent', room_name: 'echoai-test-room' });
  addLog('microphone', 'Track published — speak now to verify input level');
  startMicrophoneMeter(mediaTrack);
  await waitForAudibleInput(15000);
  microphoneTesting = false;
  browserReady = true;
  addLog('livekit', 'Browser microphone published to the real room');
}

async function loadMicrophones(preferredDeviceId = '') {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  const devices = (await navigator.mediaDevices.enumerateDevices())
    .filter((device) => device.kind === 'audioinput');
  const selected = preferredDeviceId || els.microphone.value;
  els.microphone.replaceChildren();
  for (const [index, device] of devices.entries()) {
    const option = document.createElement('option');
    option.value = device.deviceId;
    option.textContent = device.label || `Microphone ${index + 1}`;
    els.microphone.append(option);
  }
  if (selected && devices.some((device) => device.deviceId === selected)) {
    els.microphone.value = selected;
  }
}

function startMicrophoneMeter(mediaTrack) {
  stopMicrophoneMeter();
  const AudioContext = window.AudioContext || window.webkitAudioContext;
  audioContext = new AudioContext();
  const analyser = audioContext.createAnalyser();
  analyser.fftSize = 1024;
  meterSource = audioContext.createMediaStreamSource(new MediaStream([mediaTrack]));
  meterSource.connect(analyser);
  const samples = new Float32Array(analyser.fftSize);
  const update = () => {
    analyser.getFloatTimeDomainData(samples);
    let power = 0;
    for (const sample of samples) power += sample * sample;
    const rms = Math.sqrt(power / samples.length);
    const percent = Math.min(100, Math.round(rms * 700));
    els.micMeter.style.width = `${percent}%`;
    els.micMeter.classList.toggle('audible', rms >= 0.003);
    els.micLevel.textContent = rms >= 0.003 ? 'Live' : `${percent}%`;
    audibleFrames = rms >= 0.003 ? audibleFrames + 1 : 0;
    if (audibleFrames >= 3 && audibleInputResolve) {
      const resolve = audibleInputResolve;
      audibleInputResolve = null;
      clearTimeout(audibleInputTimer);
      resolve();
    }
    meterAnimation = requestAnimationFrame(update);
  };
  update();
}

function waitForAudibleInput(timeoutMs) {
  return new Promise((resolve, reject) => {
    audibleInputResolve = resolve;
    audibleInputTimer = setTimeout(() => {
      audibleInputResolve = null;
      reject(new Error('The selected microphone is silent. Choose another microphone and try again.'));
    }, timeoutMs);
  });
}

function stopMicrophoneMeter() {
  if (meterAnimation) cancelAnimationFrame(meterAnimation);
  meterAnimation = null;
  if (audibleInputTimer) clearTimeout(audibleInputTimer);
  audibleInputTimer = null;
  audibleInputResolve = null;
  audibleFrames = 0;
  if (meterSource) meterSource.disconnect();
  meterSource = null;
  if (audioContext) audioContext.close().catch(() => {});
  audioContext = null;
  els.micMeter.style.width = '0%';
  els.micMeter.classList.remove('audible');
  els.micLevel.textContent = 'Idle';
}

async function disconnectBrowserAudio() {
  if (!room) return;
  try { await room.localParticipant.setMicrophoneEnabled(false); } catch (_) { /* already gone */ }
  room.disconnect();
  room = null;
  browserReady = false;
  microphoneTesting = false;
  stopMicrophoneMeter();
  els.enableAudio.hidden = true;
  els.remoteAudio.replaceChildren();
}

els.start.addEventListener('click', async () => {
  els.start.disabled = true;
  callStarting = true;
  browserReady = false;
  setState(els.connection, 'connecting');
  setState(els.listening, 'idle');
  try {
    const credentials = await api('/api/calls/credentials', { method: 'POST' });
    const pipelineStart = api('/api/calls/start', { method: 'POST' });
    await connectBrowserAudio(credentials);
    const state = await pipelineStart;
    callStarting = false;
    applySession(state);
    addLog('system', 'Microphone and voice pipeline are ready — speak now');
  } catch (error) {
    callStarting = false;
    showToast(error.message);
    addLog('error', error.message);
    await disconnectBrowserAudio();
    try { applySession(await api('/api/calls/end', { method: 'POST' })); } catch (_) { /* preserve first error */ }
  }
});

els.end.addEventListener('click', async () => {
  els.end.disabled = true;
  callStarting = false;
  try {
    await disconnectBrowserAudio();
    applySession(await api('/api/calls/end', { method: 'POST' }));
  } catch (error) {
    showToast(error.message);
    addLog('error', error.message);
  }
});

els.enableAudio.addEventListener('click', async () => {
  if (!room) return;
  try {
    await room.startAudio();
    els.enableAudio.hidden = true;
  } catch (_) {
    showToast('Audio playback is still blocked by the browser.');
  }
});

els.persona.addEventListener('change', async () => {
  try {
    const persona = await api('/api/persona', {
      method: 'PUT',
      body: JSON.stringify({ persona_id: els.persona.value }),
    });
    addLog('persona', `Selected ${persona.name}`);
  } catch (error) { showToast(error.message); }
});

els.microphone.addEventListener('change', () => {
  addLog('microphone', `Selected ${els.microphone.selectedOptions[0]?.textContent || 'microphone'}`);
});

els.clearLog.addEventListener('click', () => els.log.replaceChildren());

async function initialize() {
  try {
    const [personas, state] = await Promise.all([api('/api/personas'), api('/api/state')]);
    for (const persona of personas) {
      const option = document.createElement('option');
      option.value = persona.id;
      option.textContent = `${persona.name} · ${persona.speech_rate}x`;
      els.persona.append(option);
    }
    applySession(state);
    await loadMicrophones();
    listenForPipelineEvents();
    addLog('system', 'Phase 7 console connected to backend events');
  } catch (error) { showToast(error.message); }
}

initialize();

navigator.mediaDevices?.addEventListener('devicechange', () => loadMicrophones());
