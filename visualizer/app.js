const DEFAULT_TRACE_URL = "../results/visualization/rollout_trace.json";
const COLORS = {
  local: "#e6a23c",
  bs: "#2077b4",
  sat: "#5b6fb5",
  user: "#17885b",
  failed: "#bc3b3b",
  earth: "#1d5f8f",
  earthDark: "#123f64",
  land: "#2b8c67",
  orbit: "#9fb3c8",
  text: "#172033",
  muted: "#637083",
  grid: "#d8e1ec",
};

const state = {
  trace: null,
  policy: "MADDPG",
  step: 0,
  playing: false,
  timer: null,
};

const canvas = document.getElementById("sceneCanvas");
const ctx = canvas.getContext("2d");
const policyTabs = document.getElementById("policyTabs");
const playButton = document.getElementById("playButton");
const stepSlider = document.getElementById("stepSlider");
const stepLabel = document.getElementById("stepLabel");
const speedSelect = document.getElementById("speedSelect");
const metricGrid = document.getElementById("metricGrid");
const userTable = document.getElementById("userTable");
const traceStatus = document.getElementById("traceStatus");
const traceFile = document.getElementById("traceFile");
const policySummary = document.getElementById("policySummary");
const orbitDetails = document.getElementById("orbitDetails");

function fmt(value, digits = 2) {
  if (!Number.isFinite(value)) return "-";
  return value.toFixed(digits);
}

function pct(value) {
  if (!Number.isFinite(value)) return "-";
  return `${(value * 100).toFixed(1)}%`;
}

function activePolicyTrace() {
  return state.trace.policies[state.policy];
}

function activeStep() {
  return activePolicyTrace().steps[state.step];
}

function policyNames() {
  return Object.keys(state.trace.policies);
}

function setCanvasResolution() {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(640, Math.floor(rect.width * ratio));
  canvas.height = Math.max(420, Math.floor(rect.height * ratio));
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function drawGrid(width, height) {
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#f8fbfe";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 1;
  for (let x = 40; x < width; x += 80) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = 40; y < height; y += 80) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
}

function orbitProject(point, center, earthRadius) {
  return {
    x: center.x + point.x_norm * earthRadius,
    y: center.y - point.y_norm * earthRadius,
  };
}

function drawEarth(center, radius) {
  const gradient = ctx.createRadialGradient(
    center.x - radius * 0.35,
    center.y - radius * 0.45,
    radius * 0.15,
    center.x,
    center.y,
    radius,
  );
  gradient.addColorStop(0, "#3ea6d8");
  gradient.addColorStop(0.55, COLORS.earth);
  gradient.addColorStop(1, COLORS.earthDark);
  ctx.save();
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(center.x, center.y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "#ffffff";
  ctx.globalAlpha = 0.95;
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.globalAlpha = 0.28;
  ctx.fillStyle = COLORS.land;
  [
    [-0.35, -0.28, 0.28, 0.14],
    [0.18, -0.1, 0.32, 0.18],
    [-0.08, 0.28, 0.34, 0.13],
  ].forEach(([x, y, rx, ry]) => {
    ctx.beginPath();
    ctx.ellipse(center.x + x * radius, center.y + y * radius, rx * radius, ry * radius, 0.3, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.globalAlpha = 0.22;
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 1;
  for (let i = -2; i <= 2; i += 1) {
    ctx.beginPath();
    ctx.ellipse(center.x, center.y, radius, radius * (0.22 + Math.abs(i) * 0.17), 0, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.restore();
}

function drawOrbit(center, earthRadius, orbitRadius) {
  ctx.save();
  ctx.strokeStyle = COLORS.orbit;
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 8]);
  ctx.globalAlpha = 0.85;
  ctx.beginPath();
  ctx.arc(center.x, center.y, orbitRadius, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

function drawOrbitLabel(text, x, y) {
  ctx.save();
  ctx.fillStyle = COLORS.text;
  ctx.font = "12px Microsoft YaHei, Segoe UI, sans-serif";
  ctx.fillText(text, x, y);
  ctx.restore();
}

function drawLink(from, to, color, ratio) {
  if (ratio <= 0.02) return;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.globalAlpha = Math.min(0.85, 0.18 + ratio * 0.75);
  ctx.lineWidth = 1 + ratio * 6;
  ctx.beginPath();
  ctx.moveTo(from.x, from.y);
  ctx.lineTo(to.x, to.y);
  ctx.stroke();
  ctx.restore();
}

function drawNode(point, radius, color, label) {
  ctx.save();
  ctx.fillStyle = color;
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = COLORS.text;
  ctx.font = "12px Microsoft YaHei, Segoe UI, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(label, point.x, point.y - radius - 8);
  ctx.restore();
}

function drawOrbitNode(point, center, radius, color, label) {
  ctx.save();
  ctx.fillStyle = color;
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();

  const dx = point.x - center.x;
  const dy = point.y - center.y;
  const length = Math.max(1, Math.hypot(dx, dy));
  const labelX = point.x + (dx / length) * (radius + 12);
  const labelY = point.y + (dy / length) * (radius + 12);
  ctx.fillStyle = COLORS.text;
  ctx.font = "12px Microsoft YaHei, Segoe UI, sans-serif";
  ctx.textAlign = dx >= 0 ? "left" : "right";
  ctx.textBaseline = "middle";
  ctx.fillText(label, labelX, labelY);
  ctx.restore();
}

function drawScene() {
  if (!state.trace) return;
  setCanvasResolution();
  const rect = canvas.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  const step = activeStep();
  const orbit = step.orbit_view;
  const center = { x: width * 0.48, y: height * 0.52 };
  const orbitScale = orbit.orbit_radius_km / orbit.earth_radius_km;
  const earthRadius = Math.min(width, height) / (orbitScale * 2 + 0.9);
  const orbitRadius = earthRadius * orbitScale;
  const bsPoint = orbitProject(orbit.base_station, center, earthRadius);
  const satPoint = orbitProject(orbit.satellite, center, earthRadius);
  const orbitUsers = new Map(orbit.users.map((user) => [user.id, user]));

  drawGrid(width, height);
  drawOrbit(center, earthRadius, orbitRadius);
  drawEarth(center, earthRadius);

  step.users.forEach((user) => {
    const orbitUser = orbitUsers.get(user.id);
    const userPoint = orbitProject(orbitUser, center, earthRadius);
    drawLink(userPoint, bsPoint, COLORS.bs, user.action[1]);
    drawLink(userPoint, satPoint, COLORS.sat, user.action[2]);
  });

  drawOrbitNode(bsPoint, center, 12, COLORS.bs, "BS MEC");
  drawOrbitNode(satPoint, center, 13, COLORS.sat, "LEO MEC");

  step.users.forEach((user) => {
    const orbitUser = orbitUsers.get(user.id);
    const userPoint = orbitProject(orbitUser, center, earthRadius);
    drawOrbitNode(
      userPoint,
      center,
      8 + user.task_data_mb * 0.8,
      user.success ? COLORS.user : COLORS.failed,
      `U${user.id + 1}`,
    );
  });

  ctx.fillStyle = COLORS.muted;
  ctx.font = "12px Microsoft YaHei, Segoe UI, sans-serif";
  ctx.textAlign = "left";
  drawOrbitLabel(`演示轨道视图：卫星高度 ${Math.round(orbit.satellite.altitude_km)} km`, 16, height - 34);
  drawOrbitLabel("动画按一轮仿真展示完整绕行；链路和指标仍来自原仿真模型", 16, height - 16);
}

function renderPolicyTabs() {
  policyTabs.innerHTML = "";
  policyNames().forEach((name) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = name;
    button.className = name === state.policy ? "active" : "";
    button.addEventListener("click", () => {
      state.policy = name;
      state.step = Math.min(state.step, activePolicyTrace().steps.length - 1);
      stopPlayback();
      render();
    });
    policyTabs.appendChild(button);
  });
}

function renderMetrics() {
  const step = activeStep();
  const metrics = [
    ["平均时延", `${fmt(step.metrics.avg_delay)} s`],
    ["平均能耗", `${fmt(step.metrics.avg_energy)} J`],
    ["成功率", pct(step.metrics.success_rate)],
    ["平均奖励", fmt(step.metrics.avg_reward, 3)],
  ];
  metricGrid.innerHTML = metrics
    .map((item) => `<div class="metric-card"><span>${item[0]}</span><strong>${item[1]}</strong></div>`)
    .join("");

  const local = step.metrics.avg_local_ratio;
  const bs = step.metrics.avg_bs_ratio;
  const sat = step.metrics.avg_sat_ratio;
  document.getElementById("avgLocalBar").style.width = pct(local);
  document.getElementById("avgBsBar").style.width = pct(bs);
  document.getElementById("avgSatBar").style.width = pct(sat);
  document.getElementById("ratioLabels").innerHTML = `
    <span>本地计算 ${pct(local)}</span>
    <span>地面基站 MEC ${pct(bs)}</span>
    <span>低轨卫星 MEC ${pct(sat)}</span>
  `;
  renderOrbitDetails(step);
}

function renderOrbitDetails(step) {
  const orbit = step.orbit_view;
  const avgSatDistanceKm =
    step.users.reduce((total, user) => total + user.sat_distance_m / 1000, 0) / Math.max(1, step.users.length);
  const maxSatRatioUser = step.users.reduce((best, user) => (user.action[2] > best.action[2] ? user : best), step.users[0]);
  orbitDetails.innerHTML = `
    <span><strong>地球半径：</strong>${fmt(orbit.earth_radius_km, 0)} km</span>
    <span><strong>轨道半径：</strong>${fmt(orbit.orbit_radius_km, 0)} km</span>
    <span><strong>卫星高度：</strong>${fmt(orbit.satellite.altitude_km, 0)} km</span>
    <span><strong>平均星地距离：</strong>${fmt(avgSatDistanceKm, 1)} km</span>
    <span><strong>卫星卸载最高用户：</strong>U${maxSatRatioUser.id + 1}，${pct(maxSatRatioUser.action[2])}</span>
    <span>${orbit.model_note}</span>
  `;
}

function ratioStack(action) {
  return `
    <div class="user-ratio">
      <div class="mini-stack">
        <div class="ratio-segment local" style="width:${pct(action[0])}"></div>
        <div class="ratio-segment bs" style="width:${pct(action[1])}"></div>
        <div class="ratio-segment sat" style="width:${pct(action[2])}"></div>
      </div>
      <span>本地 ${pct(action[0])} / 基站 ${pct(action[1])} / 卫星 ${pct(action[2])}</span>
    </div>
  `;
}

function renderUserTable() {
  const step = activeStep();
  userTable.innerHTML = step.users
    .map(
      (user) => `
      <tr>
        <td>U${user.id + 1}</td>
        <td>${fmt(user.task_data_mb)}</td>
        <td>${fmt(user.deadline_s)}</td>
        <td>${fmt(user.delay_s)}</td>
        <td>${fmt(user.energy_j)}</td>
        <td>${ratioStack(user.action)}</td>
        <td><span class="status ${user.success ? "ok" : "bad"}">${user.success ? "完成" : "超时"}</span></td>
      </tr>
    `,
    )
    .join("");
}

function renderControls() {
  const steps = activePolicyTrace().steps.length;
  stepSlider.max = String(steps - 1);
  stepSlider.value = String(state.step);
  stepLabel.textContent = `${state.step + 1}/${steps}`;
  playButton.textContent = state.playing ? "Ⅱ" : "▶";
  policySummary.textContent = `${state.policy}，第 ${state.step + 1} / ${steps} 步`;
}

function render() {
  if (!state.trace) return;
  renderPolicyTabs();
  renderControls();
  renderMetrics();
  renderUserTable();
  drawScene();
}

function nextStep() {
  const maxStep = activePolicyTrace().steps.length - 1;
  if (state.step >= maxStep) {
    state.step = 0;
  } else {
    state.step += 1;
  }
  render();
}

function startPlayback() {
  if (state.playing) return;
  state.playing = true;
  playButton.textContent = "Ⅱ";
  state.timer = window.setInterval(nextStep, Number(speedSelect.value));
}

function stopPlayback() {
  state.playing = false;
  if (state.timer) window.clearInterval(state.timer);
  state.timer = null;
  playButton.textContent = "▶";
}

function setTrace(trace, sourceLabel) {
  state.trace = trace;
  state.policy = trace.policies[state.policy] ? state.policy : policyNames()[0];
  state.step = 0;
  traceStatus.textContent = sourceLabel;
  render();
}

async function loadDefaultTrace() {
  try {
    const response = await fetch(DEFAULT_TRACE_URL, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const trace = await response.json();
    setTrace(trace, "已加载默认轨迹");
  } catch (error) {
    traceStatus.textContent = "默认轨迹不可用，请选择 JSON 文件";
  }
}

playButton.addEventListener("click", () => {
  if (!state.trace) return;
  if (state.playing) stopPlayback();
  else startPlayback();
});

stepSlider.addEventListener("input", () => {
  if (!state.trace) return;
  stopPlayback();
  state.step = Number(stepSlider.value);
  render();
});

speedSelect.addEventListener("change", () => {
  if (state.playing) {
    stopPlayback();
    startPlayback();
  }
});

traceFile.addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;
  const text = await file.text();
  setTrace(JSON.parse(text), `已加载 ${file.name}`);
});

window.addEventListener("resize", () => {
  window.requestAnimationFrame(drawScene);
});

loadDefaultTrace();
