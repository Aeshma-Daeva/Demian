import * as THREE from "./vendor/three.module.js";

const DEFAULT_DATASET = "/data/substrate_lab/v9_5ch_evo_trajectory_3d_20260509_full/trajectory_3d.json";
const datasetUrl = new URLSearchParams(window.location.search).get("data") || DEFAULT_DATASET;
const SUBSTRATE_COLORS = {
  demian_native_v9: 0x45a3ff,
  demian_native_v8: 0xe66f51,
  bound_no_release: 0x8fa0aa,
  manual_release_medium: 0xf5c451,
  learned_release_neutral: 0x64d68a,
  learned_release_high_threshold: 0xa78bfa,
  gen000_candidate000: 0x64d68a,
  gen000_candidate001: 0xf5c451,
  gen000_candidate002: 0xa78bfa,
  gen000_candidate003: 0x45a3ff,
};

const canvas = document.querySelector("#trajectory-canvas");
const emptyState = document.querySelector("#empty-state");
const inspectorFields = document.querySelector("#inspector-fields");
const fileInput = document.querySelector("#json-file");
const releaseChart = document.querySelector("#release-chart");
const recoveryChart = document.querySelector("#recovery-chart");
const phaseChart = document.querySelector("#phase-chart");
const axisLoadings = document.querySelector("#axis-loadings");
const controls = {
  showV9: document.querySelector("#show-v9"),
  showV8: document.querySelector("#show-v8"),
  showClean: document.querySelector("#show-clean"),
  showPerturbed: document.querySelector("#show-perturbed"),
  colorMode: document.querySelector("#color-mode"),
  seedFilter: document.querySelector("#seed-filter"),
  candidateFilter: document.querySelector("#candidate-filter"),
  familyFilter: document.querySelector("#family-filter"),
  scaleFilter: document.querySelector("#scale-filter"),
  regimeFilter: document.querySelector("#regime-filter"),
  stepLimit: document.querySelector("#step-limit"),
  stepOutput: document.querySelector("#step-output"),
  releaseMetric: document.querySelector("#release-metric"),
  releaseThreshold: document.querySelector("#release-threshold"),
  releaseThresholdOutput: document.querySelector("#release-threshold-output"),
  releaseWindow: document.querySelector("#release-window"),
  releaseWindowOutput: document.querySelector("#release-window-output"),
  eventAlign: document.querySelector("#event-align"),
  releaseWindowOnly: document.querySelector("#release-window-only"),
  showReleaseMarkers: document.querySelector("#show-release-markers"),
  showReleaseVectors: document.querySelector("#show-release-vectors"),
  showRecoveryVectors: document.querySelector("#show-recovery-vectors"),
  showDifferenceVectors: document.querySelector("#show-difference-vectors"),
};

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x080a0d);

const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 1000);
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

const orbit = {
  target: new THREE.Vector3(),
  radius: 9,
  yaw: Math.PI / 4,
  pitch: 0.62,
  dragging: false,
  lastX: 0,
  lastY: 0,
};

scene.add(new THREE.AmbientLight(0xffffff, 0.75));
const keyLight = new THREE.DirectionalLight(0xffffff, 0.7);
keyLight.position.set(4, 6, 5);
scene.add(keyLight);

const grid = new THREE.GridHelper(10, 10, 0x26313b, 0x151b22);
grid.rotation.x = Math.PI / 2;
scene.add(grid);

const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 0.08;
const pointer = new THREE.Vector2();

let dataset = null;
let viewData = null;
let pointsObject = null;
const lineGroup = new THREE.Group();
const eventGroup = new THREE.Group();
const vectorGroup = new THREE.Group();
let hoveredIndex = -1;
scene.add(lineGroup);
scene.add(eventGroup);
scene.add(vectorGroup);

fileInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  dataset = normalizeDataset(JSON.parse(await file.text()));
  configureControls();
  buildScene();
});

for (const control of Object.values(controls)) {
  control.addEventListener("change", buildScene);
}
controls.stepLimit.addEventListener("input", () => {
  controls.stepOutput.value = controls.stepLimit.value;
  buildScene();
});
controls.releaseThreshold.addEventListener("input", () => {
  controls.releaseThresholdOutput.value = formatThreshold(Number(controls.releaseThreshold.value));
  buildScene();
});
controls.releaseMetric.addEventListener("change", () => {
  configureReleaseThreshold();
  buildScene();
});
controls.releaseWindow.addEventListener("input", () => {
  controls.releaseWindowOutput.value = controls.releaseWindow.value;
  buildScene();
});

canvas.addEventListener("pointermove", handlePick);
canvas.addEventListener("click", () => {
  if (hoveredIndex >= 0 && pointsObject) {
    showPoint(pointsObject.userData.visiblePoints[hoveredIndex]);
  }
});
canvas.addEventListener("pointerdown", (event) => {
  orbit.dragging = true;
  orbit.lastX = event.clientX;
  orbit.lastY = event.clientY;
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointerup", (event) => {
  orbit.dragging = false;
  canvas.releasePointerCapture(event.pointerId);
});
canvas.addEventListener("pointerleave", () => {
  orbit.dragging = false;
});
canvas.addEventListener("pointermove", (event) => {
  if (!orbit.dragging) return;
  const dx = event.clientX - orbit.lastX;
  const dy = event.clientY - orbit.lastY;
  orbit.lastX = event.clientX;
  orbit.lastY = event.clientY;
  orbit.yaw -= dx * 0.008;
  orbit.pitch = Math.max(-1.35, Math.min(1.35, orbit.pitch - dy * 0.008));
  updateCamera();
});
canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  orbit.radius = Math.max(0.4, orbit.radius * (event.deltaY > 0 ? 1.08 : 0.92));
  updateCamera();
}, { passive: false });

window.addEventListener("resize", resize);
resize();
animate();
showMessage("Loading default trajectory export.");
loadDefaultDataset();

async function loadDefaultDataset() {
  try {
    const response = await fetch(datasetUrl);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    dataset = normalizeDataset(await response.json());
    configureControls();
    emptyState.classList.add("hidden");
    buildScene();
  } catch {
    showMessage("Use the file picker to load trajectory_3d.json.");
  }
}

function normalizeDataset(raw) {
  const metricsByPoint = new Map(raw.metrics.map((row) => [row.point_id, row]));
  const pointById = new Map();
  const cleanByKey = new Map();

  const points = raw.points.map((point) => {
    const metrics = metricsByPoint.get(point.point_id) || {};
    const fullPoint = {
      ...point,
      metrics,
      original: new THREE.Vector3(point.x, point.y, point.z),
    };
    pointById.set(point.point_id, fullPoint);
    if (point.run_kind === "clean") {
      cleanByKey.set(cleanKey(point), fullPoint);
    }
    return fullPoint;
  });

  for (const point of points) {
    const clean = cleanByKey.get(cleanKey(point));
    point.clean_distance = clean ? point.original.distanceTo(clean.original) : 0;
    point.path_distance = point.clean_distance;
  }

  const pathOrigins = new Map();
  const perturbStep = raw.metadata?.perturb_step;
  for (const point of points) {
    if (point.step === perturbStep) {
      pathOrigins.set(pathKey(point), point.original.clone());
    }
  }
  for (const point of points) {
    if (point.clean_distance > 0) continue;
    const origin = pathOrigins.get(pathKey(point));
    point.path_distance = origin ? point.original.distanceTo(origin) : 0;
  }

  return {
    ...raw,
    points,
    events: raw.events.map((event) => ({
      ...event,
      metrics: metricsByPoint.get(event.point_id) || {},
      original: new THREE.Vector3(event.x, event.y, event.z),
    })),
    pointById,
    cleanByKey,
  };
}

function configureControls() {
  const seeds = [...new Set(dataset.points.map((point) => point.seed))].sort((a, b) => a - b);
  const candidates = [...new Set(dataset.points.map((point) => point.candidate_id || point.substrate))]
    .sort();
  const families = [...new Set(dataset.points.map((point) => point.perturb_family || "none"))]
    .sort();
  const scales = [...new Set(dataset.points
    .filter((point) => point.perturb_scale !== null && point.perturb_scale !== undefined)
    .map((point) => String(point.perturb_scale)))].sort((a, b) => Number(a) - Number(b));
  const regimes = [...new Set(dataset.points.map((point) => point.regime_class || "unknown"))]
    .sort();
  fillSelect(controls.seedFilter, ["all", ...seeds.map(String)]);
  fillSelect(controls.candidateFilter, ["all", ...candidates]);
  fillSelect(controls.familyFilter, ["all", ...families]);
  fillSelect(controls.scaleFilter, ["all", "clean", ...scales]);
  fillSelect(controls.regimeFilter, ["all", ...regimes]);
  controls.stepLimit.max = String(dataset.metadata.steps);
  controls.stepLimit.value = String(dataset.metadata.steps);
  controls.stepOutput.value = String(dataset.metadata.steps);
  configureReleaseThreshold();
  renderAxisLoadings();
}

function configureReleaseThreshold() {
  const metric = controls.releaseMetric.value;
  const extent = metricExtent(metric);
  const max = Math.max(extent.max, 0.001);
  const defaultValue = metric === "release_strength_mean"
    ? Math.min(0.005, max)
    : max * 0.72;
  controls.releaseThreshold.max = String(max);
  controls.releaseThreshold.step = String(max < 0.02 ? 0.0001 : 0.001);
  controls.releaseThreshold.value = String(defaultValue);
  controls.releaseThresholdOutput.value = formatThreshold(defaultValue);
}

function fillSelect(select, values) {
  const previous = select.value;
  select.replaceChildren(...values.map((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    return option;
  }));
  select.value = values.includes(previous) ? previous : "all";
}

function buildScene() {
  clearObject(pointsObject);
  clearGroup(lineGroup);
  clearGroup(eventGroup);
  clearGroup(vectorGroup);
  if (!dataset) return;

  viewData = makeViewData();
  const visiblePoints = viewData.points;
  const positions = new Float32Array(visiblePoints.length * 3);
  const colors = new Float32Array(visiblePoints.length * 3);

  visiblePoints.forEach((point, index) => {
    positions[index * 3] = point.view.x;
    positions[index * 3 + 1] = point.view.y;
    positions[index * 3 + 2] = point.view.z;
    const color = colorFor(point);
    colors[index * 3] = color.r;
    colors[index * 3 + 1] = color.g;
    colors[index * 3 + 2] = color.b;
  });

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  pointsObject = new THREE.Points(
    geometry,
    new THREE.PointsMaterial({ size: 0.055, vertexColors: true, transparent: true, opacity: 0.92 }),
  );
  pointsObject.userData.visiblePoints = visiblePoints;
  scene.add(pointsObject);

  buildLines(visiblePoints);
  buildEvents(viewData.events);
  buildReleaseMarkers(viewData.releasePoints);
  buildVectors(viewData.points);
  drawReleaseChart(viewData.points, viewData.releasePoints);
  drawRecoveryChart(viewData.points);
  drawPhaseChart(viewData.points);
  framePoints(visiblePoints);
  showMessage(`${visiblePoints.length} visible points; ${viewData.releasePoints.length} release points`);
}

function makeViewData() {
  const stepLimit = Number(controls.stepLimit.value);
  const origins = eventAlignmentOrigins();
  const releaseWindowKeys = releaseWindowKeysFor(stepLimit);
  const points = dataset.points
    .filter((point) => isVisible(point) && point.step <= stepLimit && isInReleaseWindow(point, releaseWindowKeys))
    .map((point) => attachViewPosition(point, origins));
  const events = dataset.events
    .filter((event) => isVisible(event) && event.step <= stepLimit && isInReleaseWindow(event, releaseWindowKeys))
    .map((event) => attachViewPosition(event, origins));
  const releasePoints = points.filter((point) => isReleasePoint(point));
  return { points, events, releasePoints };
}

function releaseWindowKeysFor(stepLimit) {
  if (!controls.releaseWindowOnly.checked) return null;
  const windowSize = Number(controls.releaseWindow.value);
  const keys = new Set();
  const releaseStepsByPath = new Map();
  for (const point of dataset.points) {
    if (!isVisible(point) || point.step > stepLimit || !isReleasePoint(point)) continue;
    const key = pathKey(point);
    if (!releaseStepsByPath.has(key)) releaseStepsByPath.set(key, []);
    releaseStepsByPath.get(key).push(point.step);
  }
  for (const [key, steps] of releaseStepsByPath.entries()) {
    for (const releaseStep of steps) {
      const start = Math.max(1, releaseStep - windowSize);
      const end = Math.min(stepLimit, releaseStep + windowSize);
      for (let step = start; step <= end; step += 1) {
        keys.add(`${key}|${step}`);
      }
    }
  }
  return keys;
}

function isInReleaseWindow(point, releaseWindowKeys) {
  if (!releaseWindowKeys) return true;
  return releaseWindowKeys.has(`${pathKey(point)}|${point.step}`);
}

function attachViewPosition(point, origins) {
  const origin = controls.eventAlign.checked
    ? origins.get(originKey(point)) || new THREE.Vector3()
    : new THREE.Vector3();
  return {
    ...point,
    view: point.original.clone().sub(origin),
  };
}

function eventAlignmentOrigins() {
  const origins = new Map();
  if (!controls.eventAlign.checked) return origins;
  const perturbStep = dataset.metadata.perturb_step;
  for (const point of dataset.points) {
    if (point.run_kind === "clean" && point.step === perturbStep) {
      origins.set(originKey(point), point.original.clone());
    }
  }
  return origins;
}

function buildLines(visiblePoints) {
  const groups = new Map();
  for (const point of visiblePoints) {
    const key = pathKey(point);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(point);
  }
  for (const group of groups.values()) {
    group.sort((a, b) => a.step - b.step);
    const positions = new Float32Array(group.length * 3);
    group.forEach((point, index) => {
      positions[index * 3] = point.view.x;
      positions[index * 3 + 1] = point.view.y;
      positions[index * 3 + 2] = point.view.z;
    });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    lineGroup.add(new THREE.Line(
      geometry,
      new THREE.LineBasicMaterial({
        color: colorFor(group[0]),
        transparent: true,
        opacity: group[0].run_kind === "clean" ? 0.62 : 0.34,
      }),
    ));
  }
}

function buildEvents(events) {
  const markerGeometry = new THREE.SphereGeometry(0.09, 16, 12);
  for (const event of events) {
    const material = new THREE.MeshStandardMaterial({
      color: event.event_kind === "perturbation" ? 0xf5c451 : 0xf2f4f6,
      emissive: event.event_kind === "perturbation" ? 0x4b3100 : 0x22272c,
      roughness: 0.6,
    });
    const marker = new THREE.Mesh(markerGeometry, material);
    marker.position.copy(event.view);
    marker.userData.point = event;
    eventGroup.add(marker);
  }
}

function buildReleaseMarkers(releasePoints) {
  if (!controls.showReleaseMarkers.checked) return;
  const markerGeometry = new THREE.SphereGeometry(0.075, 16, 12);
  const maxRelease = Math.max(...releasePoints.map((point) => releaseValue(point)), 1e-6);
  for (const point of releasePoints) {
    const normalized = Math.min(releaseValue(point) / maxRelease, 1);
    const material = new THREE.MeshStandardMaterial({
      color: 0xffb347,
      emissive: 0x5c2d00,
      emissiveIntensity: 0.5 + normalized,
      roughness: 0.55,
      transparent: true,
      opacity: 0.62 + normalized * 0.34,
    });
    const marker = new THREE.Mesh(markerGeometry, material);
    marker.scale.setScalar(0.75 + normalized * 2.6);
    marker.position.copy(point.view);
    marker.userData.point = point;
    eventGroup.add(marker);
  }
}

function buildVectors(visiblePoints) {
  if (controls.showReleaseVectors.checked) {
    buildReleaseVectors(visiblePoints);
  }
  if (controls.showRecoveryVectors.checked) {
    buildRecoveryVectors(visiblePoints);
  }
  if (controls.showDifferenceVectors.checked) {
    buildDifferenceVectors(visiblePoints);
  }
}

function buildReleaseVectors(visiblePoints) {
  const byPath = new Map();
  const windowSize = Number(controls.releaseWindow.value);
  for (const point of visiblePoints) {
    const key = pathKey(point);
    if (!byPath.has(key)) byPath.set(key, []);
    byPath.get(key).push(point);
  }
  for (const points of byPath.values()) {
    points.sort((a, b) => a.step - b.step);
    const byStep = new Map(points.map((point) => [point.step, point]));
    for (const point of points) {
      if (!isReleasePoint(point)) continue;
      const before = byStep.get(Math.max(1, point.step - windowSize));
      const after = byStep.get(point.step + windowSize);
      if (!before || !after) continue;
      addVector(before.view, after.view, new THREE.Color(0xffb347), 0.46);
    }
  }
}

function buildRecoveryVectors(visiblePoints) {
  const perturbStep = dataset.metadata.perturb_step;
  for (const point of visiblePoints) {
    if (point.run_kind !== "perturbed" || point.step !== perturbStep) continue;
    const clean = visiblePoints.find((candidate) =>
      candidate.substrate === point.substrate
      && candidate.seed === point.seed
      && candidate.run_kind === "clean"
      && candidate.step === point.step
    );
    if (!clean) continue;
    addVector(clean.view, point.view, colorFor(point), 0.52);
  }
}

function buildDifferenceVectors(visiblePoints) {
  const byPair = new Map();
  for (const point of visiblePoints) {
    const key = `${point.seed}|${point.run_kind}|${point.perturb_scale ?? "clean"}|${point.step}`;
    if (!byPair.has(key)) byPair.set(key, {});
    byPair.get(key)[point.substrate] = point;
  }
  for (const pair of byPair.values()) {
    const v8 = pair.demian_native_v8;
    const v9 = pair.demian_native_v9;
    if (!v8 || !v9 || v8.step % 8 !== 0) continue;
    addVector(v8.view, v9.view, new THREE.Color(0xa7d8ff), 0.2);
  }
}

function addVector(start, end, color, opacity) {
  const delta = end.clone().sub(start);
  const length = delta.length();
  if (length < 1e-6) return;
  const arrow = new THREE.ArrowHelper(
    delta.clone().normalize(),
    start,
    length,
    color instanceof THREE.Color ? color.getHex() : color,
    Math.min(length * 0.22, 0.24),
    Math.min(length * 0.1, 0.1),
  );
  arrow.cone.material.transparent = true;
  arrow.line.material.transparent = true;
  arrow.cone.material.opacity = opacity;
  arrow.line.material.opacity = opacity;
  vectorGroup.add(arrow);
}

function isVisible(item) {
  const candidate = item.candidate_id || item.substrate;
  const isV9Line = item.substrate === "demian_native_v9"
    || item.substrate?.startsWith("gen")
    || candidate.startsWith("gen")
    || item.substrate?.includes("release");
  if (isV9Line && !controls.showV9.checked) return false;
  if (item.substrate === "demian_native_v8" && !controls.showV8.checked) return false;
  if (item.run_kind === "clean" && !controls.showClean.checked) return false;
  if (item.run_kind === "perturbed" && !controls.showPerturbed.checked) return false;
  if (controls.seedFilter.value !== "all" && String(item.seed) !== controls.seedFilter.value) return false;
  if (controls.candidateFilter.value !== "all" && candidate !== controls.candidateFilter.value) return false;
  const family = item.perturb_family || "none";
  if (controls.familyFilter.value !== "all" && family !== controls.familyFilter.value) return false;
  const regime = item.regime_class || "unknown";
  if (controls.regimeFilter.value !== "all" && regime !== controls.regimeFilter.value) return false;
  if (controls.scaleFilter.value === "clean" && item.run_kind !== "clean") return false;
  if (controls.scaleFilter.value !== "all" && controls.scaleFilter.value !== "clean") {
    if (String(item.perturb_scale) !== controls.scaleFilter.value) return false;
  }
  return true;
}

function isReleasePoint(point) {
  return releaseValue(point) >= Number(controls.releaseThreshold.value);
}

function releaseValue(point) {
  return metricValue(point, controls.releaseMetric.value);
}

function colorFor(point) {
  const mode = controls.colorMode.value;
  const color = new THREE.Color();
  if (mode === "substrate") {
    return color.set(SUBSTRATE_COLORS[point.substrate] || 0xb9c5cc);
  }
  if (mode === "perturb_scale") {
    if (point.perturb_scale === null || point.perturb_scale === undefined) return color.set(0xb9c5cc);
    return color.setHSL(0.12 - Math.min(Number(point.perturb_scale), 1) * 0.1, 0.82, 0.58);
  }
  const value = metricValue(point, mode);
  const maxValue = metricExtent(mode).max || 1;
  const normalized = Math.max(0, Math.min(value / maxValue, 1));
  return color.setHSL(0.58 - normalized * 0.48, 0.76, 0.56);
}

function metricValue(point, key) {
  if (key === "clean_distance") return point.clean_distance || 0;
  if (key === "path_distance") return point.path_distance || point.clean_distance || 0;
  if (key === "carrier_residual_norm") {
    return Number(point.metrics?.route_metrics?.carrier_residual_norm || 0);
  }
  if (point.metrics?.route_metrics && key in point.metrics.route_metrics) {
    return Number(point.metrics.route_metrics[key] || 0);
  }
  return Number(point.metrics?.[key] || 0);
}

function metricExtent(key) {
  const values = dataset.points.map((point) => metricValue(point, key)).filter(Number.isFinite);
  return { min: Math.min(...values, 0), max: Math.max(...values, 0) };
}

function drawReleaseChart(points, releasePoints) {
  const ctx = setupChart(releaseChart);
  const metric = controls.releaseMetric.value;
  const curves = groupCurves(points);
  const all = [...curves.values()].flat();
  const maxStep = Number(controls.stepLimit.value);
  const maxY = Math.max(
    ...all.map((point) => metricValue(point, metric)),
    Number(controls.releaseThreshold.value),
    1e-6,
  );
  drawAxes(ctx, releaseChart.width, releaseChart.height, "step", metric.replace("release_", ""));
  const thresholdY = releaseChart.height - 24
    - (Number(controls.releaseThreshold.value) / maxY) * (releaseChart.height - 42);
  ctx.strokeStyle = "#ffb347";
  ctx.globalAlpha = 0.45;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(34, thresholdY);
  ctx.lineTo(releaseChart.width - 12, thresholdY);
  ctx.stroke();
  ctx.setLineDash([]);
  for (const curve of curves.values()) {
    curve.sort((a, b) => a.step - b.step);
    ctx.beginPath();
    curve.forEach((point, index) => {
      const x = 34 + (point.step / maxStep) * (releaseChart.width - 48);
      const y = releaseChart.height - 24 - (metricValue(point, metric) / maxY) * (releaseChart.height - 42);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = colorForCanvas(curve[0]);
    ctx.globalAlpha = 0.55;
    ctx.stroke();
  }
  ctx.globalAlpha = 0.95;
  ctx.fillStyle = "#ffb347";
  for (const point of releasePoints) {
    const x = 34 + (point.step / maxStep) * (releaseChart.width - 48);
    const y = releaseChart.height - 24 - (metricValue(point, metric) / maxY) * (releaseChart.height - 42);
    ctx.fillRect(x - 1.5, y - 1.5, 3, 3);
  }
  ctx.globalAlpha = 1;
}

function drawRecoveryChart(points) {
  const ctx = setupChart(recoveryChart);
  const curves = groupCurves(points.filter((point) => point.run_kind === "perturbed"));
  const all = [...curves.values()].flat();
  const maxStep = Number(controls.stepLimit.value);
  const maxY = Math.max(...all.map((point) => point.path_distance), 1e-6);
  drawAxes(ctx, recoveryChart.width, recoveryChart.height, "step", "path distance");
  for (const curve of curves.values()) {
    curve.sort((a, b) => a.step - b.step);
    ctx.beginPath();
    curve.forEach((point, index) => {
      const x = 34 + (point.step / maxStep) * (recoveryChart.width - 48);
      const y = recoveryChart.height - 24 - (point.path_distance / maxY) * (recoveryChart.height - 42);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = colorForCanvas(curve[0]);
    ctx.globalAlpha = 0.7;
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

function drawPhaseChart(points) {
  const ctx = setupChart(phaseChart);
  const plotted = points.filter((point) => point.run_kind === "clean" || point.run_kind === "perturbed");
  const xs = plotted.map((point) => metricValue(point, "slow_state_norm"));
  const ys = plotted.map((point) => metricValue(point, "message_state_norm"));
  const maxX = Math.max(...xs, 1e-6);
  const maxY = Math.max(...ys, 1e-6);
  drawAxes(ctx, phaseChart.width, phaseChart.height, "slow norm", "message norm");
  for (const point of plotted) {
    const x = 34 + (metricValue(point, "slow_state_norm") / maxX) * (phaseChart.width - 48);
    const y = phaseChart.height - 24 - (metricValue(point, "message_state_norm") / maxY) * (phaseChart.height - 42);
    ctx.fillStyle = colorForCanvas(point);
    ctx.globalAlpha = point.run_kind === "clean" ? 0.55 : 0.32;
    ctx.fillRect(x - 1, y - 1, 2, 2);
  }
  ctx.globalAlpha = 1;
}

function setupChart(chart) {
  const rect = chart.getBoundingClientRect();
  chart.width = Math.max(260, Math.floor(rect.width));
  chart.height = Math.max(110, Math.floor(rect.height));
  const ctx = chart.getContext("2d");
  ctx.clearRect(0, 0, chart.width, chart.height);
  return ctx;
}

function drawAxes(ctx, width, height, xLabel, yLabel) {
  ctx.strokeStyle = "#31404c";
  ctx.fillStyle = "#8fa0aa";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(34, 10);
  ctx.lineTo(34, height - 24);
  ctx.lineTo(width - 12, height - 24);
  ctx.stroke();
  ctx.font = "11px sans-serif";
  ctx.fillText(xLabel, width - 82, height - 7);
  ctx.save();
  ctx.translate(11, 72);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
}

function colorForCanvas(point) {
  return `#${colorFor(point).getHexString()}`;
}

function groupCurves(points) {
  const groups = new Map();
  for (const point of points) {
    const key = pathKey(point);
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(point);
  }
  return groups;
}

function renderAxisLoadings() {
  const axes = dataset?.projection?.axis_loadings || [];
  axisLoadings.replaceChildren(...axes.map((axis) => {
    const div = document.createElement("div");
    const features = axis.top_features
      .slice(0, 4)
      .map((item) => `${item.feature.replace("route_metrics.", "")} ${format(item.weight)}`)
      .join(", ");
    div.innerHTML = `<strong>PC${axis.axis}</strong> ${features}`;
    return div;
  }));
}

function framePoints(points) {
  if (!points.length) return;
  const box = new THREE.Box3();
  for (const point of points) {
    box.expandByPoint(point.view);
  }
  const center = box.getCenter(new THREE.Vector3());
  const size = Math.max(box.getSize(new THREE.Vector3()).length(), 1);
  orbit.target.copy(center);
  orbit.radius = size * 1.35;
  camera.near = Math.max(size / 1000, 0.01);
  camera.far = size * 20;
  camera.updateProjectionMatrix();
  updateCamera();
}

function handlePick(event) {
  if (!pointsObject || !dataset) return;
  if (orbit.dragging) return;
  const rect = canvas.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -(((event.clientY - rect.top) / rect.height) * 2 - 1);
  raycaster.setFromCamera(pointer, camera);
  const markerHit = raycaster.intersectObjects(eventGroup.children)[0];
  if (markerHit) {
    hoveredIndex = -1;
    showPoint(markerHit.object.userData.point);
    return;
  }
  const hit = raycaster.intersectObject(pointsObject)[0];
  if (hit && hit.index !== hoveredIndex) {
    hoveredIndex = hit.index;
    showPoint(pointsObject.userData.visiblePoints[hit.index]);
  }
}

function showPoint(point) {
  const metrics = point.metrics || {};
  const route = metrics.route_metrics || {};
  const rows = {
    event: point.event_kind || "trajectory",
    substrate: point.substrate,
    seed: point.seed,
    step: point.step,
    run: point.run_kind,
    candidate_id: point.candidate_id || point.substrate,
    perturb_family: point.perturb_family || "none",
    regime_class: point.regime_class || "unknown",
    perturb_scale: point.perturb_scale ?? "clean",
    path_distance: format(point.path_distance),
    residual_norm: format(metrics.residual_norm),
    residual_delta: format(metrics.residual_delta),
    temporal_coherence: format(metrics.temporal_coherence),
    fast_state_norm: format(metrics.fast_state_norm),
    slow_state_norm: format(metrics.slow_state_norm),
    message_state_norm: format(metrics.message_state_norm),
    release_open: format(route.release_open_mean),
    release_strength: format(route.release_strength_mean),
    release_pressure: format(route.release_pressure_mean),
    release_drive: format(route.release_drive_mean),
    release_bias: format(route.release_bias_norm),
    carrier_residual: format(route.carrier_residual_norm),
    message_signature: format(route.message_signature_tail),
    carrier_signature: format(route.carrier_signature_tail),
    release_metric: controls.releaseMetric.value,
    release_metric_value: format(releaseValue(point)),
    release_window_hit: isReleasePoint(point) ? "yes" : "no",
    release_transfer: format(route.release_transfer),
    release_pre_accumulation: format(route.release_pre_accumulation),
    release_local_displacement: format(route.release_local_displacement),
    release_local_signature_push: format(route.release_local_signature_push),
    release_local_causality: format(route.release_local_causality),
    first_release_step: format(route.first_release_step),
    release_timing_score: format(route.release_timing_score),
    eligible_release: format(route.release_eligible_fraction),
    early_release: format(route.release_early_fraction),
    internal_richness: format(route.internal_richness),
    geometric_coherence: format(route.geometric_coherence),
  };
  inspectorFields.replaceChildren(...Object.entries(rows).flatMap(([key, value]) => {
    const dt = document.createElement("dt");
    dt.textContent = key;
    const dd = document.createElement("dd");
    dd.textContent = String(value);
    return [dt, dd];
  }));
}

function showMessage(message) {
  inspectorFields.replaceChildren();
  const dt = document.createElement("dt");
  dt.textContent = "status";
  const dd = document.createElement("dd");
  dd.textContent = message;
  inspectorFields.append(dt, dd);
}

function cleanKey(point) {
  return `${point.substrate}|${point.seed}|${point.step}`;
}

function originKey(point) {
  return `${point.substrate}|${point.seed}`;
}

function pathKey(point) {
  return `${point.substrate}|${point.seed}|${point.run_kind}|${point.perturb_scale ?? "clean"}|${point.perturb_family ?? "none"}|${point.motif_index ?? 0}`;
}

function format(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(5) : "0.00000";
}

function formatThreshold(value) {
  if (!Number.isFinite(value)) return "0";
  if (value < 0.01) return value.toFixed(4);
  return value.toFixed(3);
}

function clearObject(object) {
  if (!object) return;
  scene.remove(object);
  object.geometry?.dispose();
  object.material?.dispose();
}

function clearGroup(group) {
  while (group.children.length) {
    const child = group.children.pop();
    child.geometry?.dispose();
    child.material?.dispose();
    child.cone?.material?.dispose();
    child.line?.material?.dispose();
  }
}

function resize() {
  const rect = canvas.parentElement.getBoundingClientRect();
  renderer.setSize(rect.width, rect.height, false);
  camera.aspect = rect.width / Math.max(rect.height, 1);
  camera.updateProjectionMatrix();
  if (viewData) {
    drawReleaseChart(viewData.points, viewData.releasePoints);
    drawRecoveryChart(viewData.points);
    drawPhaseChart(viewData.points);
  }
}

function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}

function updateCamera() {
  const cosPitch = Math.cos(orbit.pitch);
  camera.position.set(
    orbit.target.x + orbit.radius * Math.sin(orbit.yaw) * cosPitch,
    orbit.target.y + orbit.radius * Math.sin(orbit.pitch),
    orbit.target.z + orbit.radius * Math.cos(orbit.yaw) * cosPitch,
  );
  camera.lookAt(orbit.target);
}
