"use strict";
const $ = (id) => document.getElementById(id);
const token = document.querySelector('meta[name="expo-token"]').content;
let current = null,
  selected = null,
  history = [],
  first = true,
  lastEpisode = null,
  replayFrame = 0,
  polling = false;
const phaseNames = {
  idle: "准备就绪",
  connecting: "等待学习器",
  awaiting_reset: "等待现场复位",
  running: "正在执行",
  review_pending: "等待回合复核",
  submitted: "等待服务器确认",
  waiting_server: "等待服务器",
  stopped: "已停止并锁定",
  fault: "异常停止",
  completed: "会话已完成",
};
const statusNames = {
  recording: "采集中",
  review_pending: "待复核",
  submitted: "已提交 · 待确认",
  accepted: "已加入经验",
  rejected: "已拒收",
  aborted: "中断归档",
  demo_reviewed: "模拟已复核",
};
function notice(message, error = false) {
  $("notice").textContent = message;
  $("notice").className = "notice" + (error ? " error" : "");
  $("notice").style.display = message ? "block" : "none";
}
async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: { "X-Expo-Token": token, ...options.headers },
  });
  const data = await response.json();
  if (!response.ok) throw Error(data.error || "请求失败");
  return data;
}
async function command(action, values = {}) {
  try {
    await api("/api/command/" + action, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Expo-Token": token },
      body: JSON.stringify(values),
    });
    notice("");
    await poll();
    return true;
  } catch (error) {
    notice(error.message, true);
    return false;
  }
}
function view(name) {
  document
    .querySelectorAll(".view")
    .forEach((e) => (e.hidden = e.id !== "view-" + name));
  document
    .querySelectorAll(".nav")
    .forEach((e) => e.classList.toggle("active", e.dataset.view === name));
  $("page-title").textContent = {
    run: "训练，从每一条轨迹开始。",
    history: "每次尝试，都值得回看。",
    settings: "为下一次尝试做好准备。",
  }[name];
  $("subtitle").textContent = {
    run: "观察执行，复核结果，让每次尝试成为可靠的经验。",
    history: "原始自动判断、人工修正和经验收录状态，一起保留。",
    settings: "调整采集预算，确认连接与现场判定规则。",
  }[name];
  if (name === "run") draw();
}
document
  .querySelectorAll(".nav")
  .forEach((e) => (e.onclick = () => view(e.dataset.view)));
$("all-history").onclick = () => view("history");
$("open-settings").onclick = (e) => {
  e.preventDefault();
  view("settings");
};
$("start").onclick = () =>
  command("start", { motion_confirmed: $("motion-confirm").checked });
$("demo").onclick = () => command("start", { demo: true });
$("ready").onclick = () => {
  if (current.backend === "piperx" && !$("motion-confirm").checked) {
    notice("请先勾选现场预检与运动确认。", true);
    return;
  }
  command("ready");
};
$("stop").onclick = () => command("stop");
$("end").onclick = () => command("end");
$("review-form").onsubmit = (e) => {
  e.preventDefault();
  const v = document.querySelector('input[name="manual"]:checked').value;
  command("review", {
    manual_success: v === "auto" ? null : v === "success",
    admitted: $("admitted").checked,
    note: $("note").value,
  });
};
$("settings-form").onsubmit = async (e) => {
  e.preventDefault();
  const values = {
    episode_steps: Number($("max-steps").value),
    admission_mode: $("admission-mode").value,
  };
  if (current?.phase === "idle") {
    values.server_host = $("server-host").value.trim();
    values.server_port = Number($("server-port").value);
  }
  if (await command("settings", values))
    notice("参数已保存，将在下一条轨迹开始时生效。");
};
function loadImage(img, url) {
  img.onload = () => {
    img.style.display = "block";
    img.parentElement.querySelector(".camera-empty").style.display = "none";
  };
  img.src = url;
}
function images() {
  if (selected) {
    for (let c = 0; c < 2; c++)
      loadImage(
        $("camera-" + c),
        `/api/episode/${selected.id}/frame?camera=${c}&index=${replayFrame}`,
      );
  } else if (current?.state) {
    for (let c = 0; c < 2; c++)
      loadImage($("camera-" + c), `/api/frame?camera=${c}&t=${Date.now()}`);
  }
}
$("scrubber").oninput = () => {
  if (!selected && current?.episode?.status !== "recording") {
    selected = current.episode;
  }
  if (selected) {
    replayFrame = Number($("scrubber").value);
    $("frame-label").textContent = `${replayFrame} / ${selected.steps}`;
    images();
    draw();
  }
};
$("live").onclick = () => {
  selected = null;
  document.querySelector('input[name="manual"][value="auto"]').checked = true;
  $("admitted").checked = true;
  $("note").value = "";
  $("frame-label").textContent = "LIVE";
  images();
  draw();
};
function draw() {
  const canvas = $("trajectory"),
    box = canvas.getBoundingClientRect();
  if (!box.width) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = box.width * dpr;
  canvas.height = box.height * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const w = box.width,
    h = box.height,
    L = 43,
    R = w - 35,
    T = 12,
    B = h - 20;
  ctx.font = "8px monospace";
  ctx.fillStyle = "#a4b0b2";
  const points = (selected || current?.episode)?.points || [];
  const range = points.length ? points.flatMap((p) => p.slice(0, 3)) : [];
  let lo = Math.min(0, ...range),
    hi = Math.max(0.4, ...range);
  hi += 0.02;
  lo -= 0.02;
  for (let i = 0; i < 4; i++) {
    const y = T + ((B - T) * i) / 3;
    ctx.beginPath();
    ctx.strokeStyle = "#edf1ef";
    ctx.moveTo(L, y);
    ctx.lineTo(R, y);
    ctx.stroke();
    ctx.fillText((hi - ((hi - lo) * i) / 3).toFixed(2), 9, y + 3);
    ctx.fillText((80 * (1 - i / 3)).toFixed(0), R + 8, y + 3);
  }
  ctx.fillText("m", 9, 9);
  ctx.fillText("mm", R + 8, 9);
  const colors = ["#299886", "#658daf", "#d6aa55", "#a68bbd"];
  [0, 1, 2, 6].forEach((dim, j) => {
    ctx.beginPath();
    ctx.strokeStyle = colors[j];
    ctx.lineWidth = 1.7;
    points.forEach((p, i) => {
      const x = L + ((R - L) * i) / Math.max(1, points.length - 1),
        v = dim === 6 ? p[dim] / 0.08 : (p[dim] - lo) / (hi - lo),
        y = B - (B - T) * v;
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  });
  if (!points.length) {
    ctx.fillStyle = "#a4b0b2";
    ctx.textAlign = "center";
    ctx.fillText("轨迹开始后显示实测曲线", w / 2, h / 2);
    ctx.textAlign = "start";
  }
  if (selected) {
    const x = L + ((R - L) * replayFrame) / Math.max(1, points.length - 1);
    ctx.strokeStyle = "#627d72";
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(x, T);
    ctx.lineTo(x, B);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.fillStyle = "#a4b0b2";
  ctx.fillText("0", L, B + 14);
  ctx.fillText(`${Math.max(0, points.length - 1)} 步`, R - 24, B + 14);
}
window.addEventListener("resize", draw);
function render(s) {
  current = s;
  const ep = s.episode,
    phase = s.phase;
  $("phase").textContent = phaseNames[phase] || phase;
  $("mode").textContent = s.backend === "mock" ? "模拟环境" : "PIPER-X · 真机";
  $("connection").textContent = s.demo
    ? "模拟会话"
    : s.connected
      ? "学习器已连接"
      : "学习器未连接";
  $("connection-dot").style.background = s.connected ? "#299886" : "#adb8b3";
  $("steps").textContent = ep?.steps || 0;
  $("budget").textContent =
    ep?.settings?.episode_steps || s.settings.episode_steps;
  $("hz").textContent = s.settings.control_hz + " Hz";
  $("task").textContent = s.settings.language_instruction;
  $("progress-bar").style.width =
    Math.min(100, ((ep?.steps || 0) / Number($("budget").textContent)) * 100) +
    "%";
  $("review-fields").disabled = phase !== "review_pending" || !!selected;
  $("review-badge").textContent = ep
    ? statusNames[ep.status] || ep.status
    : "等待轨迹";
  $("start").hidden = phase !== "idle";
  $("demo").hidden = s.backend !== "mock" || phase !== "idle";
  $("ready").hidden = phase !== "awaiting_reset";
  $("ready").disabled = !!selected;
  $("end").hidden = phase !== "running";
  $("stop").disabled = ["idle", "stopped", "fault", "completed"].includes(
    phase,
  );
  $("motion-label").hidden = s.backend !== "piperx";
  $("server-host").disabled = phase !== "idle";
  $("server-port").disabled = phase !== "idle";
  if (first) {
    $("admission-mode").value = s.settings.admission_mode;
    $("max-steps").value = s.settings.episode_steps;
    $("server-host").value = s.settings.server_host;
    $("server-port").value = s.settings.server_port;
    const d = s.detector;
    [
      ["高度阈值", d ? `${d.height_m} m` : "未配置"],
      [
        "夹爪宽度区间",
        d
          ? d.gripper_width_m.map((x) => (x * 1000).toFixed(1)).join(" – ") +
            " mm"
          : "未配置",
      ],
      ["连续实际动作步", d ? d.consecutive_steps + " 步" : "未配置"],
      ["控制频率", s.settings.control_hz + " Hz"],
    ].forEach(([k, v]) => {
      const dt = document.createElement("dt"),
        dd = document.createElement("dd");
      dt.textContent = k;
      dd.textContent = v;
      $("detector-details").append(dt, dd);
    });
    first = false;
  }
  if (ep && ep.id !== lastEpisode) {
    lastEpisode = ep.id;
    document.querySelector('input[name="manual"][value="auto"]').checked = true;
    $("admitted").checked = true;
    $("note").value = "";
    selected = null;
  }
  const displayEp = selected || ep;
  const auto = displayEp?.auto_success;
  if (selected) {
    const label = selected.review?.manual_success;
    document.querySelector(
      `input[name="manual"][value="${label === true ? "success" : label === false ? "failure" : "auto"}"]`,
    ).checked = true;
    $("admitted").checked = selected.review?.admitted ?? false;
    $("note").value = selected.review?.note || "";
    $("review-badge").textContent =
      "回放 · " + (statusNames[selected.status] || selected.status);
  }
  $("auto-result").classList.toggle("success", auto === true);
  $("auto-title").textContent =
    auto === true
      ? "自动检出成功"
      : auto === false
        ? "未检出成功"
        : "尚无判定结果";
  $("auto-hint").textContent = selected
    ? "原始自动结果，与人工记录分别保留"
    : phase === "review_pending"
      ? "请结合画面复核，可手动修正"
      : auto !== undefined
        ? "已保存判定，收录后不能追溯改标"
        : "轨迹结束后可补充人工标记";
  $("auto-result").querySelector(".result-icon").textContent =
    auto === true ? "✓" : auto === false ? "—" : "◎";
  $("ack-hint").textContent = selected
    ? "正在回放历史档案；返回实时后可复核当前轨迹。"
    : ep?.status === "accepted"
      ? "服务器已确认入池，训练使用人工优先标签。"
      : ep?.status === "rejected"
        ? "服务器已确认拒收，仅保留本地档案。"
        : ep?.status === "demo_reviewed"
          ? "模拟复核已保存；未连接学习器，不会训练。"
          : ep?.status === "submitted"
            ? "等待服务器入池确认；请勿重复提交。"
            : "提交前不会写入学习器经验池。";
  const prompts = {
    idle: ["准备开始采集", "先启动复核模式服务器，再连接客户端。"],
    connecting: ["正在等待学习器", "模型准备完成后会请求创建环境。"],
    awaiting_reset: [
      "请人工复位并检查现场",
      "点击确认后，下一条轨迹才会开始执行。",
    ],
    running: ["正在采集轨迹", "可提前结束并复核；停止并锁定将中断本条。"],
    review_pending: [
      "执行完成，等待你的复核",
      "先检查自动结果，再选择最终标签与是否收录。",
    ],
    submitted: ["复核已提交", "等待学习器核验并确认本条经验。"],
    waiting_server: [
      "等待学习器准备下一回合",
      "模型更新结束后将提示现场复位。",
    ],
    stopped: ["会话已停止", "重新启动客户端服务后可建立新会话。"],
    fault: ["会话异常停止", "检查错误和现场状态后重新启动客户端。"],
    completed: [
      "本次采集与训练会话已完成",
      "已达到学习器预算；轨迹与复核结果保留在档案中。",
    ],
  };
  const prompt = prompts[phase] || ["", ""];
  $("action-title").textContent = prompt[0];
  $("action-hint").textContent = prompt[1];
  if (s.error) notice(s.error, true);
  if (!selected) {
    $("scrubber").max = ep?.steps || 0;
    $("scrubber").value = ep?.steps || 0;
    $("scrubber").disabled = !ep || ep.status === "recording";
    $("frame-label").textContent = "LIVE";
  }
  const state = selected?.points?.[replayFrame] || s.state;
  $("state-text").textContent = state
    ? `Z ${state[2].toFixed(3)} m · G ${(state[6] * 1000).toFixed(1)} mm`
    : "等待观测";
  draw();
}
function table(body, items) {
  body.replaceChildren();
  if (!items.length) {
    const tr = document.createElement("tr");
    tr.className = "empty-row";
    const td = document.createElement("td");
    td.colSpan = 6;
    td.textContent = "还没有轨迹。完成一次采集后，记录会出现在这里。";
    tr.append(td);
    body.append(tr);
    return;
  }
  items.forEach((ep) => {
    const tr = document.createElement("tr");
    const texts = [
      new Date(ep.created_at * 1000).toLocaleString("zh-CN", { hour12: false }),
      ep.steps,
      ep.auto_success === true
        ? "成功"
        : ep.auto_success === false
          ? "未检出"
          : "—",
      ep.review?.manual_success === true
        ? "成功"
        : ep.review?.manual_success === false
          ? "失败"
          : "未设置",
    ];
    texts.forEach((txt, i) => {
      const td = document.createElement("td");
      td.textContent = txt;
      if (i === 0) {
        const small = document.createElement("small");
        small.textContent = ep.id.slice(0, 12);
        td.append(small);
      }
      tr.append(td);
    });
    const status = document.createElement("td"),
      tag = document.createElement("span");
    tag.className = "status-tag " + ep.status;
    tag.textContent = statusNames[ep.status] || ep.status;
    status.append(tag);
    tr.append(status);
    const actions = document.createElement("td");
    if (ep.status !== "recording") {
      const play = document.createElement("button");
      play.className = "text-button";
      play.textContent = "回放 ↗";
      play.onclick = async () => {
        try {
          selected = await api(`/api/episode/${ep.id}/metadata`);
          replayFrame = 0;
          $("scrubber").max = ep.steps;
          $("scrubber").value = 0;
          $("scrubber").disabled = false;
          $("frame-label").textContent = `0 / ${ep.steps}`;
          view("run");
          render(current);
          images();
        } catch (e) {
          notice(e.message, true);
        }
      };
      const link = document.createElement("a");
      link.href = `/api/episode/${ep.id}/download`;
      link.textContent = " ↓";
      link.title = "下载轨迹HDF5";
      actions.append(play, link);
    }
    tr.append(actions);
    body.append(tr);
  });
}
async function refreshHistory() {
  try {
    history = await api("/api/history");
    $("history-count").textContent = history.length;
    table($("recent-body"), history.slice(0, 5));
    table($("history-body"), history);
  } catch (e) {
    notice(e.message, true);
  }
}
async function poll() {
  if (polling) return;
  polling = true;
  try {
    render(await api("/api/state"));
    if (!selected) images();
  } catch (error) {
    notice("无法连接本地客户端：" + error.message, true);
  } finally {
    polling = false;
  }
}
poll();
refreshHistory();
setInterval(poll, 600);
setInterval(refreshHistory, 3000);
