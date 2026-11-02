const assert = require("node:assert/strict");
const test = require("node:test");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const intervals = [];
const elements = new Map();
const context = vm.createContext({
  document: {
    getElementById: id => {
      if (!elements.has(id)) elements.set(id, {addEventListener() {}});
      return elements.get(id);
    },
    querySelector: () => ({content: "20"}),
    querySelectorAll: () => [],
    addEventListener() {},
  },
  window: {addEventListener() {}},
  fetch: () => new Promise(() => {}),
  setInterval: (callback, interval) => intervals.push(interval),
});
// 加载实际网页脚本；网络与DOM隔离，直接验证预览与状态调度是否解耦。
vm.runInContext(fs.readFileSync(path.join(__dirname, "../src/piperx_expo/client/web/app.js"), "utf8"), context);
const LatestFramePreview = vm.runInContext("LatestFramePreview", context);

test("预览20Hz独立于600ms状态轮询", () => {
  assert.ok(intervals.includes(50));
  assert.ok(intervals.includes(600));
});

function image() {
  const handlers = {};
  return {
    addEventListener: (name, handler) => { handlers[name] = handler; },
    emit: (name) => handlers[name](),
  };
}

test("慢相机不积压请求，也不阻塞另一相机", () => {
  const images = [image(), image()];
  const calls = [];
  const preview = new LatestFramePreview(images, (_, camera) => calls.push(camera));
  preview.refresh(true);
  for (let i = 0; i < 20; i++) preview.refresh(true);
  assert.deepEqual(calls, [0, 1]);
  images[1].emit("load");
  preview.refresh(true);
  assert.deepEqual(calls, [0, 1, 1]);
  images[0].emit("error");
  preview.refresh(true);
  assert.deepEqual(calls, [0, 1, 1, 0]);
});

test("历史回放和页面隐藏暂停预览，恢复后继续", () => {
  const images = [image(), image()];
  let calls = 0;
  const preview = new LatestFramePreview(images, () => { calls++; });
  preview.refresh(false);
  assert.equal(calls, 0);
  preview.refresh(true);
  images.forEach(img => img.emit("load"));
  preview.refresh(false);
  assert.equal(calls, 2);
  preview.refresh(true);
  assert.equal(calls, 4);
});
