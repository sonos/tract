// Page-side handle on worker.js, where tract tracks the hands.
export function HandClient(onProgress = () => {}) {
  const stamp = new URL(import.meta.url).searchParams.get('stamp');
  const url = new URL(`./worker.js${stamp ? `?stamp=${stamp}` : ''}`, import.meta.url);
  const worker = new Worker(url, {type: 'module'});
  let nextId = 1;
  let failed = null;
  const pending = new Map();
  const fail = error => {
    failed = error;
    for (const wait of pending.values()) {
      clearTimeout(wait.timer);
      wait.reject(error);
    }
    pending.clear();
  };
  worker.onmessage = ({data}) => {
    if (data.progress) {
      onProgress(data.progress.text, data.progress.fraction);
      return;
    }
    const wait = pending.get(data.id);
    if (!wait) return;
    pending.delete(data.id);
    clearTimeout(wait.timer);
    if (data.error) wait.reject(new Error(data.error));
    else wait.resolve(data.result);
  };
  worker.onerror = event => fail(new Error(event.message || 'The hand worker failed.'));
  return {
    request(type, bitmap, timeout = 20000, extra = {}) {
      if (failed) return Promise.reject(failed);
      const id = nextId++;
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
          pending.delete(id);
          reject(new Error(`${type} timed out.`));
        }, timeout);
        pending.set(id, {resolve, reject, timer});
        worker.postMessage({...extra, id, type, bitmap}, bitmap ? [bitmap] : []);
      });
    },
    close() {
      worker.terminate();
    },
  };
}
