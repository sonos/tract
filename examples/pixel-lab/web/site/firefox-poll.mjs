// Firefox completes mapAsync and onSubmittedWorkDone from a 100 ms timer
// (WebGPUParent::POLL_TIME_MS → MaintainDevices → wgpu_server_poll_all_devices).
// wgpu-core's queue submit ends in a non-blocking device poll, so an empty
// queue.submit([]) while a completion is outstanding flushes it. Measured on
// Firefox 155: idle host completion stays on that 100 ms tick; empty submits
// move it to the submit cadence. Chrome and Safari already complete without this.

export const FIREFOX_POLL_START_MS = 0.5;
export const FIREFOX_POLL_MAX_MS = 8;

export function isFirefox(ua) {
  return typeof ua === 'string' && /Firefox\/\d/.test(ua);
}

// First kick waits half a millisecond, then the gap doubles up to 8 ms.
export function nextPollDelay(previous) {
  if (!(previous > 0)) return FIREFOX_POLL_START_MS;
  return Math.min(previous * 2, FIREFOX_POLL_MAX_MS);
}

// One kicker per device. Several outstanding completions share it: one poll
// flushes every callback on that queue. The timer stops when none remain.
export function createCompletionPoll({firefox, device, schedule, clear}) {
  let outstanding = 0;
  let timer = 0;
  let delay = 0;

  function disarm() {
    if (!timer) return;
    clear(timer);
    timer = 0;
    delay = 0;
  }

  function arm() {
    if (!firefox || timer || outstanding <= 0) return;
    delay = nextPollDelay(delay);
    const wait = delay;
    timer = schedule(() => {
      timer = 0;
      if (outstanding <= 0) {
        delay = 0;
        return;
      }
      try {
        device.queue.submit([]);
      } catch {
        delay = 0;
        return;
      }
      if (outstanding <= 0) {
        delay = 0;
        return;
      }
      arm();
    }, wait);
  }

  return {
    track(promise) {
      if (!firefox) return promise;
      outstanding += 1;
      if (!timer) {
        delay = 0;
        arm();
      }
      let settled = false;
      const release = () => {
        if (settled) return;
        settled = true;
        outstanding -= 1;
        if (outstanding <= 0) {
          outstanding = 0;
          disarm();
        }
      };
      return promise.then(value => {
        release();
        return value;
      }, reason => {
        release();
        throw reason;
      });
    },
  };
}
