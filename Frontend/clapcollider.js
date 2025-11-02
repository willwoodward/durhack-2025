class ClapCollider {
  constructor() {
    this.events = new Map();
    this.patterns = new Map();
    this.change = false;
    this.count = 0;
    this.started = false;
  }

  handleEvent(event, onset, offset) {
    if (this.events.has(event)) {
      this.events.get(event).push([onset, offset]);
    } else {
      this.events.set(event, [[onset, offset]]);
    }
  }

  instruments = {
      clap: "bd",
      left_hand_upper: "sd",
      right_hand_upper: "hh",
  };

  get_instrument(event) {
    return this.instruments[event];
  }

  processSequence(event) {
    const sequence = this.events.get(event);
    this.events.delete(event);
    const clipped = [];
    const start = sequence[0][0];
    let running = 0;
    let count = 0;
    while (running <= (4000 - this.count) && count < sequence.length) {
      clipped.push(sequence[count]);
      running = sequence[count][1] - start;
      count++;
    }
    const num_parts = 16;
    const unit = 4 / num_parts;
    const onsets = clipped.map(([s, _]) => Math.round((s - start + (this.count/1000)) / unit));
    const pattern = new Array(num_parts).fill("~", 0, num_parts);
    for (let i = 0; i < num_parts; i++) {
      if (onsets.includes(i)) {
        pattern[i] = this.get_instrument(event);
      }
    }
    this.patterns.set(event, pattern.join(" "));
    this.change = true;
  }

  mainLoop() {
    if (this.started) {
      if (this.count >= 2000) this.count = 0;
      this.count += 100;
      document.getElementById("loop-inner").style.width = `${this.count / 20}%`
    }
    this.events.forEach((value, key) => {
      if (Object.keys(this.instruments).includes(key)) {
        const now = new Date().getTime() / 1000;
        if (value.length && now - value[value.length - 1][1] > 2) {
          this.processSequence(key);
        }
      }
    });
    if (this.change) {
      this.change = false;
      const strings = [];
      this.patterns.forEach((pattern, _) => {
        strings.push(`[${pattern}]`);
      });
      cps(0.25);
      s(strings.join(",")).play();
      if (!this.started) {
        this.started = true;
      }
    }
  }

  start() {
    setInterval(this.mainLoop.bind(this), 100);
  }

  stop() {
    hush();
    this.patterns = new Map();
    this.count = 0;
    this.started = false;
  }
}

initStrudel({
  prebake: () => samples("github:tidalcycles/dirt-samples"),
});
