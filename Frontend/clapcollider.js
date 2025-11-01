class ClapCollider {
  constructor() {
    this.events = new Map();
    this.patterns = new Map();
    this.change = false;
  }

  handleEvent(event, onset, offset) {
    if (this.events.has(event)) {
      this.events.get(event).push([onset, offset]);
    } else {
      this.events.set(event, [[onset, offset]]);
    }
  }

  get_instrument(event) {
    const instruments = {
      clap: "bd",
      snare: "sd",
      hihat: "hh",
    };
    return instruments[event];
  }

  processSequence(event) {
    const sequence = this.events.get(event);
    console.log(sequence)
    this.events.delete(event);
    const clipped = [];
    const start = sequence[0][0];
    let running = 0;
    let count = 0;
    // A bar in 4/4 120bpm
    while (running <= 2000 && count < sequence.length) {
      clipped.push(sequence[count]);
      running = sequence[count][1] - start;
      count++;
    }
    const num_parts = 16;
    const unit = running / num_parts;
    const onsets = clipped.map(
      ([s, _]) => (Math.round((s - start) / unit) * unit) / unit
    );
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
    this.events.forEach((value, key) => {
      const now = new Date().getTime() / 1000;
      if (value.length && now - value[value.length - 1][1] > 1) {
        this.processSequence(key);
      }
    });
    if (this.change) {
      this.change = false;
      const strings = [];
      this.patterns.forEach((pattern, _) => {
        strings.push(`[${pattern}]`);
      });
      cps(0.5);
      s(strings.join(",")).play();
    }
  }

  start() {
    setInterval(this.mainLoop.bind(this), 100);
  }
}

initStrudel({
  prebake: () => samples("github:tidalcycles/dirt-samples"),
});
