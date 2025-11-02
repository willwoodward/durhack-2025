class ClapCollider {
  constructor() {
    this.events = new Map();
    this.patterns = new Map();
    this.change = false;
    this.count = 0;
    this.started = false;
    this.strudelReady = false;  // Track if Strudel is initialized
    this.noteMode = false;  // Track if we're in note mode (synths) vs drum mode
    this.leftHandActive = false;  // Track if left hand has been detected
    this.rightHandActive = false;  // Track if right hand has been detected

    // Chord tracking for left hand
    this.sectionToChord = [
      "Bb-9",  // Section 0 (top)
      "C7",    // Section 1
      "Db^7",  // Section 2
      "Eb9",   // Section 3
      "F-9",   // Section 4
      "Gb^7",  // Section 5
      "Ab13",  // Section 6
      "Bb-9"   // Section 7 (bottom)
    ];
    this.leftHandChord = "Bb-9";
    this.chordChange = false;

    // Lead notes for right hand (MIDI note numbers)
    this.sectionToNote = [
      "bb5",  // Section 0 (top) - high Bb
      "ab5",  // Section 1 - Ab
      "gb5",  // Section 2 - Gb
      "f5",   // Section 3 - F
      "eb5",  // Section 4 - Eb
      "db5",  // Section 5 - Db
      "c5",   // Section 6 - C
      "bb4"   // Section 7 (bottom) - lower Bb
    ];
    this.rightHandNote = "bb4";
    this.leadChange = false;
  }

  setNoteMode(isNoteMode) {
    this.noteMode = isNoteMode;
    console.log(`🎼 Mode: ${isNoteMode ? 'SYNTHS (hand sections)' : 'DRUMS (claps/gestures)'}`);
    console.log(`   Both drums and synths continue playing in background`);
  }

  handleEvent(event, onset, offset, metadata) {
    // Handle hand section events (only in note mode)
    if (this.noteMode && metadata && metadata.section !== undefined) {
      const section = metadata.section;

      if (metadata.hand === "left") {
        // Left hand controls chords
        const chord = this.sectionToChord[section];
        if (this.leftHandChord !== chord) {
          this.leftHandChord = chord;
          this.chordChange = true;
          this.leftHandActive = true;  // Mark left hand as active
          console.log(`👋 Left Hand → Section ${section} → Chord: ${chord}`);
        }
      } else if (metadata.hand === "right") {
        // Right hand controls lead synth notes
        const note = this.sectionToNote[section];
        if (this.rightHandNote !== note) {
          this.rightHandNote = note;
          this.leadChange = true;
          this.rightHandActive = true;  // Mark right hand as active
          console.log(`🎹 Right Hand → Section ${section} → Note: ${note}`);
        }
      }
    }

    // Track events for drum patterns (only in drum mode)
    if (!this.noteMode) {
      if (this.events.has(event)) {
        this.events.get(event).push([onset, offset]);
      } else {
        this.events.set(event, [[onset, offset]]);
      }
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
    // Update ready status from global flag
    if (!this.strudelReady && window.strudelReady) {
      this.strudelReady = true;
      console.log("🎵 ClapCollider: Strudel is ready");
    }

    if (this.started) {
      if (this.count >= 2000) this.count = 0;
      this.count += 100;
      const loopInner = document.getElementById("loop-inner");
      if (loopInner) {
        loopInner.style.width = `${this.count / 20}%`;
      }
    }

    this.events.forEach((value, key) => {
      if (Object.keys(this.instruments).includes(key)) {
        const now = new Date().getTime() / 1000;
        if (value.length && now - value[value.length - 1][1] > 2) {
          this.processSequence(key);
        }
      }
    });

    // Play everything together when anything changes
    if ((this.change || this.chordChange || this.leadChange) && this.strudelReady) {
      const drumChange = this.change;
      const synthChange = this.chordChange || this.leadChange;

      this.change = false;
      this.chordChange = false;
      this.leadChange = false;

      try {
        const parts = [];

        // Add drums if we have patterns
        if (this.patterns.size > 0) {
          const strings = [];
          this.patterns.forEach((pattern, _) => {
            strings.push(`[${pattern}]`);
          });
          parts.push(s(strings.join(",")));
        }

        // Add synths based on which hands have been detected
        const synthParts = [];

        if (this.leftHandActive) {
          synthParts.push(
            chord(this.leftHandChord)
              .dict('ireal')
              .voicing()
              .sound("sawtooth")
              .cutoff(1000)
              .gain(0.3)
              .room(0.5)
          );
        }

        if (this.rightHandActive) {
          synthParts.push(
            note(this.rightHandNote)
              .sound("triangle")
              .cutoff(2000)
              .gain(0.5)
              .release(0.5)
              .room(0.3)
          );
        }

        if (synthParts.length > 0) {
          if (synthParts.length === 1) {
            parts.push(synthParts[0]);
          } else {
            parts.push(stack(...synthParts));
          }
        }

        // Play everything together
        if (parts.length > 0) {
          cps(0.25);
          stack(...parts).play();
        }

        if (!this.started && drumChange) {
          this.started = true;
        }
      } catch (e) {
        console.error("Error playing music:", e);
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
    this.change = false;
    this.chordChange = false;
    this.leadChange = false;
    this.leftHandActive = false;  // Reset left hand active flag
    this.rightHandActive = false;  // Reset right hand active flag
  }
}

// Initialize Strudel
initStrudel({
  prebake: () => samples("github:tidalcycles/dirt-samples"),
});

// Give Strudel a moment to initialize, then mark as ready
setTimeout(() => {
  console.log("✅ Strudel initialized");
  window.strudelReady = true;
}, 1000);
