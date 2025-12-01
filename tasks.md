# 🗂️ Tess AI Assistant – Roadmap

---

## 1️⃣ **Project Foundation**

### Objectives:

* Build a modular, scalable voice assistant
* Support natural language commands
* Maintain session and core memory

### Tasks:

* [x] Set up project structure (`src/`, `plugins/`, `core/`, `tests/`)
* [x] Initialize Python environment
* [x] Install dependencies: `spaCy`, `pyaudio`, `keyboard`, `SoundVolumeView`, etc.
* [x] Load SpaCy model (`en_core_web_sm`)

---

## 2️⃣ **Natural Language Processing**

### Objectives:

* Simplify raw user input
* Extract verbs, objects, and entities
* Convert messy queries into canonical commands

### Tasks:

* [x] Implement **text simplifier** using SpaCy (lemmatization + stopword removal)
* [x] Design **canonical phrase mapping** (rule-based matcher)
* [x] Test simplifier on common media/system commands
* [ ] Prepare **future slot-filling hooks** for multi-turn tasks

---

## 3️⃣ **Memory System**

### Objectives:

* Store user context
* Maintain session and long-term preferences

### Tasks:

* [ ] Implement **Session Memory** (RAM dict)
* [ ] Implement **Core Memory** (persistent storage: JSON/SQLite)
* [ ] Add **last referenced entity** for pronoun resolution
* [ ] Implement **memory update rules** for preference learning

---

## 4️⃣ **Dialogue Manager & Context**

### Objectives:

* Enable multi-turn conversations
* Handle confirmations, follow-ups, and slot filling

### Tasks:

* [ ] Build **Dialogue Manager class** (track states: idle, awaiting_confirmation, awaiting_slot)
* [ ] Integrate **Context Manager** for active intent, expected slots, and filled slots
* [ ] Implement **slot-filling logic** with SpaCy
* [ ] Add **pronoun resolution** using session memory
* [ ] Support **multi-turn tasks** like: “set volume” → “to 15”

---

## 5️⃣ **Plugin Architecture**

### Objectives:

* Modularize functionality
* Map canonical intents to skill modules

### Tasks:

* [ ] Create **Plugin Base Class**
* [ ] Implement **Intent Router / Mapper**
* [ ] Build **registration hub** for plugins
* [ ] Implement example plugins:

  * Media Control (`play`, `pause`, `next`, `previous`)
  * Volume Control (`volume_up`, `volume_down`)
  * System Control (`shutdown`, `restart`)
* [ ] Connect **Dialogue Manager → Intent → Plugin Executor**

---

## 6️⃣ **Input & Execution Layer**

### Objectives:

* Capture voice commands
* Convert STT → simplified text → execute intent

### Tasks:

* [ ] Implement **wake word detection** (Porcupine)
* [ ] Integrate **Speech-to-Text engine** (Vosk/Whisper)
* [ ] Implement **text input fallback**
* [ ] Build **execution module** for media/system commands

---

## 7️⃣ **Preference Learning & Dynamic Canonical Expansion (Optional)**

### Objectives:

* Make Tess smarter over time
* Track user habits and expand command coverage

### Tasks:

* [ ] Track **frequency of intent usage**
* [ ] Update **default preferences** automatically (volume, brightness, default apps)
* [ ] Add **dynamic canonical phrase mapping** (new synonyms from repeated use)

---

## 8️⃣ **Error Handling & Safety**

### Objectives:

* Prevent accidental destructive actions
* Provide user-friendly feedback

### Tasks:

* [ ] Implement **confirmation system** for critical commands
* [ ] Handle **unknown commands gracefully**
* [ ] Add **fallback responses**
* [ ] Log execution errors

---

## 9️⃣ **Personality & Voice (Optional)**

### Objectives:

* Give Tess a consistent personality
* Add TTS output

### Tasks:

* [ ] Define **personality traits** (tone, verbosity, style)
* [ ] Integrate **TTS engine** (pyttsx3, ElevenLabs, or gTTS)
* [ ] Make responses context-aware and expressive

---

## 🔟 **Advanced Features (Future / Optional)**

* [ ] Multi-entity pronoun tracking (“turn them off”)
* [ ] Focus stack for multi-step sessions
* [ ] Async event pipeline (hotword → STT → NLP → plugin execution)
* [ ] Background tasks: timers, reminders, downloads
* [ ] Logging & analytics dashboard

---

## ✅ **Deliverable / MVP (Minimum Viable Product)**

* NLP simplifier working with canonical matcher
* Dialogue manager + context manager with slot-filling & pronoun resolution
* Session + core memory system
* Plugin system with at least 3 skills (media, volume, system)
* Wake-word detection and STT pipeline
* Basic execution flow with confirmations

---
