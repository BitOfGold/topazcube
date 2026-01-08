/*
┌─────────┬─────────┬──────┬───────────┬─────────────┬──────────────────┐
│ Color   │ Example │ Text │ Background│ Bright Text │ Bright Background│
├─────────┼─────────┼──────┼───────────┼─────────────┼──────────────────┤
│ Black   │ Black   │ 30   │ 40        │ 90          │ 100              │
│ Red     │ Red     │ 31   │ 41        │ 91          │ 101              │
│ Green   │ Green   │ 32   │ 42        │ 92          │ 102              │
│ Yellow  │ Yellow  │ 33   │ 43        │ 93          │ 103              │
│ Blue    │ Blue    │ 34   │ 44        │ 94          │ 104              │
│ Magenta │ Magenta │ 35   │ 45        │ 95          │ 105              │
│ Cyan    │ Cyan    │ 36   │ 46        │ 96          │ 106              │
│ White   │ White   │ 37   │ 47        │ 97          │ 107              │
│ Default │         │ 39   │ 49        │ 99          │ 109              │
└─────────┴─────────┴──────┴───────────┴─────────────┴──────────────────┘

┌───────────┬─────┬─────┐
│ Effect    │ On  │ Off │
├───────────┼─────┼─────┤
│ Bold      │ 1   │ 21  │
│ Dim       │ 2   │ 22  │
│ Underline │ 4   │ 24  │
│ Blink     │ 5   │ 25  │
│ Reverse   │ 7   │ 27  │
│ Hide      │ 8   │ 28  │
└───────────┴─────┴─────┘
*/

// Terminal control code constants
export const T_RESET = 0;
export const T_BOLD = 1;
export const T_DIM = 2;
export const T_UNDERLINE = 4;
export const T_BLINK = 5;
export const T_REVERSE = 7;
export const T_HIDE = 8;
export const T_BOLD_OFF = 21;
export const T_DIM_OFF = 22;
export const T_UNDERLINE_OFF = 24;
export const T_BLINK_OFF = 25;
export const T_REVERSE_OFF = 27;
export const T_HIDE_OFF = 28;

export const T_FG_BLACK = 30;
export const T_FG_RED = 31;
export const T_FG_GREEN = 32;
export const T_FG_YELLOW = 33;
export const T_FG_BLUE = 34;
export const T_FG_MAGENTA = 35;
export const T_FG_CYAN = 36;
export const T_FG_WHITE = 37;
export const T_FG_DEFAULT = 39;

export const T_BG_BLACK = 40;
export const T_BG_RED = 41;
export const T_BG_GREEN = 42;
export const T_BG_YELLOW = 43;
export const T_BG_BLUE = 44;
export const T_BG_MAGENTA = 45;
export const T_BG_CYAN = 46;
export const T_BG_WHITE = 47;
export const T_BG_DEFAULT = 49;

export const T_FG_BRIGHT_BLACK = 90;
export const T_FG_BRIGHT_RED = 91;
export const T_FG_BRIGHT_GREEN = 92;
export const T_FG_BRIGHT_YELLOW = 93;
export const T_FG_BRIGHT_BLUE = 94;
export const T_FG_BRIGHT_MAGENTA = 95;
export const T_FG_BRIGHT_CYAN = 96;
export const T_FG_BRIGHT_WHITE = 97;
export const T_FG_BRIGHT_DEFAULT = 99;

export const T_BG_BRIGHT_BLACK = 100;
export const T_BG_BRIGHT_RED = 101;
export const T_BG_BRIGHT_GREEN = 102;
export const T_BG_BRIGHT_YELLOW = 103;
export const T_BG_BRIGHT_BLUE = 104;
export const T_BG_BRIGHT_MAGENTA = 105;
export const T_BG_BRIGHT_CYAN = 106;
export const T_BG_BRIGHT_WHITE = 107;
export const T_BG_BRIGHT_DEFAULT = 109;


// Set terminal title
export function setTitle(title) {
  process.stdout.write("\x1b]0;" + title + "\x07");
}

// Set terminal color / effect
export function setColor(color) {
  process.stdout.write(`\x1b[${color}m`);
}

// Reset terminal color
export function resetColor() {
  setColor(T_RESET);
}

// Clear line
export function clearLine() {
  process.stdout.write("\x1b[2K");
}

// Erase to end of line
export function eraseEOL() {
  process.stdout.write("\x1b[K");
}

// Clear screen
export function clear() {
  process.stdout.write("\x1b[2J");
}

// Set cursor position
export function cursorPosition(line, column) {
  process.stdout.write(`\x1b[${line};${column}H`);
}
// Move cursor up
export function cursorUp(lines = 1) {
  process.stdout.write(`\x1b[${lines}A`);
}
// Move cursor down
export function cursorDown(lines = 1) {
  process.stdout.write(`\x1b[${lines}B`);
}

// Move cursor forward
export function cursorForward(columns = 1) {
  process.stdout.write(`\x1b[${columns}C`);
}
// Move cursor backward
export function cursorBackward(columns = 1) {
  process.stdout.write(`\x1b[${columns}D`);
}

// Save cursor position
export function saveCursor() {
  process.stdout.write("\x1b[s");
}

// Restore cursor position
export function restoreCursor() {
  process.stdout.write("\x1b[u");
}
