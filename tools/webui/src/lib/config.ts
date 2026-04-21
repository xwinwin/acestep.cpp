// UI constants
export const PROPS_POLL_MS = 10000;
export const FETCH_TIMEOUT_MS = 2000;
export const JOB_POLL_MS = 2000;
export const SSE_RECONNECT_MS = 2000;
export const LOG_MAX_LINES = 50;
export const WAVEFORM_HEIGHT = 64;

// task types (mirrors task-types.h)
export const TASK_TEXT2MUSIC = 'text2music';
export const TASK_COVER = 'cover';
export const TASK_COVER_NOFSQ = 'cover-nofsq';
export const TASK_REPAINT = 'repaint';
export const TASK_LEGO = 'lego';
export const TASK_EXTRACT = 'extract';
export const TASK_COMPLETE = 'complete';

// inference method (mirrors task-types.h INFER_*)
export const INFER_ODE = 'ode';
export const INFER_SDE = 'sde';

// DCW modes (mirrors task-types.h DCW_MODE_*)
export const DCW_MODE_LOW = 'low';
export const DCW_MODE_HIGH = 'high';
export const DCW_MODE_DOUBLE = 'double';
export const DCW_MODE_PIX = 'pix';

export const TRACK_NAMES = [
	'vocals',
	'backing_vocals',
	'drums',
	'bass',
	'guitar',
	'keyboard',
	'percussion',
	'strings',
	'synth',
	'fx',
	'brass',
	'woodwinds'
] as const;
