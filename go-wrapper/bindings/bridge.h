#ifndef IR_BRIDGE_H
#define IR_BRIDGE_H

#include "ir_runtime.h"

// C wrapper that calls into Go. Implemented in bridge.c.
IRStreamCallback get_go_stream_callback(void);

#endif
