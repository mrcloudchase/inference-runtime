#include <stdbool.h>
#include "_cgo_export.h"
#include "bridge.h"

static bool streamBridge(const char* token, void* user_data) {
    return goStreamCallback((char*)token, user_data);
}

IRStreamCallback get_go_stream_callback(void) {
    return streamBridge;
}
