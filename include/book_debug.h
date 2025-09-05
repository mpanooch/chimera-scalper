// include/book_debug.h
#pragma once
#include "shared_book.h"
#include <cstdio>

#ifdef CHIMERA_DEBUG_BOOK
inline void dump_once_writer(const ChiSharedBook* b) {
    static bool done = false; if (done) return; done = true;
    std::fprintf(stderr, "[WRITER] sizeof(ChiSharedBook)=%zu off(bids)=%zu off(asks)=%zu\n",
        sizeof(ChiSharedBook),
        offsetof(ChiSharedBook, bids),
        offsetof(ChiSharedBook, asks));
}
#endif
