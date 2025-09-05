// include/book_utils.h
#pragma once
#include "shared_book.h"
#include <cmath>

struct SpreadRead {
    bool   ok{false};
    double bid{0}, ask{0}, mid{0}, bps{0};
};

inline SpreadRead read_top_and_bps(const ChiSharedBook* book) {
    SpreadRead r{};
    const uint32_t d = book->hdr.depth;
    if (d == 0) return r;

    const double bid = book->bids[0].price;
    const double ask = book->asks[0].price;

    if (!std::isfinite(bid) || !std::isfinite(ask) || bid <= 0.0 || ask <= 0.0 || ask <= bid)
        return r;

    const double mid = 0.5 * (ask + bid);
    const double rel = (ask - bid) / mid;   // e.g., 0.0002 for 2 bps
    const double bps = rel * 1e4;

    // Reject obviously broken ticks (keeps console & AI clean)
    if (bps <= 0.0 || bps > 100.0) return r;

    r.ok  = true;
    r.bid = bid; r.ask = ask; r.mid = mid; r.bps = bps;
    return r;
}

// Optional: print struct offsets once
#ifdef CHIMERA_DEBUG_BOOK
#include <cstdio>
inline void book_offsets_once(const ChiSharedBook* book) {
    static bool done = false; if (done) return; done = true;
    std::fprintf(stderr, "[READ] sizeof(ChiSharedBook)=%zu off(bids)=%zu off(asks)=%zu depth=%u\n",
        sizeof(ChiSharedBook),
        offsetof(ChiSharedBook, bids),
        offsetof(ChiSharedBook, asks),
        book->hdr.depth
    );
    if (book->hdr.depth > 0) {
        std::fprintf(stderr, "[READ] %s bid0=%.8f x %.3f | ask0=%.8f x %.3f\n",
            book->hdr.symbol,
            book->bids[0].price, book->bids[0].qty,
            book->asks[0].price, book->asks[0].qty
        );
    }
}
#endif
